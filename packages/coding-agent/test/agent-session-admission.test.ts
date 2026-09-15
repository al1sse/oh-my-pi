/**
 * Atomic user-message admission (`AgentSession.admitUserMessage`).
 *
 * `sendUserMessage` always requests `streamingBehavior: "steer"`, so a submission
 * that races an in-flight run is silently merged into it: the call resolves, raises
 * nothing, and returns no identity. Admission is the strict sibling — it reserves
 * the session synchronously, refuses with a bounded reason instead of steering, and
 * returns the native entry id of the input it persisted, so the check-then-submit
 * race that motivated it stays closed.
 *
 * Every test drives real session state rather than mocking the decision: the
 * concurrency cases synchronize on explicit promises, never on elapsed time.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "bun:test";
import * as path from "node:path";
import { scheduler } from "node:timers/promises";
import { type } from "@oh-my-pi/omptype";
import { Agent, type AgentTool } from "@oh-my-pi/pi-agent-core";
import type { AssistantMessage, Context, Model } from "@oh-my-pi/pi-ai";
import { createMockModel } from "@oh-my-pi/pi-ai/providers/mock";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { ExtensionRuntime, loadExtensionFromFactory } from "@oh-my-pi/pi-coding-agent/extensibility/extensions/loader";
import { ExtensionRunner } from "@oh-my-pi/pi-coding-agent/extensibility/extensions/runner";
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent/extensibility/extensions/types";
import { initializeExtensions } from "@oh-my-pi/pi-coding-agent/modes/runtime-init";
import { AgentSession, type UserMessageAdmissionResult } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { convertToLlm } from "@oh-my-pi/pi-coding-agent/session/messages";
import type { SessionEntry } from "@oh-my-pi/pi-coding-agent/session/session-entries";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { EventBus } from "@oh-my-pi/pi-coding-agent/utils/event-bus";
import { TempDir } from "@oh-my-pi/pi-utils";

const zeroUsage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
} satisfies AssistantMessage["usage"];

// AgentSession schedules continuations through `scheduler.wait(delayMs)`. Collapse
// that blind settle delay to a macrotask hop so a continuation's turn boundary is
// reached deterministically instead of on the wall clock.
const originalSchedulerWait = scheduler.wait.bind(scheduler);
function collapseSchedulerSettleDelays(): void {
	vi.spyOn(scheduler, "wait").mockImplementation((_delayMs, options) => originalSchedulerWait(0, options));
}

function assistantMessage(
	model: Model,
	content: AssistantMessage["content"],
	stopReason: AssistantMessage["stopReason"],
): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: zeroUsage,
		stopReason,
		timestamp: Date.now(),
	};
}
function errorAssistantMessage(model: Model): AssistantMessage {
	return {
		...assistantMessage(model, [{ type: "text", text: "fatal error" }], "error"),
		errorMessage: "permanent admission probe failure",
		errorStatus: 400,
	};
}

/** A stream that completes on the next microtask, like the mock provider's. */
function completedStream(message: AssistantMessage): AssistantMessageEventStream {
	const stream = new AssistantMessageEventStream();
	queueMicrotask(() => {
		stream.push({ type: "start", partial: message });
		stream.push({ type: "done", reason: message.stopReason === "toolUse" ? "toolUse" : "stop", message });
	});
	return stream;
}

/** Concatenated text of a message whose content may be a string or text blocks. */
function messageText(message: { role: string; content?: unknown }): string {
	const content = message.content;
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	let text = "";
	for (const block of content) {
		if (block && typeof block === "object" && "type" in block && block.type === "text" && "text" in block) {
			text += String(block.text ?? "");
		}
	}
	return text;
}

function entryText(entry: SessionEntry | undefined): string | undefined {
	if (entry?.type !== "message") return undefined;
	return messageText(entry.message);
}

/** Text of every user-role message the provider was asked to answer. */
function submittedUserTexts(contexts: Context[]): string[] {
	return contexts.flatMap(context =>
		context.messages.filter(message => message.role === "user").map(message => messageText(message)),
	);
}

describe("AgentSession atomic user-message admission", () => {
	let tempDir: TempDir;
	let authStorage: AuthStorage;
	let modelRegistry: ModelRegistry;
	let session: AgentSession | undefined;

	beforeEach(async () => {
		tempDir = TempDir.createSync("@pi-admission-");
		authStorage = await AuthStorage.create(path.join(tempDir.path(), "auth.db"));
		authStorage.setRuntimeApiKey("openai", "openai-test-key");
		modelRegistry = new ModelRegistry(authStorage);
	});

	afterEach(async () => {
		await session?.dispose();
		authStorage.close();
		tempDir.removeSync();
		vi.restoreAllMocks();
	});

	/** A session whose provider answers with `text` for every call. */
	function buildSession(text = "Done.", tools: AgentTool[] = []): { contexts: Context[] } {
		const model = createMockModel({ provider: "openai", id: "gpt-test" }).model;
		const contexts: Context[] = [];
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model, systemPrompt: ["Test"], tools, messages: [] },
			convertToLlm,
			streamFn: (_model, context) => {
				contexts.push(context);
				return completedStream(assistantMessage(model, [{ type: "text", text }], "stop"));
			},
		});
		const settings = Settings.isolated({ "compaction.enabled": false, "todo.enabled": false });
		settings.setModelRole("default", `${model.provider}/${model.id}`);
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(tempDir.path()),
			settings,
			modelRegistry,
			toolRegistry: new Map(tools.map(tool => [tool.name, tool])),
		});
		return { contexts };
	}

	/** A session whose first provider call asks for `slow`, which blocks until released. */
	function buildBlockedSession(): { contexts: Context[]; started: Promise<void>; release: () => void } {
		const model = createMockModel({ provider: "openai", id: "gpt-test" }).model;
		const contexts: Context[] = [];
		const started = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const slowTool: AgentTool = {
			name: "slow",
			label: "Slow",
			description: "Blocks until released",
			parameters: type({}),
			execute: async () => {
				started.resolve();
				await release.promise;
				return { content: [{ type: "text", text: "SLOW_DONE" }] };
			},
		};
		let callCount = 0;
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model, systemPrompt: ["Test"], tools: [slowTool], messages: [] },
			convertToLlm,
			streamFn: (_model, context) => {
				contexts.push(context);
				const first = callCount === 0;
				callCount++;
				return completedStream(
					first
						? assistantMessage(
								model,
								[{ type: "toolCall", id: "tc-slow", name: "slow", arguments: {} }],
								"toolUse",
							)
						: assistantMessage(model, [{ type: "text", text: "BLOCKED_DONE" }], "stop"),
				);
			},
		});
		const settings = Settings.isolated({ "compaction.enabled": false, "todo.enabled": false });
		settings.setModelRole("default", `${model.provider}/${model.id}`);

		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(tempDir.path()),
			settings,
			modelRegistry,
			toolRegistry: new Map([[slowTool.name, slowTool]]),
		});
		return { contexts, started: started.promise, release: release.resolve };
	}
	/** A session whose provider returns a non-retryable terminal error. */
	function buildErrorSession(): void {
		const model = createMockModel({ provider: "openai", id: "gpt-test" }).model;
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model, systemPrompt: ["Test"], tools: [], messages: [] },
			convertToLlm,
			streamFn: () => completedStream(errorAssistantMessage(model)),
		});
		const settings = Settings.isolated({ "compaction.enabled": false, "todo.enabled": false });
		settings.setModelRole("default", `${model.provider}/${model.id}`);
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(tempDir.path()),
			settings,
			modelRegistry,
			toolRegistry: new Map(),
		});
	}

	it("admits exact content into an idle session and returns the native entry id of that input", async () => {
		const { contexts } = buildSession();
		// A template/slash-looking payload: the admitted path must not expand or
		// execute it, so the persisted input is byte-for-byte what was submitted.
		const submitted = "NEXTMCP_COMPAT_PROBE {{not_a_template}} /not-a-command";

		const result = await session!.admitUserMessage(submitted);

		expect(result.accepted).toBe(true);
		if (!result.accepted) throw new Error("expected the idle session to admit the message");
		const entry = session!.sessionManager.getEntry(result.inputEntryId);
		expect(entry?.type).toBe("message");
		expect(entry?.type === "message" ? entry.message.role : undefined).toBe("user");
		expect(entryText(entry)).toBe(submitted);

		await session!.waitForIdle();
		expect(contexts).toHaveLength(1);
		expect(submittedUserTexts(contexts)).toEqual([submitted]);
	});
	it("keeps the final assistant entry correlated to the admitted input", async () => {
		buildSession();
		const admitted = await session!.admitUserMessage("CORRELATION_ROOT");
		expect(admitted.accepted).toBe(true);
		if (!admitted.accepted) throw new Error("expected the correlation probe to be admitted");

		await session!.waitForIdle();
		const finalAssistant = [...session!.sessionManager.getBranch()]
			.reverse()
			.find(entry => entry.type === "message" && entry.message.role === "assistant");
		expect(finalAssistant).toBeDefined();
		if (!finalAssistant) throw new Error("expected a persisted final assistant entry");

		let currentId: string | null = finalAssistant.parentId;
		let foundInput = false;
		while (currentId !== null) {
			if (currentId === admitted.inputEntryId) {
				foundInput = true;
				break;
			}
			currentId = session!.sessionManager.getEntry(currentId)?.parentId ?? null;
		}
		expect(foundInput).toBe(true);
	});

	it("admits only one of two simultaneous admissions", async () => {
		const { contexts } = buildSession();

		const results = await Promise.all([session!.admitUserMessage("FIRST"), session!.admitUserMessage("SECOND")]);

		const accepted = results.filter(result => result.accepted);
		const refused = results.filter(result => !result.accepted);
		expect(accepted).toHaveLength(1);
		expect(refused).toEqual([{ accepted: false, reason: "busy" }]);

		const winner = accepted[0];
		if (!winner?.accepted) throw new Error("expected exactly one admitted message");
		const winnerText = results[0] === winner ? "FIRST" : "SECOND";
		const loserText = winnerText === "FIRST" ? "SECOND" : "FIRST";

		expect(entryText(session!.sessionManager.getEntry(winner.inputEntryId))).toBe(winnerText);

		await session!.waitForIdle();
		expect(submittedUserTexts(contexts)).toEqual([winnerText]);
		// The refused call left nothing behind for the loser to be steered from.
		expect(session!.sessionManager.getBranch().some(entry => entryText(entry) === loserText)).toBe(false);
	});

	it("refuses admission while a run is active without creating an entry or starting a run", async () => {
		const { contexts, started, release } = buildBlockedSession();
		const run = session!.prompt("initial message");
		await started;

		const entryCount = session!.sessionManager.getBranch().length;
		const result = await session!.admitUserMessage("FOREIGN_ADMISSION");

		expect(result).toEqual({ accepted: false, reason: "busy" });
		expect(session!.sessionManager.getBranch()).toHaveLength(entryCount);
		expect(contexts).toHaveLength(1);

		release();
		await run;
		await session!.waitForIdle();
		expect(submittedUserTexts([contexts[0]!])).toEqual(["initial message"]);
		expect(JSON.stringify(contexts)).not.toContain("FOREIGN_ADMISSION");
	});

	it("refuses admission while input is already queued for the session", async () => {
		buildSession();
		// A follow-up queued on a fresh session cannot auto-resume (there is no
		// provider-valid tail to continue from), so it stays queued while idle —
		// the one reachable state where admitting would interleave with delivery.
		await session!.sendUserMessage("QUEUED_FOLLOW_UP", { deliverAs: "followUp" });
		expect(session!.isStreaming).toBe(false);
		expect(session!.queuedMessageCount).toBe(1);

		const result = await session!.admitUserMessage("AFTER_QUEUE");

		expect(result).toEqual({ accepted: false, reason: "pending_message" });
		expect(session!.sessionManager.getBranch().some(entry => entryText(entry) === "AFTER_QUEUE")).toBe(false);
	});

	it("refuses admission while compaction owns the transcript", async () => {
		buildSession();
		// `isCompacting` is a getter on the session; shadow it with the state real
		// compaction produces so the refusal mapping is exercised without driving a
		// full (slow, model-dependent) compaction pass.
		Object.defineProperty(session!, "isCompacting", { configurable: true, get: () => true });

		const result = await session!.admitUserMessage("DURING_COMPACTION");

		expect(result).toEqual({ accepted: false, reason: "compacting" });
	});

	it("refuses admission from a disposing session", async () => {
		buildSession();
		session!.beginDispose();

		const result = await session!.admitUserMessage("DURING_DISPOSE");

		expect(result).toEqual({ accepted: false, reason: "busy" });
	});

	it("does not persist or run a reservation revoked after it was taken", async () => {
		const { contexts } = buildSession();

		// The reservation is synchronous, so revoking it immediately lands inside the
		// dispatch's pre-dispatch awaits — after the session was reserved, before any
		// message exists. The caller is told nothing started, and that must hold: the
		// frame may not fall through to the ordinary prompt path and persist the input.
		const admission = session!.admitUserMessage("REVOKED_BEFORE_DISPATCH");
		session!.beginDispose();
		const result = await admission;

		expect(result).toEqual({ accepted: false, reason: "not_started" });
		await session!.waitForIdle();
		expect(session!.sessionManager.getBranch().some(entry => entryText(entry) === "REVOKED_BEFORE_DISPATCH")).toBe(
			false,
		);
		expect(contexts).toHaveLength(0);
	});

	it("does not persist or run a reservation revoked by an abort", async () => {
		const { contexts } = buildSession();

		const admission = session!.admitUserMessage("ABORTED_BEFORE_DISPATCH");
		void session!.abort();
		const result = await admission;

		expect(result).toEqual({ accepted: false, reason: "not_started" });
		await session!.waitForIdle();
		expect(session!.sessionManager.getBranch().some(entry => entryText(entry) === "ABORTED_BEFORE_DISPATCH")).toBe(
			false,
		);
		expect(contexts).toHaveLength(0);
		// The abort ended the reservation, so the session is ownable again.
		// No re-admission assertion here: an idle abort drives its own abort-turn
		// activity, so the session is legitimately busy again afterwards. What this
		// case pins is that the revoked reservation left nothing behind.
	});
	it("releases admission ownership after a terminal error", async () => {
		buildErrorSession();
		const admitted = await session!.admitUserMessage("ERROR_RUN");
		expect(admitted.accepted).toBe(true);
		if (!admitted.accepted) throw new Error("expected the error run to be admitted");

		await session!.waitForIdle();
		const afterError = await session!.admitUserMessage("AFTER_ERROR");

		expect(afterError.accepted).toBe(true);
	});

	it("refuses admission at an intermediate settle and frees the session only at the terminal one", async () => {
		collapseSchedulerSettleDelays();
		const model = createMockModel({ provider: "openai", id: "gpt-test" }).model;
		const contexts: Context[] = [];
		const duringGap: UserMessageAdmissionResult[] = [];
		const gapProbe = Promise.withResolvers<void>();
		let stopCount = 0;
		const extensionRuntime = new ExtensionRuntime();
		const extension = await loadExtensionFromFactory(
			pi => {
				pi.on("agent_end", event => {
					// Fires at the intermediate settle, while the continuation is scheduled
					// but the run is not over: ownership must still be held here.
					if (event.willContinue === true && duringGap.length === 0) {
						void session!.admitUserMessage("DURING_CONTINUATION_GAP").then(result => {
							duringGap.push(result);
							gapProbe.resolve();
						});
					}
				});
				pi.on("session_stop", () => {
					stopCount++;
					return stopCount === 1 ? { continue: true, additionalContext: "Keep going." } : undefined;
				});
			},
			tempDir.path(),
			new EventBus(),
			extensionRuntime,
			"admission-continuation",
		);
		const sessionManager = SessionManager.inMemory(tempDir.path());
		const extensionRunner = new ExtensionRunner(
			[extension],
			extensionRuntime,
			tempDir.path(),
			sessionManager,
			modelRegistry,
		);
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model, systemPrompt: ["Test"], tools: [], messages: [] },
			convertToLlm,
			streamFn: (_model, context) => {
				contexts.push(context);
				return completedStream(assistantMessage(model, [{ type: "text", text: "Done." }], "stop"));
			},
		});
		const settings = Settings.isolated({ "compaction.enabled": false, "todo.enabled": false });
		settings.setModelRole("default", `${model.provider}/${model.id}`);
		session = new AgentSession({ agent, sessionManager, settings, modelRegistry, extensionRunner });

		// Drive the run through admission so the gap probe observes an owned session,
		// not just an ordinary run's own streaming state.
		const admitted = await session.admitUserMessage("first");
		expect(admitted.accepted).toBe(true);
		await gapProbe.promise;
		await session.waitForIdle();

		expect(duringGap).toEqual([{ accepted: false, reason: "busy" }]);
		expect(contexts).toHaveLength(2);
		// Only the terminal settle releases the reservation, so a fresh admission then
		// succeeds — and one taken before that settle never would have.
		const afterSettle = await session.admitUserMessage("AFTER_SETTLE");
		expect(afterSettle.accepted).toBe(true);
	});

	it("keeps an admitted input's identity through an abort", async () => {
		const { started, release } = buildBlockedSession();
		const admission = session!.admitUserMessage("WILL_BE_ABORTED");
		await started;

		const admitted = await admission;
		expect(admitted.accepted).toBe(true);
		if (!admitted.accepted) throw new Error("expected the admitted run to record its input");
		expect(entryText(session!.sessionManager.getEntry(admitted.inputEntryId))).toBe("WILL_BE_ABORTED");

		// abort() resolves only once the run settles, which requires the blocked tool
		// to return — release it concurrently instead of awaiting abort first.
		const aborting = session!.abort();
		release();
		await aborting;
		await session!.waitForIdle();

		// The abort settled the owned run, so the session is ownable again.
		const readmitted = await session!.admitUserMessage("AFTER_ABORT");
		expect(readmitted.accepted).toBe(true);
	});

	it("leaves ordinary sendUserMessage steering unchanged", async () => {
		const { contexts, started, release } = buildBlockedSession();
		const run = session!.prompt("initial message");
		await started;

		await session!.sendUserMessage("STEERED_WHILE_RUNNING");
		// sendUserMessage still merges into the live run rather than starting a second one.
		expect(contexts).toHaveLength(1);
		expect(session!.agent.peekSteeringQueue()).toHaveLength(1);

		release();
		await run;
		await session!.waitForIdle();

		expect(contexts).toHaveLength(2);
		// The steer reached the *same* run's next provider call, not a second run.
		expect(submittedUserTexts([contexts[0]!])).toEqual(["initial message"]);
		expect(submittedUserTexts([contexts[1]!])).toEqual(["initial message", "STEERED_WHILE_RUNNING"]);
	});

	it("does not execute a registered extension command from admitted text", async () => {
		const model = createMockModel({ provider: "openai", id: "gpt-test" }).model;
		const contexts: Context[] = [];
		let commandRuns = 0;
		let api: ExtensionAPI | undefined;
		const extensionRuntime = new ExtensionRuntime();
		const extension = await loadExtensionFromFactory(
			pi => {
				api = pi;
				pi.registerCommand("admission-probe-command", {
					description: "must not run for an admitted message",
					handler: async () => {
						commandRuns++;
					},
				});
			},
			tempDir.path(),
			new EventBus(),
			extensionRuntime,
			"admission-command-guard",
		);
		const sessionManager = SessionManager.inMemory(tempDir.path());
		const extensionRunner = new ExtensionRunner(
			[extension],
			extensionRuntime,
			tempDir.path(),
			sessionManager,
			modelRegistry,
		);
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model, systemPrompt: ["Test"], tools: [], messages: [] },
			convertToLlm,
			streamFn: (_model, context) => {
				contexts.push(context);
				return completedStream(assistantMessage(model, [{ type: "text", text: "Done." }], "stop"));
			},
		});
		const settings = Settings.isolated({ "compaction.enabled": false, "todo.enabled": false });
		settings.setModelRole("default", `${model.provider}/${model.id}`);
		session = new AgentSession({ agent, sessionManager, settings, modelRegistry, extensionRunner });
		await initializeExtensions(session, {
			reportSendError: (_action, error) => {
				throw error;
			},
			reportRuntimeError: error => {
				throw error.error;
			},
		});

		if (!api) throw new Error("extension factory did not receive the API");
		// Admitted text is submitted verbatim; it is never routed through command
		// handling or template expansion.
		const submitted = "/admission-probe-command argument";
		const result = await api.admitUserMessage(submitted);

		expect(result.accepted).toBe(true);
		if (!result.accepted) throw new Error("expected the command-looking message to be admitted");
		expect(commandRuns).toBe(0);
		expect(entryText(sessionManager.getEntry(result.inputEntryId))).toBe(submitted);
		await session.waitForIdle();
		expect(submittedUserTexts(contexts)).toEqual([submitted]);
	});

	it("exposes the primitive to extensions through the print/RPC runtime wiring", async () => {
		const model = createMockModel({ provider: "openai", id: "gpt-test" }).model;
		const contexts: Context[] = [];
		let api: ExtensionAPI | undefined;
		const extensionRuntime = new ExtensionRuntime();
		const extension = await loadExtensionFromFactory(
			pi => {
				api = pi;
			},
			tempDir.path(),
			new EventBus(),
			extensionRuntime,
			"admission-api",
		);
		const sessionManager = SessionManager.inMemory(tempDir.path());
		const extensionRunner = new ExtensionRunner(
			[extension],
			extensionRuntime,
			tempDir.path(),
			sessionManager,
			modelRegistry,
		);
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model, systemPrompt: ["Test"], tools: [], messages: [] },
			convertToLlm,
			streamFn: (_model, context) => {
				contexts.push(context);
				return completedStream(assistantMessage(model, [{ type: "text", text: "Done." }], "stop"));
			},
		});
		const settings = Settings.isolated({ "compaction.enabled": false, "todo.enabled": false });
		settings.setModelRole("default", `${model.provider}/${model.id}`);
		session = new AgentSession({ agent, sessionManager, settings, modelRegistry, extensionRunner });
		await initializeExtensions(session, {
			reportSendError: (_action, error) => {
				throw error;
			},
			reportRuntimeError: error => {
				throw error.error;
			},
		});

		if (!api) throw new Error("extension factory did not receive the API");
		const result = await api.admitUserMessage("FROM_EXTENSION");

		expect(result.accepted).toBe(true);
		if (!result.accepted) throw new Error("expected the extension call to be admitted");
		expect(entryText(sessionManager.getEntry(result.inputEntryId))).toBe("FROM_EXTENSION");
		await session.waitForIdle();
		expect(submittedUserTexts(contexts)).toEqual(["FROM_EXTENSION"]);
	});

	it("keeps RPC admission tracking refusal-aware without unhandled rejections", async () => {
		const model = createMockModel({ provider: "openai", id: "gpt-test" }).model;
		let api: ExtensionAPI | undefined;
		const extensionRuntime = new ExtensionRuntime();
		const extension = await loadExtensionFromFactory(
			pi => {
				api = pi;
			},
			tempDir.path(),
			new EventBus(),
			extensionRuntime,
			"admission-rpc-tracking",
		);
		const sessionManager = SessionManager.inMemory(tempDir.path());
		const extensionRunner = new ExtensionRunner(
			[extension],
			extensionRuntime,
			tempDir.path(),
			sessionManager,
			modelRegistry,
		);
		const release = Promise.withResolvers<void>();
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model, systemPrompt: ["Test"], tools: [], messages: [] },
			convertToLlm,
			streamFn: () => {
				const stream = new AssistantMessageEventStream();
				void release.promise.then(() => {
					const message = assistantMessage(model, [{ type: "text", text: "Done." }], "stop");
					stream.push({ type: "start", partial: message });
					stream.push({ type: "done", reason: "stop", message });
				});
				return stream;
			},
		});
		const settings = Settings.isolated({ "compaction.enabled": false, "todo.enabled": false });
		settings.setModelRole("default", `${model.provider}/${model.id}`);
		session = new AgentSession({ agent, sessionManager, settings, modelRegistry, extensionRunner });
		const tracked: Promise<unknown>[] = [];
		await initializeExtensions(session, {
			reportSendError: (_action, error) => {
				throw error;
			},
			reportRuntimeError: error => {
				throw error.error;
			},
			trackAgentInvokingMessage: task => tracked.push(task),
		});
		if (!api) throw new Error("extension factory did not receive the API");
		const accepted = await api.admitUserMessage("TRACKED_WINNER");
		const refused = await api.admitUserMessage("TRACKED_BUSY");

		expect(accepted.accepted).toBe(true);
		expect(refused).toEqual({ accepted: false, reason: "busy" });
		const trackedResults = await Promise.allSettled(tracked);
		expect(trackedResults[0]?.status).toBe("fulfilled");
		expect(trackedResults[1]?.status).toBe("rejected");
		release.resolve();
		await session.waitForIdle();
	});
});
