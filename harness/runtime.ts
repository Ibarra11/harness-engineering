import { DBOS } from "@dbos-inc/dbos-sdk";
import { emit } from "harness/bus";
import type { ModelMessage, JSONValue, ToolSet } from "ai";
import { streamText } from "ai";
import { EventType } from "@shared/events";
import { model } from "./model";
import { runTool } from "./tools";
import { agents, triageAgent } from "./agents";

import {
  buildContext,
  summarize,
  estimateTokens,
  MAX_CONTEXT_TOKENS,
  KEEP_CONTEXT_TOKENS,
} from "./memory";

// A safety cap so a confused model can't loop forever. Higher than Lesson 1 now
// that one task can span many items (and therefore many turns).
const MAX_STEPS = 30;

type ToolCall = {
  toolCallId: string;
  toolName: string;
  input: Record<string, unknown>;
};
type Turn = {
  text: string;
  toolCalls: ToolCall[];
  responseMessages: ModelMessage[];
};

// One model turn: stream the tokens out as events, then return the assistant's
// message(s) and any tool calls. We run this as a DBOS step, so a completed turn
// is checkpointed and never re-called — a crash won't re-bill the LLM.
async function modelTurn(
  workflowId: string,
  context: ModelMessage[],
  agentTools: ToolSet,
): Promise<Turn> {
  const result = streamText({ model, messages: context, tools: agentTools });

  for await (const part of result.fullStream) {
    if (part.type === "text-delta") {
      await emit({ type: EventType.ModelDelta, workflowId, text: part.text });
    }
  }

  const rawCalls = await result.toolCalls;
  return {
    text: await result.text,
    toolCalls: rawCalls.map((c) => ({
      toolCallId: c.toolCallId,
      toolName: c.toolName,
      input: c.input as Record<string, unknown>,
    })),
    responseMessages: (await result.response).messages,
  };
  ``;
}

function toolResultMessage(call: ToolCall, value: JSONValue): ModelMessage {
  return {
    role: "tool",
    content: [
      {
        type: "tool-result",
        toolCallId: call.toolCallId,
        toolName: call.toolName,
        output: { type: "json", value },
      },
    ],
  };
}

// Execute one tool. We run this as a DBOS step so its side effect (e.g.
// sendReply actually emailing someone) runs EXACTLY ONCE — a completed tool step
// is never re-run when DBOS recovers the workflow after a crash.
async function toolStep(
  workflowId: string,
  call: ToolCall,
): Promise<Record<string, unknown>> {
  await emit({
    type: EventType.ToolRequested,
    workflowId,
    toolCallId: call.toolCallId,
    name: call.toolName,
    args: call.input,
  });
  const output = await runTool(call.toolName, call.input);
  await emit({
    type: EventType.ToolCompleted,
    workflowId,
    toolCallId: call.toolCallId,
    result: output,
  });
  return output;
}
// The loop is identical to before, with two additions:
//   · it runs the CURRENT agent's prompt + tools (start: triage)
//   · the `handoff` tool isn't executed — the harness intercepts it and SWITCHES
//     the running agent, keeping the conversation. Control transfers laterally.

export async function agentWorkflow(opts: { input: string }): Promise<string> {
  const workflowId = DBOS.workflowID ?? "unknown";
  const { input } = opts;
  await DBOS.runStep(
    () => emit({ type: EventType.WorkflowStarted, workflowId, input }),
    { name: "started" },
  );

  let currentAgent = triageAgent;

  const turns: ModelMessage[][] = [];
  let summary = "";

  // THE LOOP. We drive it ourselves — each pass is exactly one model turn,
  // because streamText does a single generation by default.
  let step = 0;
  while (step < MAX_STEPS) {
    // 1. Compact: while the recent window is over budget, peel the oldest turns
    //    into the running summary (keeping at least the last turn verbatim).
    // if (estimateTokens(turns.flat()) > MAX_CONTEXT_TOKENS) {
    //   const old: ModelMessage[][] = [];
    //   while (
    //     turns.length > 1 &&
    //     estimateTokens(turns.flat()) > KEEP_CONTEXT_TOKENS
    //   ) {
    //     const oldest = turns.shift();
    //     if (oldest) old.push(oldest);
    //   }
    //   if (old.length > 0) {
    //     summary = await DBOS.runStep(() => summarize(old, summary), {
    //       name: `summarize-${step}`,
    //     });
    //     const contextTokens = estimateTokens(
    //       buildContext(input, summary, turns, currentAgent.systemPrompt),
    //     );
    //     await DBOS.runStep(
    //       () =>
    //         emit({
    //           type: EventType.MemoryCompacted,
    //           workflowId,
    //           summarizedTurns: old.length,
    //           contextTokens,
    //           summary,
    //         }),
    //       { name: `compacted-${step}` },
    //     );
    //   }
    // }

    // 2 + 3. Hydrate the context and run one turn over it.
    const context = buildContext(
      input,
      summary,
      turns,
      currentAgent.systemPrompt,
    );
    const turn = await DBOS.runStep(
      () => modelTurn(workflowId, context, currentAgent.tools),
      {
        name: `model-${step}`,
      },
    );

    const turnMessages: ModelMessage[] = [...turn.responseMessages];
    if (turn.toolCalls.length === 0) {
      await DBOS.runStep(
        () =>
          emit({ type: EventType.ModelCompleted, workflowId, text: turn.text }),
        { name: `model-done-${step}` },
      );
      await DBOS.runStep(
        () =>
          emit({
            type: EventType.WorkflowCompleted,
            workflowId,
            output: turn.text,
          }),
        { name: "completed" },
      );
      return turn.text;
    }

    for (const call of turn.toolCalls) {
      if (call.toolName === "handoff") {
        // The harness intercepts handoff: switch the running agent, don't run a tool.
        const to = String(call.input.to ?? "");
        const reason = String(call.input.reason ?? "");
        const from = currentAgent.name;
        await DBOS.runStep(
          () =>
            emit({
              type: EventType.AgentHandoff,
              workflowId,
              from,
              to,
              reason,
            }),
          { name: `handoff-${call.toolCallId}` },
        );
        currentAgent = agents[to] ?? currentAgent;
        turnMessages.push(
          toolResultMessage(call, {
            ok: true,
            handedOffTo: to,
            reason: `You were delegated from ${from} to ${to} to complete the task: ${context[1].content}`,
          }),
        );
      } else {
        const output = await DBOS.runStep(() => toolStep(workflowId, call), {
          name: `tool-${call.toolCallId}`,
        });
        turnMessages.push(toolResultMessage(call, output as JSONValue));
      }
    }
    turns.push(turnMessages);

    step++;
  }

  await DBOS.runStep(
    () =>
      emit({
        type: EventType.WorkflowFailed,
        workflowId,
        error: `Hit the ${MAX_STEPS}-step limit without finishing.`,
      }),
    { name: "failed" },
  );
  return "";
}
export const runAgentWorkflow = DBOS.registerWorkflow(agentWorkflow, {
  name: "agentWorkflow",
});
