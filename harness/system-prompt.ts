// What the agent is told to do. Deliberately simple — the agent is the boring
// payload; the harness is the course.
export const SYSTEM_PROMPT = `Agent prompts now live in agents.ts (one per agent). This file just holds a simple  task to try.
A single refund request: triage will recognize it can't issue refunds and hand off to the billing specialist.`;

export const SAMPLE_TASK = `Customer cus_88121 says they were charged twice and wants the duplicate charge refunded. Sort it out.`;
