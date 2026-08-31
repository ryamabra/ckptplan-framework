# Token-Efficient Collaboration Workflow

## Purpose

This repository is the shared memory for the human project owner and the project assistants. Assistants must reconstruct project state from concise files and the repository rather than requiring the owner to paste previous conversations.

This workflow reduces repeated context, but it cannot transfer account quota, hidden conversation state, or provider-specific memory.

## Source of Truth

Read these in order at the start of a new session:

1. `STATE.md` — current task, accepted decisions, constraints, and next action.
2. `ARCHITECTURE.md` — stable system boundaries and technical design.
3. `WORKFLOW.md` — collaboration and handoff rules.
4. Only files directly relevant to the current task.

Do not begin by reading the entire repository. When `graphify-out/graph.json` exists and the question concerns code relationships, query Graphify first and inspect only the files returned by that query.

## Session Start Prompt

```text
Read STATE.md, ARCHITECTURE.md, and WORKFLOW.md. Treat them as the project source of truth. Inspect only files needed for the current task. Do not summarize the documents back to me. Acknowledge with: "Ready. Current task: [Current Task from STATE.md]" and wait for my instruction.
```

## Task Prompt

```text
Work only on the Current Task recorded in STATE.md.

Before editing:
- inspect the relevant files and existing tests;
- state the exact deliverable in one sentence;
- flag only blockers that materially change the design.

While working:
- keep changes narrow;
- use supported public APIs;
- preserve unrelated work;
- test in proportion to risk;
- avoid repeating repository context already recorded in project files.

Before finishing:
- run the relevant tests;
- update STATE.md with decisions, completed work, remaining risks, and the next atomic task;
- update ARCHITECTURE.md only if the accepted architecture changed;
- report changed files, verification results, and the next task concisely.
```

## Handoff Rules

At the end of every meaningful work session, the active assistant must update `STATE.md` so another assistant can continue without the previous chat transcript.

The update must record:

- what was completed;
- decisions accepted and why;
- files changed;
- tests or benchmarks run and their results;
- unresolved risks or blockers;
- one clearly worded current task;
- the next task after the current one.

Do not store chain-of-thought, long chat transcripts, speculative alternatives, or duplicated architecture explanations in `STATE.md`.

## Graphify Policy

- The project-scoped knowledge-graph integration is installed in strict query-first mode under the project configuration directory.
- Do not rebuild the graph after every small edit.
- Build it once after the repository has a meaningful code skeleton.
- Use incremental updates after structural changes or before a provider handoff.
- Use a strict query token budget for ordinary architecture questions.
- Prefer deterministic code extraction; semantic extraction of documents can consume model usage.
- Graphify supplements the source files. It does not replace tests, version control, `STATE.md`, or `ARCHITECTURE.md`.

Suggested commands once the repository contains code:

```text
/graphify . --no-viz
/graphify . --update --no-viz
/graphify query "Which components are affected by the current task?" --budget 1200
```

## Usage-Conservation Rules

- Use one assistant at a time.
- Give each assistant one atomic task per session.
- Use a lower-cost model for formatting, simple tests, routine refactors, and documentation cleanup when the provider offers one.
- Reserve the strongest reasoning model for architecture decisions, PyTorch internals, scheduler formulation, difficult debugging, and final review.
- Keep reasoning effort low or medium for routine work and increase it only when the task requires it.
- Avoid asking two assistants to independently solve the same problem unless performing a deliberate review.
- Ask the second assistant to review a diff or decision record, not to reread the entire project.
- Keep terminal output bounded and inspect targeted file sections rather than dumping large files.
- Commit or checkpoint coherent changes before switching providers.
- Check the provider's usage dashboard rather than guessing remaining quota.

## Active Collaboration Model

- The implementation agent edits the codebase, runs tests and benchmarks, and performs routine Git work.
- The reasoning partner helps define specifications, evaluate design tradeoffs, review plans and diffs, explain ML systems concepts, and prepare focused implementation instructions.
- The human project owner approves scope and transfers architecture decisions or review findings between the two surfaces.
- Architecture decisions must be recorded in `ARCHITECTURE.md`; execution state and the next atomic task must be recorded in `STATE.md`.

### Work Cycle

1. Discuss the design or unresolved decision with the reasoning partner.
2. Record or approve the resulting decision in the project documents.
3. Give the implementation agent one focused task with acceptance criteria.
4. The implementation agent edits, tests, and updates `STATE.md`.
5. Ask the reasoning partner to review the resulting diff, test evidence, or new design question.
6. Return concrete review findings to the implementation agent for correction.

## Subagent Policy

Use subagents for independent analysis and verification on substantial tasks. More agents are not automatically better: every subagent consumes additional usage, and parallel editors can conflict.

For each substantial implementation milestone, use this structure:

1. One primary implementation agent owns all edits for the milestone.
2. One specification-compliance subagent maps every acceptance criterion to evidence in the diff or tests.
3. One correctness/test subagent looks for missing cases and runs or inspects relevant tests.
4. One architecture/performance subagent reviews boundary violations, PyTorch assumptions, and avoidable performance risks.
5. The primary agent reconciles findings, applies fixes, reruns verification, and reports an acceptance-criteria checklist.

Reviewer subagents should be read-only whenever possible. Do not let multiple subagents edit the same files concurrently. Use isolated git worktrees only when tasks are genuinely independent and each has a separate file or component boundary.

Default to no more than three reviewer subagents in parallel. For small edits, use one primary agent and one reviewer rather than spending usage on a full panel.

If the implementation agent reaches its usage limit, stop implementation at a clean checkpoint. Do not automatically switch implementation agents unless the project owner explicitly changes this workflow again.

## Plugin Policy

No external connector plugin is required for the core workflow. Plugins cannot merge assistant quotas or automatically transfer hidden context between them.

Consider a project-management plugin only if the owner decides to maintain tasks in an external service such as GitHub, Linear, Trello, or Asana. Until then, repository issues and the three project documents are simpler and consume less context.
