# TODO — Implementation vs Architecture

Based on `accelerator_agent_architecture.md`. **Done** = implemented; **Todo** = missing or partial.

---

## Done

| Area | Item | Notes |
|------|------|--------|
| **Core** | ReAct loop | `core/react_loop.py` — think→act→observe, tool calls, max iterations |
| **Core** | Conversation | `core/conversation.py` — user/assistant/tool messages, OpenAI-style for LiteLLM |
| **Core** | Context / AGENT.md | `core/context.py` — system prompt, AGENT.md + extra dirs |
| **Core** | Action plan model | `core/plan.py` — READ, WRITE, WAIT, SCAN; `ActionPlan.from_dict` |
| **Core** | Plan executor | `core/plan_executor.py` — execute plan, rollback on failure, SCAN loop |
| **Core** | Sub-agent dispatcher | `core/subagent.py` — device_search, log_analysis, diagnostics tool sets |
| **Core** | Orchestrator wiring | `core/agent.py` — LLM, adapter, tools, ReAct, subagent |
| **LLM** | Unified interface | LiteLLM in `llm/litellm_client.py` (replaces per-provider clients) |
| **LLM** | Model switching | `llm/router.py` — named models, `/provider` |
| **Tools** | get_parameter | Read device value via adapter |
| **Tools** | search_device | Pattern + optional subsystem |
| **Tools** | read_log | source, time_range, filters |
| **Tools** | propose_plan | Approval gate + plan executor; schema for 4 action types |
| **Tools** | execute_code | Static analysis, subprocess run, DATA_DIR/WORK_DIR/SCRIPTS_DIR in preamble |
| **Tools** | run_script | Run script from scripts dir via execute_code path |
| **Tools** | list_scripts | List scripts + docstring descriptions, optional filter |
| **Tools** | spawn_subagent | Dispatcher with restricted tool sets |
| **Tools** | launch_display | Stub only (returns not_implemented) |
| **Safety** | Approval gate | `safety/approval.py` — review plan, approve/reject/modify (steps, dwell) |
| **Safety** | Parameter validator | `safety/validators.py` — safety_limits.toml, wildcards, SafetyError |
| **Safety** | Code analyzer | `safety/code_analysis.py` — blocked imports/calls, allowlist |
| **Safety** | Audit log class | `safety/audit.py` — `record(plan, approved, operator, result)` |
| **Safety** | Audit log wired | `propose_plan.py` logs approval/rejection/execution; operator from config/env |
| **Core** | Conversation format | `conversation.py` — OpenAI-style tool result messages for LiteLLM |
| **Adapters** | Base ABC | `adapters/base.py` — get, set, search_devices, read_log, ReadResult |
| **Adapters** | Simulator | `adapters/simulator.py` — in-memory state, mock search/log |
| **Sandbox** | execute_code harness | Inline in tool: timeout, DATA_DIR/WORK_DIR/SCRIPTS_DIR, MPLBACKEND |
| **CLI** | Terminal UI | `cli/app.py` — prompt_toolkit, rich, /help, /provider, /context, /clear, /quit |
| **Config** | default.toml | LLM (LiteLLM models), adapter, safety_limits path, sandbox |
| **Config** | safety_limits.toml | Example device/wildcard limits |
| **Tests** | Plan model | `tests/test_plan.py` — ActionPlan.from_dict |

---

## Todo

### Phase 1 — Make the agent system work

| # | Item | Notes |
|---|------|--------|
| 1 | **Error handling** | (a) Wrap `tool.execute()` in ReAct loop with try/except — tool failures should return error results to the LLM, not crash the CLI. (b) Catch LLM API errors (rate limits, timeouts, network) in `litellm_client.py` with retry/backoff. (c) Surface errors clearly to the operator in the CLI. |
| 2 | **Simulator adapter** | Seed with a realistic device database so sub-agent search and example flows actually work. |
| 3 | **Rolling context window** | Conversation grows unbounded and will eventually exceed LLM context limits. Implement truncation or summarization of older turns to keep within token budget while preserving key context. |
| 4 | **Token/cost tracking** | `LLMResponse.usage` is parsed but never displayed or accumulated. Track per-turn and session totals, display to operator (e.g. `/usage` command or status line). Important for cost awareness especially with sub-agents. |
| 5 | **Integration tests** | Only unit test is `test_plan.py`. Add tests for ReAct + simulator, propose_plan flow, subagent flow. |

### Phase 2 — UI/UX and robustness

| # | Item | Notes |
|---|------|--------|
| 6 | **UI/UX improvements** | (a) Streaming: stream LLM text output token-by-token instead of blocking behind a spinner. (b) Progress feedback: show which tool is executing during ReAct steps. (c) `cli/rendering.py` is empty — implement output formatting helpers. (d) SCAN progress bar and early abort support in plan executor. |
| 7 | **Conversation persistence and logging** | Conversations are lost on exit. Save conversation history to disk (JSONL or SQLite) for shift handoffs, post-session review, and debugging. |
| 8 | **Approval: selective approve** | Add option to approve/reject individual actions in a multi-write plan. |

### Phase 3 — Extensibility and hardening

| # | Item | Notes |
|---|------|--------|
| 9 | **MCP (Model Context Protocol)** | Expose agent tools as MCP servers and/or consume external MCP tool servers. Enables integration with other AI tools and workflows. |
| 10 | **Skills** | Reusable, composable multi-step procedures that the agent can invoke by name (e.g. "orbit correction", "RF conditioning"). Sits between raw tool calls and full free-form planning — operator-vetted recipes the agent can follow. |
| 11 | **launch_display** | Currently returns `not_implemented`. Implement facility-specific display launch. |
| 12 | **Harden execute_code sandbox** | Add `resource.setrlimit`, `safe_open()` override, `sys.meta_path` import blocker in `_wrap_code`. Strip control system env vars from subprocess environment. |
| 13 | **OS-level sandbox** | `sandbox/namespace.py` is a TODO. Add bwrap or Docker isolation for production. |

---

## Summary

- **Implemented:** Core ReAct loop, plans, approval gate, validators, code analysis, all tools (most complete; launch_display stub), LiteLLM, simulator adapter, CLI with main commands, config, basic test.
- **Phase 1:** Error handling, simulator seeding, rolling context window, token tracking, integration tests. Get the system running end-to-end.
- **Phase 2:** Streaming/UI, conversation persistence, selective approval. Make it usable for operators.
- **Phase 3:** MCP, skills, launch_display, sandbox hardening. Extensibility and production readiness. Facility-specific adapters (e.g. EPICS, TANGO, DOOCS) are out of scope but supported via the `ControlSystemAdapter` ABC.
