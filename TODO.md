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
| **Adapters** | Base ABC | `adapters/base.py` — get, set, search_devices, read_log, ReadResult |
| **Adapters** | Simulator | `adapters/simulator.py` — in-memory state, mock search/log |
| **Sandbox** | execute_code harness | Inline in tool: timeout, DATA_DIR/WORK_DIR/SCRIPTS_DIR, MPLBACKEND |
| **CLI** | Terminal UI | `cli/app.py` — prompt_toolkit, rich, /help, /provider, /context, /clear, /quit |
| **Config** | default.toml | LLM (LiteLLM models), adapter, safety_limits path, sandbox |
| **Config** | safety_limits.toml | Example device/wildcard limits |
| **Tests** | Plan model | `tests/test_plan.py` — ActionPlan.from_dict |

---

## Todo

### High priority

| # | Item | Architecture ref | Notes |
|---|------|------------------|--------|
| 1 | **Wire audit log** | §9 Audit logging | `AuditLog` exists but is never called. Call `audit.record(...)` from `propose_plan` (or approval gate) after review/execute, with operator identity (e.g. env or config). |
| 2 | **Implement /history** | §14 CLI | Help lists "/history — Show conversation summary"; no handler in `_handle_command`. Add case that shows a short summary of conversation (e.g. message count, last N turns). |
| 3 | **Harden execute_code sandbox** | §10 Subprocess sandbox | Architecture requires: (1) `resource.setrlimit` (RLIMIT_AS, RLIMIT_NPROC), (2) safe `open()` override (read only from DATA_DIRS/SCRIPTS_DIR, write only WORK_DIR), (3) `sys.meta_path` import blocker in wrapped code. Current `_wrap_code` only sets path vars and Agg; no resource limits, no open override, no import blocker. |

### Medium priority

| # | Item | Architecture ref | Notes |
|---|------|------------------|--------|
| 4 | **EPICS adapter** | §8, project structure | `adapters/epics.py` is a stub (NotImplementedError). Implement with pyepics/pcaspy/pvAccess for one facility. |
| 5 | **TANGO adapter** | §8, project structure | `adapters/tango.py` is a stub. Implement with PyTango for facilities that use TANGO. |
| 6 | **launch_display** | §7 Tool set | Currently returns `not_implemented`. Implement facility-specific launch of plot/control displays (e.g. Open XAL, CS-Studio). |
| 7 | **Approval: selective approve** | §4 Approval gate | Doc: "For plans with multiple write actions, the operator can selectively approve a subset." Current choices are only all / none / modify. Add option to approve/reject per action (e.g. by index). |
| 8 | **SCAN progress & early abort** | §3 Plan executor | Doc: "progress bars, early abort" for SCAN. Executor runs SCAN loop with no progress feedback and no way to abort mid-scan. Add progress display and abort (e.g. signal or key). |

### Lower priority / hardening

| # | Item | Architecture ref | Notes |
|---|------|------------------|--------|
| 9 | **CLI rendering / streaming** | §14, §17 Phase 10 | `cli/rendering.py` is empty (docstring only). Architecture: "Output formatting, streaming." Add streaming of LLM/tool output and optional formatting helpers. |
| 10 | **OS-level sandbox** | §10, §17 Phase 9 | `sandbox/namespace.py` is a TODO. Add bwrap or Docker isolation for production (--unshare-net, read-only mounts, etc.). |
| 11 | **DOOCS adapter** | Overview diagram | Doc shows "DOOCS" next to EPICS/TANGO. No `doocs.py` or DOOCS adapter; add if needed for a facility. |
| 12 | **Error handling & token budgets** | §17 Phase 10 | ReAct loop and CLI could: handle LLM/adapter errors more explicitly, enforce token budgets, retry/backoff. |
| 13 | **Integration tests** | §17 Phase 10 | Only unit test is `test_plan.py`. Add integration tests: ReAct + simulator, propose_plan → approval → execute (or mock), subagent flow. |
| 14 | **Operator identity for audit** | §9 | Audit log needs operator (e.g. username or config). Decide source (env, config, CLI prompt) and pass through to `audit.record`. |

---

## Summary

- **Implemented:** Core ReAct loop, plans, approval gate, validators, code analysis, all tools (most complete; launch_display stub), LiteLLM, simulator adapter, CLI with main commands, config, basic test.
- **Missing / partial:** Audit log not wired; /history not implemented; execute_code sandbox missing resource limits, safe open, and import blocker; EPICS/TANGO/DOOCS adapters stubbed or absent; launch_display stub; no SCAN progress/abort; no selective approve; rendering/streaming empty; no OS sandbox; limited tests and error-handling hardening.
