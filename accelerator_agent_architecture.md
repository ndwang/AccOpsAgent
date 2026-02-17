# Accelerator Operator Agent — Architecture

## Overview

A CLI-based ReAct agent that assists accelerator operators via natural language. Operators interact through a terminal interface reminiscent of Claude Code. The agent can read machine state, search devices, analyze logs, execute sandboxed Python scripts, and propose control actions for human approval. An orchestrator agent dispatches sub-agents for contained subtasks like device discovery across large namespaces.

```
┌─────────────────────────────────────────────────────────┐
│                      CLI Interface                       │
│              (rich terminal, prompt_toolkit)              │
├─────────────────────────────────────────────────────────┤
│                   Orchestrator Agent                     │
│            (ReAct loop + conversation state)             │
│                                                          │
│  ┌─────────┐  ┌──────────┐  ┌────────────────────────┐  │
│  │ Context  │  │ LLM      │  │ Tool Registry          │  │
│  │ Manager  │  │ Router   │  │                        │  │
│  │(AGENT.md)│  │(LiteLLM  │  │ get_parameter          │  │
│  │          │  │ switcher)│  │ search_device           │  │
│  └─────────┘  └──────────┘  │ read_log                │  │
│                              │ launch_display           │  │
│  ┌──────────────────────┐   │ propose_plan (gated)     │  │
│  │ Sub-Agent Dispatcher  │   │ execute_code             │  │
│  │                      │   │ run_script               │  │
│  │ • device_search      │   │ list_scripts             │  │
│  │ • log_analysis       │   │ spawn_subagent           │  │
│  │ • diagnostics        │   └────────────────────────┘  │
│  └──────────────────────┘                                │
│                              ┌────────────────────────┐  │
│                              │ Safety / Approval Gate  │  │
│                              └────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│              Control System Adapter (ABC)                 │
│         EPICS  │  TANGO  │  DOOCS  │  Custom             │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
accel-agent/
├── AGENT.md                     # Persistent context (facility knowledge, conventions)
├── pyproject.toml
├── accel_agent/
│   ├── __init__.py
│   ├── main.py                  # CLI entry point
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── app.py               # Terminal UI (prompt_toolkit / rich)
│   │   └── rendering.py         # Output formatting, streaming
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py             # Top-level orchestrator wiring
│   │   ├── react_loop.py        # Core think→act→observe cycle
│   │   ├── conversation.py      # Message history management
│   │   ├── context.py           # AGENT.md loader + system prompt builder
│   │   ├── plan.py              # Action plan data model
│   │   ├── plan_executor.py     # Executes approved plans
│   │   └── subagent.py          # Sub-agent spawner & lifecycle
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract LLM interface + response types
│   │   ├── litellm_client.py    # Unified client (LiteLLM — all providers)
│   │   └── router.py            # Named model selection + runtime switching
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py          # Tool registration + schema generation
│   │   ├── base.py              # BaseTool ABC
│   │   ├── get_parameter.py     # Read machine parameter
│   │   ├── search_device.py     # Device name search
│   │   ├── read_log.py          # Log / e-log / archive reading
│   │   ├── launch_display.py    # Open plot / control windows
│   │   ├── propose_plan.py      # Propose action plans (approval-gated)
│   │   ├── execute_code.py      # Write + run new Python in sandbox
│   │   ├── run_script.py        # Run existing scripts in sandbox
│   │   ├── list_scripts.py      # Browse available scripts
│   │   └── spawn_subagent.py    # Dispatch sub-agent
│   ├── safety/
│   │   ├── __init__.py
│   │   ├── approval.py          # Human-in-the-loop plan review
│   │   ├── validators.py        # Parameter range checks
│   │   ├── code_analysis.py     # Static analysis for Python sandbox
│   │   └── audit.py             # Append-only action audit log
│   ├── sandbox/
│   │   ├── __init__.py
│   │   ├── subprocess.py        # Python-level sandbox harness
│   │   └── namespace.py         # OS-level isolation (bwrap / Docker)
│   └── adapters/
│       ├── __init__.py
│       ├── base.py              # ControlSystemAdapter ABC
│       ├── epics.py             # EPICS / Channel Access
│       ├── tango.py             # TANGO
│       └── simulator.py         # Mock adapter for testing
├── configs/
│   ├── default.toml             # Agent configuration
│   ├── devices/                 # Device databases per facility
│   └── safety_limits.toml       # Parameter range limits
├── scripts/                     # Operator analysis scripts (reusable)
└── tests/
```

---

## 1. Core ReAct Loop

The orchestrator runs a standard ReAct cycle. On each iteration it sends the conversation to the LLM, which either produces a final text response or requests tool calls. Read-only tools execute immediately and their results feed back into the next iteration. The one special case is `propose_plan`, which routes through the approval gate before the plan executor runs the approved actions. Everything else is a plain tool call.

```python
# accel_agent/core/react_loop.py
from dataclasses import dataclass, field


@dataclass
class ReActState:
    steps: list = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 30


class ReActLoop:
    """
    Core think→act→observe cycle. Provider-agnostic — takes an LLM
    interface and a tool registry, loops until the LLM produces a
    final text response or hits the iteration limit.

    The loop has no special cases. Every tool — including propose_plan —
    is self-contained and handles its own logic inside execute().
    """

    def __init__(self, llm, tool_registry, context_manager):
        self.llm = llm
        self.tools = tool_registry
        self.context = context_manager

    async def run(self, user_message: str, conversation) -> str:
        state = ReActState()
        conversation.add_user_message(user_message)
        system_prompt = self.context.build_system_prompt()

        while state.iteration < state.max_iterations:
            state.iteration += 1

            response = await self.llm.complete(
                system=system_prompt,
                messages=conversation.messages,
                tools=self.tools.schemas(),
            )

            # No tool calls means the LLM is done — return final answer
            if not response.tool_calls:
                conversation.add_assistant_message(response.text)
                return response.text

            # Process each tool call — no special cases
            tool_results = []
            for call in response.tool_calls:
                tool = self.tools.get(call.name)
                result = await tool.execute(call.input)
                tool_results.append({"call_id": call.id, "result": result})

            conversation.add_assistant_message(response.text, response.tool_calls)
            conversation.add_tool_results(tool_results)

        return "Reached maximum iterations. Please refine your request."
```

---

## 2. Action Plan Model

Every interaction with the control system goes through a plan. A plan is a sequence of four primitive action types that compose to express any operational procedure.

**READ** retrieves the current value of one or more devices. The agent uses this within plans to observe machine response after a write, to capture baseline values, or to collect data at each step of a scan.

**WRITE** sets a device to a new value. Every write in a plan is presented to the operator for approval before any part of the plan executes. The write includes a reason string so the operator understands intent.

**WAIT** pauses execution for a specified duration. Accelerators need settling time between writes and reads — magnets ramp, RF cavities fill, BPM averaging windows complete. The agent specifies dwell times explicitly so the operator can see and adjust them.

**SCAN** is syntactic sugar for a common pattern: sweep one device across a range of values while reading other devices at each step. Without SCAN, the agent would need to emit a long interleaved sequence of WRITE-WAIT-READ actions. SCAN captures the intent declaratively, which makes the plan easier for operators to review and for the executor to optimize (progress bars, early abort, automatic rollback).

```python
# accel_agent/core/plan.py
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(Enum):
    READ = "read"
    WRITE = "write"
    WAIT = "wait"
    SCAN = "scan"


@dataclass
class Action:
    """A single step in an action plan."""
    type: ActionType
    params: dict

    # ── READ params ──────────────────────────────────────
    # {
    #   "devices": ["LINAC:BPM:01:X", "LINAC:BPM:02:X"],
    #   "label": "baseline orbit"         # optional, for display
    # }

    # ── WRITE params ─────────────────────────────────────
    # {
    #   "device": "LINAC:COR:H01:SETPT",
    #   "value": 1.7,
    #   "reason": "Bump corrector by +0.5 mrad"
    # }

    # ── WAIT params ──────────────────────────────────────
    # {
    #   "seconds": 2.0,
    #   "reason": "Settling time for magnet ramp"
    # }

    # ── SCAN params ──────────────────────────────────────
    # {
    #   "write_device": "RF:PHASE:SETPT",
    #   "start": -10.0,
    #   "stop": 10.0,
    #   "steps": 20,
    #   "read_devices": ["DIAG:ENERGY:MEAS"],
    #   "dwell_seconds": 2.0
    # }


@dataclass
class ActionPlan:
    """
    A complete plan proposed by the agent for operator approval.
    Made of an ordered sequence of actions, a human-readable
    description, and optional rollback actions.
    """
    description: str
    actions: list[Action]
    rollback: list[Action] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ActionPlan":
        return cls(
            description=d["description"],
            actions=[Action(type=ActionType(a["type"]), params=a["params"])
                     for a in d["actions"]],
            rollback=[Action(type=ActionType(a["type"]), params=a["params"])
                      for a in d.get("rollback", [])],
        )
```

### Composing Plans from Primitives

Any operational procedure composes from these four actions. Some examples of how common patterns map:

**Write then observe** — a WRITE followed by a WAIT followed by a READ. The operator sees exactly what will be set, how long the agent waits, and what it reads back.

**Batch configuration change** — a sequence of WRITE actions. The operator reviews the full list and approves or cherry-picks.

**Parameter scan** — a single SCAN action. The executor expands it internally into the WRITE-WAIT-READ loop, but the operator reviews and approves the scan as one coherent operation with clear start, stop, step count, and dwell time.

**Multi-step procedure** — any interleaving of the four primitives. For instance: READ (capture baseline), WRITE (change a corrector), WAIT (settle), READ (measure response), WRITE (restore), WAIT, READ (verify restoration). The plan reads like a recipe.

**Scan with setup** — WRITE actions to configure the machine, then a SCAN, then WRITE actions to restore. The entire sequence is one plan, approved as a unit.

---

## 3. Plan Executor

The executor walks through the action list sequentially, dispatching each action to the appropriate handler. It collects results from every step into a structured record that the agent receives back for analysis or reporting.

For SCAN actions, the executor runs the sweep loop autonomously once approved, tracking progress and supporting early abort. After the final step (or on abort), it runs the plan's rollback actions if present.

```python
# accel_agent/core/plan_executor.py
import asyncio
from accel_agent.core.plan import ActionPlan, Action, ActionType


class PlanExecutor:
    """
    Executes an approved action plan step by step.
    Dispatches each action type to the control system adapter
    and collects results into a structured record.
    """

    def __init__(self, adapter, validator):
        self.adapter = adapter
        self.validator = validator

    async def execute(self, plan: ActionPlan) -> dict:
        results = []

        try:
            for i, action in enumerate(plan.actions):
                result = await self._execute_action(action, step=i)
                results.append(result)
        except Exception as e:
            # On failure, attempt rollback
            await self._execute_rollback(plan.rollback)
            return {
                "completed_steps": len(results),
                "total_steps": len(plan.actions),
                "results": results,
                "error": str(e),
                "rolled_back": True,
            }

        # Execute rollback if present (e.g., restore original values after scan)
        if plan.rollback:
            await self._execute_rollback(plan.rollback)

        return {
            "completed_steps": len(results),
            "total_steps": len(plan.actions),
            "results": results,
        }

    async def _execute_action(self, action: Action, step: int) -> dict:
        match action.type:
            case ActionType.READ:
                return await self._do_read(action.params)
            case ActionType.WRITE:
                return await self._do_write(action.params)
            case ActionType.WAIT:
                return await self._do_wait(action.params)
            case ActionType.SCAN:
                return await self._do_scan(action.params)

    async def _do_read(self, params: dict) -> dict:
        readings = {}
        for device in params["devices"]:
            result = await self.adapter.get(device)
            readings[device] = {
                "value": result.value,
                "units": result.units,
                "timestamp": result.timestamp.isoformat(),
                "alarm": result.alarm_status,
            }
        return {
            "action": "read",
            "label": params.get("label"),
            "readings": readings,
        }

    async def _do_write(self, params: dict) -> dict:
        # Validate against safety limits before writing
        self.validator.check(params["device"], params["value"])

        await self.adapter.set(params["device"], params["value"])

        # Read back to confirm
        readback = await self.adapter.get(params["device"])
        return {
            "action": "write",
            "device": params["device"],
            "requested": params["value"],
            "readback": readback.value,
            "success": abs(readback.value - params["value"]) < readback.tolerance,
        }

    async def _do_wait(self, params: dict) -> dict:
        await asyncio.sleep(params["seconds"])
        return {
            "action": "wait",
            "seconds": params["seconds"],
            "reason": params.get("reason", ""),
        }

    async def _do_scan(self, params: dict) -> dict:
        import numpy as np

        setpoints = np.linspace(
            params["start"], params["stop"], params["steps"]
        ).tolist()

        scan_data = []
        for i, sp in enumerate(setpoints):
            # Validate each setpoint
            self.validator.check(params["write_device"], sp)

            # Write
            await self.adapter.set(params["write_device"], sp)

            # Dwell
            await asyncio.sleep(params.get("dwell_seconds", 1.0))

            # Read
            readings = {"setpoint": sp}
            for dev in params["read_devices"]:
                result = await self.adapter.get(dev)
                readings[dev] = result.value

            scan_data.append(readings)

        return {
            "action": "scan",
            "write_device": params["write_device"],
            "range": [params["start"], params["stop"]],
            "points_completed": len(scan_data),
            "data": scan_data,
        }

    async def _execute_rollback(self, rollback_actions: list[Action]):
        for action in rollback_actions:
            try:
                await self._execute_action(action, step=-1)
            except Exception:
                pass  # Best-effort rollback; log the failure
```

---

## 4. Approval Gate

The approval gate presents the entire plan to the operator in a structured terminal UI before any actions execute. Because plans are sequences of well-typed actions, the gate can render each action type appropriately — device and value for writes, device list for reads, duration for waits, and a summary table for scans.

The operator can approve the plan, reject it, or modify parameters (like step count or dwell time in a scan). For plans with multiple write actions, the operator can selectively approve a subset.

```python
# accel_agent/safety/approval.py
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from accel_agent.core.plan import ActionPlan, ActionType

console = Console()


class ApprovalGate:

    async def review_plan(self, plan: ActionPlan) -> bool:
        console.print(Panel(
            plan.description,
            title="[yellow bold]Proposed Action Plan[/]",
            border_style="yellow",
        ))

        # Render each action in the plan
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", width=4)
        table.add_column("Action", width=8)
        table.add_column("Details")

        for i, action in enumerate(plan.actions):
            table.add_row(
                str(i + 1),
                action.type.value.upper(),
                self._describe_action(action),
            )

        console.print(table)

        # Show rollback if present
        if plan.rollback:
            console.print("\n[dim]Rollback on completion/failure:[/]")
            for action in plan.rollback:
                console.print(f"  [dim]← {self._describe_action(action)}[/]")

        console.print()

        # Approval prompt
        has_writes = any(
            a.type in (ActionType.WRITE, ActionType.SCAN)
            for a in plan.actions
        )

        if not has_writes:
            # Read-only plans can auto-approve (or still ask)
            return True

        write_count = sum(
            1 for a in plan.actions
            if a.type in (ActionType.WRITE, ActionType.SCAN)
        )

        if write_count == 1:
            return Confirm.ask("[bold]Approve?[/]", default=False)

        choice = Prompt.ask(
            "[bold]Approve?[/]",
            choices=["all", "none", "modify"],
            default="none",
        )

        if choice == "all":
            console.print("[green]✓ Plan approved[/]")
            return True
        elif choice == "modify":
            return await self._modify_plan(plan)
        else:
            console.print("[red]✗ Plan rejected[/]")
            return False

    async def _modify_plan(self, plan: ActionPlan) -> bool:
        """Let operator adjust scan parameters, dwell times, etc."""
        for i, action in enumerate(plan.actions):
            if action.type == ActionType.SCAN:
                console.print(f"\n[cyan]Scan (step {i + 1}):[/]")
                new_steps = Prompt.ask(
                    f"  Steps [{action.params['steps']}]",
                    default=str(action.params["steps"]),
                )
                action.params["steps"] = int(new_steps)
                new_dwell = Prompt.ask(
                    f"  Dwell seconds [{action.params.get('dwell_seconds', 1.0)}]",
                    default=str(action.params.get("dwell_seconds", 1.0)),
                )
                action.params["dwell_seconds"] = float(new_dwell)

            elif action.type == ActionType.WAIT:
                console.print(f"\n[cyan]Wait (step {i + 1}):[/]")
                new_wait = Prompt.ask(
                    f"  Seconds [{action.params['seconds']}]",
                    default=str(action.params["seconds"]),
                )
                action.params["seconds"] = float(new_wait)

        # Re-display and confirm
        return await self.review_plan(plan)

    def _describe_action(self, action) -> str:
        p = action.params
        match action.type:
            case ActionType.READ:
                devices = ", ".join(p["devices"])
                label = f' ({p["label"]})' if p.get("label") else ""
                return f"{devices}{label}"
            case ActionType.WRITE:
                return (f'{p["device"]} → {p["value"]}'
                        f'  ({p.get("reason", "")})')
            case ActionType.WAIT:
                return (f'{p["seconds"]}s'
                        f'  ({p.get("reason", "")})')
            case ActionType.SCAN:
                return (
                    f'{p["write_device"]}: {p["start"]} → {p["stop"]} '
                    f'({p["steps"]} steps, {p.get("dwell_seconds", 1.0)}s dwell) '
                    f'reading {", ".join(p["read_devices"])}'
                )
```

---

## 5. The `propose_plan` Tool

This is the agent's sole interface to the control system for anything that modifies state. The agent never calls set/write operations directly — it always composes a plan and proposes it. The tool schema teaches the LLM the four action types and how to compose them.

```python
# accel_agent/tools/propose_plan.py
from accel_agent.tools.base import BaseTool
from accel_agent.core.plan import ActionPlan


class ProposePlanTool(BaseTool):
    name = "propose_plan"
    description = """Propose an action plan for operator approval. Use this for
ALL operations that interact with the control system beyond simple reads.

A plan is an ordered sequence of actions. There are four action types:

READ  — Read device values. Use to capture baselines, observe responses,
        or collect data during procedures.
        params: { "devices": ["DEV1", "DEV2"], "label": "optional label" }

WRITE — Set a device to a value. Every write requires operator approval.
        Always capture current values with get_parameter BEFORE proposing
        writes so you can include rollback actions.
        params: { "device": "DEV", "value": 1.23, "reason": "why" }

WAIT  — Pause execution. Use for settling time after writes (magnet ramp,
        RF fill, BPM averaging).
        params: { "seconds": 2.0, "reason": "magnet settling" }

SCAN  — Sweep a device across a range while reading others at each step.
        Shorthand for a WRITE-WAIT-READ loop. Use for parameter scans,
        response measurements, optimization sweeps.
        params: { "write_device": "DEV", "start": -10, "stop": 10,
                  "steps": 20, "read_devices": ["DEV1", "DEV2"],
                  "dwell_seconds": 2.0 }

Compose these to express any procedure. The full plan is shown to the
operator for review — they can approve, reject, or modify parameters.

Include rollback actions to restore original state after the plan completes
or if it fails partway through."""

    def __init__(self, approval_gate, plan_executor):
        self.approval = approval_gate
        self.executor = plan_executor

    def parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Human-readable summary of what this "
                                   "plan does and why.",
                },
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["read", "write", "wait", "scan"],
                            },
                            "params": {
                                "type": "object",
                                "description": "Parameters for this action "
                                               "(see tool description).",
                            },
                        },
                        "required": ["type", "params"],
                    },
                    "description": "Ordered sequence of actions.",
                },
                "rollback": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["read", "write", "wait"]},
                            "params": {"type": "object"},
                        },
                        "required": ["type", "params"],
                    },
                    "description": "Actions to restore original state after "
                                   "completion or on failure.",
                },
            },
            "required": ["description", "actions"],
        }

    async def execute(self, params):
        plan = ActionPlan.from_dict(params)

        approved = await self.approval.review_plan(plan)
        if not approved:
            return {"status": "rejected", "message": "Operator rejected the plan."}

        result = await self.executor.execute(plan)
        return {"status": "completed", "result": result}
```

---

## 6. LLM Abstraction (LiteLLM)

The agent uses **LiteLLM** for a unified interface to multiple LLM providers. One client speaks to any supported provider (Anthropic, OpenAI, etc.) via LiteLLM’s common API; API keys are read from the environment (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). A router holds named models from configuration and supports runtime switching via the `/provider` command.

```python
# accel_agent/llm/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall] | None
    usage: dict | None = None
    raw: object = None


class BaseLLM(ABC):
    """Abstract interface implemented by the LiteLLM client."""

    @abstractmethod
    async def complete(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        ...
```

The LiteLLM client calls `litellm.acompletion()` with OpenAI-style messages and tools. LiteLLM normalizes responses to the OpenAI ChatCompletion shape, so the client parses `choices[0].message` (content and tool_calls) into the shared `LLMResponse` dataclass. Model strings use LiteLLM format: `anthropic/claude-sonnet-4-20250514`, `openai/gpt-4o`, etc.

```python
# accel_agent/llm/litellm_client.py
import json
from litellm import acompletion
from accel_agent.llm.base import BaseLLM, LLMResponse, ToolCall


class LiteLLMClient(BaseLLM):
    """Single client for any provider via LiteLLM. Model: e.g. anthropic/..., openai/..."""

    def __init__(self, model: str, max_tokens: int = 8096):
        self.model = model
        self.max_tokens = max_tokens

    async def complete(self, system, messages, tools=None, temperature=0.0):
        full_messages = [{"role": "system", "content": system}] + list(messages)
        kwargs = {"model": self.model, "messages": full_messages,
                  "temperature": temperature, "max_tokens": self.max_tokens}
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": {"name": t["name"],
                 "description": t["description"], "parameters": t["parameters"]}}
                for t in tools
            ]
            kwargs["tool_choice"] = "auto"
        response = await acompletion(**kwargs)
        return self._parse(response)

    def _parse(self, response) -> LLMResponse:
        msg = response.choices[0].message
        tool_calls = None
        if getattr(msg, "tool_calls", None):
            tool_calls = [ToolCall(id=tc.id, name=tc.function.name,
                                  input=json.loads(tc.function.arguments or "{}"))
                         for tc in msg.tool_calls]
        return LLMResponse(text=getattr(msg, "content", None), tool_calls=tool_calls,
                           usage={...} if getattr(response, "usage", None) else None, raw=response)


# accel_agent/llm/router.py
class LLMRouter:
    """Named model selection from config. Supports runtime switching (/provider <name>)."""

    def __init__(self, config: dict):
        self._clients = {}
        self.default = config.get("default", "default")
        max_tokens = config.get("max_tokens", 8096)
        if not config.get("models"):
            self._clients["default"] = LiteLLMClient(
                model=config.get("model", "anthropic/claude-sonnet-4-20250514"),
                max_tokens=max_tokens)
        for name, model in config.get("models", {}).items():
            self._clients[name] = LiteLLMClient(model=model, max_tokens=max_tokens)
        if self._clients and self.default not in self._clients:
            self.default = next(iter(self._clients))

    def get(self, provider: str | None = None) -> BaseLLM:
        return self._clients.get(provider or self.default) or next(iter(self._clients.values()))
```

Config example (`configs/default.toml`):

```toml
[llm]
model = "anthropic/claude-sonnet-4-20250514"
max_tokens = 8096
default = "claude"
[llm.models]
claude = "anthropic/claude-sonnet-4-20250514"
gpt4 = "openai/gpt-4o"
```

---

## 7. Tool System

All tools share a base interface with a name, description, JSON schema for parameters, and an async execute method. A registry collects tools and generates the schema list that gets passed to the LLM. The registry also provides filtered subsets for sub-agents.

```python
# accel_agent/tools/base.py
from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    def parameters_schema(self) -> dict:
        """JSON Schema for tool input."""
        ...

    @abstractmethod
    async def execute(self, params: dict) -> Any:
        """Run the tool. Returns a result dict for the LLM."""
        ...

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema(),
        }


# accel_agent/tools/registry.py
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        return self._tools[name]

    def schemas(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]

    def subset(self, names: list[str]) -> "ToolRegistry":
        """Return a registry containing only the named tools."""
        sub = ToolRegistry()
        for name in names:
            if name in self._tools:
                sub.register(self._tools[name])
        return sub
```

### Read Tools

```python
# accel_agent/tools/get_parameter.py
class GetParameterTool(BaseTool):
    name = "get_parameter"
    description = (
        "Read the current value of a machine parameter or device. "
        "Returns the value, units, timestamp, and alarm status. "
        "Use this to check current state before proposing plans."
    )

    def __init__(self, adapter):
        self.adapter = adapter

    def parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "device_name": {
                    "type": "string",
                    "description": "Full device/PV name (e.g., 'LINAC:BPM:01:X')",
                },
            },
            "required": ["device_name"],
        }

    async def execute(self, params):
        result = await self.adapter.get(params["device_name"])
        return {
            "device": params["device_name"],
            "value": result.value,
            "units": result.units,
            "timestamp": result.timestamp.isoformat(),
            "alarm": result.alarm_status,
        }


# accel_agent/tools/search_device.py
class SearchDeviceTool(BaseTool):
    name = "search_device"
    description = (
        "Search for device names matching a pattern. Supports wildcards. "
        "Use this to find device names when the operator refers to devices "
        "by informal names or subsystem (e.g., 'linac BPMs')."
    )

    def __init__(self, adapter):
        self.adapter = adapter

    def parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern with wildcards (e.g., 'LINAC:BPM*')",
                },
                "subsystem": {
                    "type": "string",
                    "description": "Optional subsystem filter",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, params):
        results = await self.adapter.search_devices(
            params["pattern"], params.get("subsystem")
        )
        return {"matches": results, "count": len(results)}


# accel_agent/tools/read_log.py
class ReadLogTool(BaseTool):
    name = "read_log"
    description = (
        "Read log entries from the e-log, alarm log, or data archive. "
        "Specify a time range and optional filters."
    )

    def __init__(self, adapter):
        self.adapter = adapter

    def parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "enum": ["elog", "alarm", "archive"],
                },
                "start_time": {"type": "string", "description": "ISO 8601"},
                "end_time": {"type": "string", "description": "ISO 8601"},
                "filters": {
                    "type": "object",
                    "description": "Optional key-value filters (e.g., subsystem, severity)",
                },
            },
            "required": ["source", "start_time", "end_time"],
        }

    async def execute(self, params):
        from datetime import datetime
        entries = await self.adapter.read_log(
            source=params["source"],
            time_range=(
                datetime.fromisoformat(params["start_time"]),
                datetime.fromisoformat(params["end_time"]),
            ),
            filters=params.get("filters"),
        )
        return {"entries": entries, "count": len(entries)}
```

### Complete Tool Set

```
TOOLS:
├── get_parameter      — Read a device value
├── search_device      — Search device names by pattern
├── read_log           — Read e-log / alarm log / archive
├── launch_display     — Open plot/control windows (colleague's impl)
├── propose_plan       — Propose action plans for approval
├── execute_code       — Write and run new Python scripts (sandboxed)
├── run_script         — Run existing analysis scripts (sandboxed)
├── list_scripts       — Browse available scripts
└── spawn_subagent     — Dispatch sub-agent for contained tasks
```

---

## 8. Control System Adapter

The control system is abstracted behind an interface so the agent is portable across facilities. Each facility implements the adapter for their own system. The agent and all tools interact with the machine exclusively through this interface.

The adapter is responsible for its own safety checks — this is the third and final safety layer. When the agent's plan executor calls `set()`, the adapter can enforce hardware interlocks, rate limits, or any facility-specific protections independently of the agent's validation.

```python
# accel_agent/adapters/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ReadResult:
    value: float | str | list
    units: str
    timestamp: datetime
    alarm_status: str       # "OK", "MINOR", "MAJOR", "INVALID"
    tolerance: float = 0.0  # for readback verification


class ControlSystemAdapter(ABC):
    """
    Abstract interface to the accelerator control system.
    Each facility implements this for their system (EPICS, TANGO, etc.).
    """

    @abstractmethod
    async def get(self, device_name: str) -> ReadResult:
        ...

    @abstractmethod
    async def set(self, device_name: str, value) -> None:
        ...

    @abstractmethod
    async def search_devices(
        self, pattern: str, subsystem: str | None = None
    ) -> list[dict]:
        ...

    @abstractmethod
    async def read_log(
        self, source: str, time_range: tuple, filters: dict | None = None
    ) -> list[dict]:
        ...


# accel_agent/adapters/simulator.py
class SimulatorAdapter(ControlSystemAdapter):
    """Mock adapter for development and testing."""

    def __init__(self):
        self._state = {}

    async def get(self, device_name):
        return ReadResult(
            value=self._state.get(device_name, 0.0),
            units="mm", timestamp=datetime.now(),
            alarm_status="OK", tolerance=0.01,
        )

    async def set(self, device_name, value):
        self._state[device_name] = value

    async def search_devices(self, pattern, subsystem=None):
        import re
        regex = re.compile(pattern.replace("*", ".*"), re.IGNORECASE)
        return [{"name": k, "description": "simulated"}
                for k in self._state if regex.match(k)]

    async def read_log(self, source, time_range, filters=None):
        return [{"timestamp": datetime.now().isoformat(),
                 "message": "Simulated log entry"}]
```

---

## 9. Safety System

Safety operates in three independent layers. Each layer can reject an operation without relying on the others.

**Layer 1 — Human approval.** Every plan that contains WRITE or SCAN actions is presented to the operator for review before execution. The operator sees the full action sequence, can modify parameters, and can approve, reject, or selectively approve actions. This is implemented in the approval gate described in Section 4.

**Layer 2 — Agent-side validation.** The `ParameterValidator` checks every write value against predefined safety limits loaded from a TOML configuration file. If a value falls outside the allowed range for a device, the write is rejected with a clear error before it reaches the control system. Limits can be defined per device or by wildcard pattern.

**Layer 3 — Control system enforcement.** The adapter's `set()` implementation handles hardware interlocks, permission checks, and any facility-specific safety logic. This layer operates independently of the agent and cannot be bypassed by it.

```python
# accel_agent/safety/validators.py
import tomllib


class ParameterValidator:
    """Range validation loaded from safety_limits.toml."""

    def __init__(self, limits_path: str = "configs/safety_limits.toml"):
        with open(limits_path, "rb") as f:
            self.limits = tomllib.load(f)

    def check(self, device_name: str, value) -> None:
        limits = self._find_limits(device_name)
        if limits is None:
            raise SafetyError(
                f"No safety limits defined for {device_name}. "
                f"Cannot write without defined limits."
            )
        if not (limits["min"] <= value <= limits["max"]):
            raise SafetyError(
                f"Value {value} for {device_name} outside safe range "
                f"[{limits['min']}, {limits['max']}] {limits.get('units', '')}"
            )

    def _find_limits(self, device_name):
        if device_name in self.limits:
            return self.limits[device_name]
        for pattern, limits in self.limits.items():
            if "*" in pattern:
                import fnmatch
                if fnmatch.fnmatch(device_name, pattern):
                    return limits
        return None


class SafetyError(Exception):
    pass
```

```toml
# configs/safety_limits.toml

["LINAC:QUAD:*:CURRENT"]
min = -50.0
max = 50.0
units = "A"

["LINAC:RF:*:PHASE"]
min = -180.0
max = 180.0
units = "deg"

["RING:BPM:*:X"]
min = -10.0
max = 10.0
units = "mm"
```

**Audit logging.** Every action the agent takes — whether approved, rejected, or failed — is written to an append-only JSONL log. Each entry records the timestamp, operator identity, the action attempted, and the outcome.

```python
# accel_agent/safety/audit.py
import json
from datetime import datetime
from pathlib import Path


class AuditLog:
    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def record(self, plan: dict, approved: bool, operator: str,
               result: dict | None = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operator": operator,
            "plan": plan,
            "approved": approved,
            "result": result,
        }
        log_file = self.log_dir / f"{datetime.now():%Y-%m-%d}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

---

## 10. Python Code Execution Sandbox

The code executor runs Python scripts for data analysis in a sandbox that is physically unable to reach the control system. Data comes from files and databases, not live reads. The sandbox enforces isolation through three independent mechanisms.

### Static Analysis (Pre-Execution Gate)

Before execution, an AST-based analyzer scans the script for blocked imports (control system libraries like `epics`, `tango`, `pydoocs`, as well as network and subprocess modules) and dangerous calls (`os.system`, `eval`, etc.). This catches obvious violations immediately so the agent can rewrite the script without waiting for a subprocess to fail. The analyzer maintains an explicit allowlist of data analysis libraries (numpy, scipy, pandas, matplotlib, h5py, lmfit, sklearn, etc.) and flags anything not on either list as a warning.

```python
# accel_agent/safety/code_analysis.py
import ast
from dataclasses import dataclass


@dataclass
class AnalysisResult:
    safe: bool
    violations: list[str]
    warnings: list[str]


BLOCKED_IMPORTS = {
    # Control system
    "epics", "pcaspy", "pyepics", "cothread",
    "tango", "PyTango", "taurus",
    "pydoocs",
    # Network
    "subprocess", "shutil", "ctypes",
    "socket", "http", "urllib", "requests",
    "httpx", "aiohttp", "ftplib", "smtplib",
    "xmlrpc", "paramiko", "fabric",
    # Shell / eval
    "code", "codeop",
}

ALLOWED_IMPORTS = {
    "numpy", "scipy", "pandas", "matplotlib", "matplotlib.pyplot",
    "json", "csv", "math", "statistics", "datetime", "time",
    "collections", "itertools", "functools", "dataclasses",
    "typing", "re", "io", "os.path", "glob",
    "h5py", "tables", "sqlite3",
    "struct", "pickle",
    "logging", "warnings",
    "sklearn", "lmfit",
    "PIL", "skimage",
}

BLOCKED_CALLS = {
    "exec", "eval", "compile", "__import__",
    "os.system", "os.popen",
    "os.remove", "os.unlink", "os.rmdir", "os.rename",
    "os.chmod", "os.chown",
}


class CodeAnalyzer:
    def analyze(self, code: str) -> AnalysisResult:
        violations, warnings = [], []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return AnalysisResult(False, [f"Syntax error: {e}"], [])

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._check_import(alias.name, violations, warnings)
            elif isinstance(node, ast.ImportFrom) and node.module:
                self._check_import(node.module, violations, warnings)
            elif isinstance(node, ast.Call):
                name = self._get_call_name(node)
                if name in BLOCKED_CALLS:
                    violations.append(
                        f"Blocked call: {name}() at line {node.lineno}"
                    )

        return AnalysisResult(len(violations) == 0, violations, warnings)

    def _check_import(self, module, violations, warnings):
        root = module.split(".")[0]
        if root in BLOCKED_IMPORTS or module in BLOCKED_IMPORTS:
            violations.append(f"Blocked import: {module}")
        elif root not in ALLOWED_IMPORTS and module not in ALLOWED_IMPORTS:
            warnings.append(f"Unknown import: {module}")

    def _get_call_name(self, node):
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            n = node.func
            while isinstance(n, ast.Attribute):
                parts.append(n.attr)
                n = n.value
            if isinstance(n, ast.Name):
                parts.append(n.id)
            return ".".join(reversed(parts))
        return ""
```

### Subprocess Sandbox

Scripts run in a separate process with resource limits, a controlled `open()` function that enforces filesystem policy, and a runtime import blocker on `sys.meta_path`. The harness wraps user code with a preamble that sets up three path variables available to every script:

- **`DATA_DIR`** / **`DATA_DIRS`** — read-only directories containing archived data, log files, and databases.
- **`SCRIPTS_DIR`** — read-only directory containing existing analysis scripts.
- **`WORK_DIR`** — the only writable directory, for output files like plots and CSVs.

The `open()` override resolves paths and checks whether the operation is a read (allowed from data and scripts dirs) or a write (allowed only in work dir). Any other path raises `PermissionError`. The harness also sets `RLIMIT_NPROC` to zero to prevent spawning child processes, and forces matplotlib to use the non-interactive Agg backend.

```python
# accel_agent/tools/execute_code.py
import asyncio
import tempfile
import textwrap
from pathlib import Path
from accel_agent.tools.base import BaseTool
from accel_agent.safety.code_analysis import CodeAnalyzer


class ExecuteCodeTool(BaseTool):
    name = "execute_code"
    description = """Execute a Python script for data analysis in a sandbox.

Available libraries: numpy, scipy, pandas, matplotlib, h5py, lmfit, sklearn.

DATA ACCESS:
- Read files from DATA_DIR / DATA_DIRS (archived data, databases)
- Read files from SCRIPTS_DIR (existing analysis scripts)
- Write output files (plots, CSVs) to WORK_DIR
- Query databases via sqlite3

RESTRICTIONS:
- No network access
- No control system access (use get_parameter for live reads instead)
- Cannot write outside WORK_DIR
- Cannot spawn subprocesses

Stdout, stderr, and a list of output files in WORK_DIR are returned."""

    def __init__(self, config: dict):
        self.analyzer = CodeAnalyzer()
        self.data_dirs = config.get("data_dirs", [])
        self.scripts_dir = config.get("scripts_dir", "")
        self.work_dir = Path(config.get("work_dir", "/tmp/agent-workspace"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = config.get("timeout_seconds", 120)
        self.max_memory_mb = config.get("max_memory_mb", 2048)

    def parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "description": {"type": "string", "description": "What this script does"},
                "save_as": {
                    "type": "string",
                    "description": "Optional filename to save script for reuse",
                },
            },
            "required": ["code", "description"],
        }

    async def execute(self, params):
        code = params["code"]

        # Static analysis gate
        analysis = self.analyzer.analyze(code)
        if not analysis.safe:
            return {
                "status": "blocked",
                "violations": analysis.violations,
                "hint": "Rewrite without blocked imports/calls. "
                        "Use DATA_DIR for reads, WORK_DIR for writes.",
            }

        # Save for reuse if requested
        if params.get("save_as"):
            self._save_script(params["save_as"], code, params["description"])

        # Execute in subprocess
        result = await self._run_sandboxed(code)
        output_files = self._collect_outputs()

        return {
            "status": "success" if result["returncode"] == 0 else "error",
            "stdout": result["stdout"][-5000:],
            "stderr": result["stderr"][-2000:],
            "output_files": output_files,
            "warnings": analysis.warnings,
        }

    async def _run_sandboxed(self, code: str) -> dict:
        wrapped = self._wrap_code(code)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(wrapped)
            script_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._sandbox_env(),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return {"returncode": -1, "stdout": "",
                        "stderr": f"Timed out after {self.timeout}s"}

            return {
                "returncode": proc.returncode,
                "stdout": stdout.decode(errors="replace"),
                "stderr": stderr.decode(errors="replace"),
            }
        finally:
            Path(script_path).unlink(missing_ok=True)

    def _wrap_code(self, code: str) -> str:
        """Prepend sandbox harness: resource limits, safe open(), import blocker."""
        data_dirs_repr = repr(self.data_dirs)
        return textwrap.dedent(f"""\
            import resource, sys, os

            # Resource limits
            resource.setrlimit(resource.RLIMIT_AS,
                ({self.max_memory_mb} * 1024 * 1024,
                 {self.max_memory_mb} * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))

            # Matplotlib non-interactive
            import matplotlib
            matplotlib.use('Agg')

            # Path variables
            DATA_DIR = {repr(str(self.data_dirs[0]) if self.data_dirs else '/data')}
            DATA_DIRS = {data_dirs_repr}
            WORK_DIR = {repr(str(self.work_dir))}
            SCRIPTS_DIR = {repr(str(self.scripts_dir))}

            # Safe open()
            _builtin_open = open
            _READ_DIRS = {data_dirs_repr} + [{repr(str(self.scripts_dir))}]
            _WRITE_DIR = {repr(str(self.work_dir))}

            def _safe_open(path, mode='r', *a, **kw):
                resolved = os.path.realpath(path)
                if any(m in mode for m in ('w', 'a', 'x')):
                    if not resolved.startswith(_WRITE_DIR):
                        raise PermissionError(
                            f"Cannot write to {{path}}. Use WORK_DIR.")
                else:
                    ok = any(resolved.startswith(os.path.realpath(d))
                             for d in _READ_DIRS) or resolved.startswith(_WRITE_DIR)
                    if not ok:
                        raise PermissionError(
                            f"Cannot read {{path}}. Use DATA_DIR or SCRIPTS_DIR.")
                return _builtin_open(path, mode, *a, **kw)

            import builtins
            builtins.open = _safe_open

            # Import blocker
            _BLOCKED = {{
                'subprocess', 'shutil', 'socket', 'http', 'urllib',
                'requests', 'httpx', 'aiohttp', 'ftplib', 'smtplib',
                'xmlrpc', 'paramiko', 'ctypes', 'epics', 'pcaspy',
                'pyepics', 'tango', 'PyTango', 'taurus', 'pydoocs',
            }}

            class _ImportBlocker:
                def find_module(self, name, path=None):
                    if name.split('.')[0] in _BLOCKED:
                        return self
                    return None
                def load_module(self, name):
                    raise ImportError(
                        f"Module '{{name}}' is blocked in the sandbox.")

            sys.meta_path.insert(0, _ImportBlocker())

            # ── User code ──
        """) + code

    def _sandbox_env(self) -> dict:
        return {
            "PATH": "/usr/bin:/usr/local/bin",
            "HOME": str(self.work_dir),
            "MPLBACKEND": "Agg",
            "PYTHONHASHSEED": "0",
            # Control system env vars explicitly excluded
        }

    def _save_script(self, filename, code, description):
        path = Path(self.scripts_dir)
        path.mkdir(parents=True, exist_ok=True)
        header = f'"""\n{description}\nGenerated by accelerator agent\n"""\n\n'
        (path / filename).write_text(header + code)

    def _collect_outputs(self) -> list[str]:
        return [str(p.relative_to(self.work_dir))
                for p in self.work_dir.rglob("*") if p.is_file()]
```

### OS-Level Isolation

The Python-level sandbox is the fast, convenient layer. For production use in a control room, an OS-level isolation layer prevents any bypass. The recommended approach is bubblewrap (`bwrap`), which uses Linux namespaces to create a minimal sandbox. The key property is `--unshare-net`, which creates an empty network namespace — even if a script somehow circumvents the Python-level import blocker, TCP/UDP to the control system is impossible at the kernel level. Data directories mount read-only, only the workspace mounts read-write, and no home directory, configuration files, or control system environment variables are visible.

Docker is an alternative for environments where bubblewrap is unavailable. The container runs with `--network none`, `--read-only`, `--cap-drop=ALL`, and `--security-opt=no-new-privileges`.

For development, the Python-level sandbox alone is sufficient. Add bwrap or Docker before deploying to operators.

---

## 11. Script Reuse Tools

Operators accumulate analysis scripts over time. The agent can discover, inspect, and run existing scripts, or write new ones and save them for future use.

`list_scripts` lists available scripts in the scripts directory, extracting each file's docstring as a description. An optional keyword filter narrows results.

`run_script` runs an existing script in the same sandbox as `execute_code`. It takes a script name and optional command-line arguments, which are injected as `sys.argv`. The script reads data from DATA_DIR and writes output to WORK_DIR like any sandboxed code.

`execute_code` with the `save_as` parameter writes a new script and runs it. The script is saved with a docstring header so `list_scripts` can discover and describe it later.

```python
# accel_agent/tools/run_script.py
class RunExistingScriptTool(BaseTool):
    name = "run_script"
    description = """Run an existing analysis script from the scripts directory.
Scripts run in the same sandbox as execute_code. Pass arguments via 'args'
which will be available as sys.argv in the script."""

    def __init__(self, config):
        self.scripts_dir = Path(config["scripts_dir"])
        self.executor = ExecuteCodeTool(config)

    def parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "script_name": {"type": "string"},
                "args": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "string"},
            },
            "required": ["script_name", "description"],
        }

    async def execute(self, params):
        path = self.scripts_dir / params["script_name"]
        if not path.exists():
            available = [f.name for f in self.scripts_dir.glob("*.py")]
            return {"status": "error",
                    "message": f"Script '{params['script_name']}' not found.",
                    "available_scripts": available}

        code = path.read_text()
        if params.get("args"):
            args_repr = repr([params["script_name"]] + params["args"])
            code = f"import sys; sys.argv = {args_repr}\n" + code

        return await self.executor.execute({
            "code": code, "description": params["description"]
        })


# accel_agent/tools/list_scripts.py
class ListScriptsTool(BaseTool):
    name = "list_scripts"
    description = "List available analysis scripts with descriptions."

    def __init__(self, config):
        self.scripts_dir = Path(config["scripts_dir"])

    def parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "filter": {"type": "string", "description": "Keyword filter"},
            },
        }

    async def execute(self, params):
        scripts = []
        for path in sorted(self.scripts_dir.glob("*.py")):
            import ast
            try:
                doc = ast.get_docstring(ast.parse(path.read_text()))
            except SyntaxError:
                doc = None
            scripts.append({
                "name": path.name,
                "description": doc or "(no description)",
            })

        if params.get("filter"):
            kw = params["filter"].lower()
            scripts = [s for s in scripts
                       if kw in s["name"].lower()
                       or kw in s["description"].lower()]

        return {"scripts": scripts, "count": len(scripts)}
```

---

## 12. Sub-Agent System

The orchestrator dispatches sub-agents for tasks that require many intermediate tool calls in a contained domain. The canonical example is device name discovery: when the operator says "all BPMs in the linac," the orchestrator doesn't know the device names and searching through a 100k+ device namespace would consume its context window. Instead it spawns a sub-agent specialized in device search.

Each sub-agent gets its own ReAct loop with a fresh conversation, a restricted tool set (read-only tools appropriate to the task type), and a focused system prompt. The sub-agent runs to completion and returns a structured result. The orchestrator sees only the final output, not the intermediate search steps. Sub-agents never have access to `propose_plan` or any write capability.

```python
# accel_agent/core/subagent.py

SUBAGENT_TOOL_SETS = {
    "device_search": ["search_device", "read_log"],
    "log_analysis":  ["read_log", "execute_code"],
    "diagnostics":   ["read_log", "get_parameter", "search_device"],
}

SUBAGENT_PROMPTS = {
    "device_search": (
        "You are a sub-agent specialized in finding device names. "
        "Search systematically: start with broad patterns, narrow down. "
        "Return a structured list of matching device names with descriptions."
    ),
    "log_analysis": (
        "You are a sub-agent specialized in reading and analyzing logs. "
        "Extract relevant entries, identify patterns, and return "
        "structured findings."
    ),
    "diagnostics": (
        "You are a sub-agent specialized in diagnosing machine issues. "
        "Read parameters, check logs, cross-reference, and identify "
        "potential causes."
    ),
}


class SubAgentDispatcher:
    def __init__(self, llm, tool_registry, context_manager):
        self.llm = llm
        self.full_registry = tool_registry
        self.context = context_manager

    async def run(self, task_type: str, instruction: str) -> dict:
        sub_tools = self.full_registry.subset(
            SUBAGENT_TOOL_SETS.get(task_type, [])
        )

        system_prompt = (
            self.context.build_system_prompt()
            + "\n\n" + SUBAGENT_PROMPTS.get(task_type, "")
        )

        from accel_agent.core.react_loop import ReActLoop
        from accel_agent.core.conversation import Conversation

        loop = ReActLoop(
            llm=self.llm,
            tool_registry=sub_tools,
            context_manager=self.context,
        )

        sub_conversation = Conversation()
        result = await loop.run(instruction, sub_conversation)

        return {
            "task_type": task_type,
            "result": result,
            "steps_taken": sub_conversation.tool_call_count,
        }


# accel_agent/tools/spawn_subagent.py
class SpawnSubagentTool(BaseTool):
    name = "spawn_subagent"
    description = """Dispatch a sub-agent for a contained subtask. Sub-agents run
with a restricted read-only tool set and return structured results.

Task types:
- "device_search": Find device names by pattern across large namespaces
- "log_analysis": Read and analyze log entries, extract patterns
- "diagnostics": Cross-reference parameters and logs to diagnose issues

Use this when a task requires many intermediate lookups that would
clutter the main conversation (e.g., finding all BPMs in a sector
out of 100,000+ devices)."""

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher

    def parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "enum": ["device_search", "log_analysis", "diagnostics"],
                },
                "instruction": {
                    "type": "string",
                    "description": "Detailed instruction for the sub-agent",
                },
            },
            "required": ["task_type", "instruction"],
        }

    async def execute(self, params):
        return await self.dispatcher.run(
            task_type=params["task_type"],
            instruction=params["instruction"],
        )
```

---

## 13. Context System

The agent loads persistent context from an `AGENT.md` file at the project root, modeled after Claude Code's `CLAUDE.md`. This file contains facility-specific knowledge: device naming conventions, common procedures, subsystem descriptions, safety notes, and operational tips. The context manager reads this file and any additional Markdown files from configured directories, then injects them into the system prompt.

Operators and accelerator physicists maintain `AGENT.md` like documentation — it's a plain text file versionable in git. The LLM uses it to understand facility-specific vocabulary, device naming patterns, and standard workflows without needing RAG or a vector database.

```python
# accel_agent/core/context.py
from pathlib import Path


SYSTEM_PROMPT_BASE = """\
You are an accelerator operator assistant. You help operators interact with \
the particle accelerator control system using natural language.

## Core Principles
- SAFETY FIRST: All write operations require human approval via propose_plan.
- Always read current values with get_parameter BEFORE proposing writes.
- When uncertain about device names, use spawn_subagent to search.
- Explain your reasoning so operators can make informed decisions.
- Flag any operation that could affect beam delivery.

## Action Plans
Use propose_plan for ALL control system modifications. Compose plans from:
- READ: capture values (baselines, responses, data collection)
- WRITE: set a device to a value (requires approval)
- WAIT: pause for settling time (magnets, RF, diagnostics)
- SCAN: sweep a parameter while reading (approved as one operation)
Include rollback actions to restore state after completion or failure.

## Code Execution
Use execute_code for data analysis. Scripts run in a sandbox with no
control system access. Data comes from files in DATA_DIR or databases.
Check list_scripts for existing scripts before writing new ones.

## Response Style
- Concise and operationally focused.
- Use accelerator physics terminology appropriate for control room staff.
- Present data in structured form when useful.
"""


class ContextManager:
    def __init__(self, agent_md_path: str = "AGENT.md",
                 extra_context_dirs: list[str] | None = None):
        self.agent_md_path = Path(agent_md_path)
        self.extra_dirs = extra_context_dirs or []

    def build_system_prompt(self) -> str:
        parts = [SYSTEM_PROMPT_BASE]

        if self.agent_md_path.exists():
            content = self.agent_md_path.read_text()
            parts.append(f"<agent_context>\n{content}\n</agent_context>")

        for d in self.extra_dirs:
            for f in sorted(Path(d).glob("*.md")):
                parts.append(
                    f'<context file="{f.name}">\n{f.read_text()}\n</context>'
                )

        return "\n\n".join(parts)
```

An example `AGENT.md` for a facility:

```markdown
# AGENT.md — SLAC LCLS-II

## Facility
LCLS-II is an X-ray free-electron laser at SLAC National Accelerator
Laboratory. The control system is EPICS with Channel Access and pvAccess.

## Naming Convention
Devices follow: SECTOR:TYPE:INSTANCE:ATTRIBUTE
- LINAC:QUAD:01:BACT   — quad actual magnetic field
- LINAC:BPM:01:X       — BPM horizontal position
- UNDH:PHASE:01:SET     — undulator phase setpoint

## Common Tasks
- Shift summary: read elog for the last 8 hours, summarize key events
- Orbit plot: find BPMs by sector, read archive data, plot
- RF trip diagnosis: check RF station logs, read interlock status

## Safety Notes
- Never set RF phase while beam is on without explicit confirmation
- Quad current changes > 5A require section lead approval

## Data Locations
- Archive data: /data/archive/ (HDF5 files, one per day)
- E-log database: /data/elog/elog.db (SQLite)
- BPM calibration: /data/calibration/bpm/
```

---

## 14. CLI Interface

The terminal interface uses `prompt_toolkit` for input with history and `rich` for formatted output. Slash commands provide meta-operations like switching LLM models (LiteLLM named models), viewing the loaded context, and clearing conversation history.

```python
# accel_agent/cli/app.py
import asyncio
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown

console = Console()


class AgentCLI:
    def __init__(self, agent):
        self.agent = agent
        self.session = PromptSession(history=FileHistory(".agent_history"))

    async def run(self):
        console.print("[bold green]Accelerator Agent[/] ready. "
                      "Type /help for commands.\n")

        while True:
            try:
                user_input = await asyncio.to_thread(
                    self.session.prompt, "operator> "
                )
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input.strip():
                continue

            if user_input.startswith("/"):
                await self._handle_command(user_input)
                continue

            with console.status("[bold cyan]Thinking...[/]"):
                response = await self.agent.run(user_input)

            console.print(Markdown(response))
            console.print()

    async def _handle_command(self, cmd):
        parts = cmd.strip().split()
        match parts[0]:
            case "/help":
                console.print(
                    "/help      — Show this message\n"
                    "/context   — Show loaded AGENT.md\n"
                    "/history   — Show conversation summary\n"
                    "/provider  — Switch LLM model (e.g., /provider gpt4)\n"
                    "/clear     — Clear conversation\n"
                    "/quit      — Exit"
                )
            case "/provider":
                if len(parts) > 1:
                    self.agent.switch_provider(parts[1])
                    console.print(f"Switched to [bold]{parts[1]}[/]")
            case "/context":
                console.print(Markdown(
                    self.agent.context.agent_md_path.read_text()
                ))
            case "/clear":
                self.agent.clear_conversation()
                console.print("Conversation cleared.")
            case "/quit":
                raise SystemExit()
```

---

## 15. Top-Level Wiring

```python
# accel_agent/core/agent.py

class OrchestratorAgent:
    """Wires all components together."""

    def __init__(self, config: dict):
        # LLM (LiteLLM: named models from config)
        self.llm_router = LLMRouter(config.get("llm", {}))
        self.llm = self.llm_router.get()

        # Adapter
        self.adapter = self._make_adapter(config["adapter"])

        # Context
        self.context = ContextManager(
            agent_md_path=config.get("agent_md", "AGENT.md"),
        )

        # Safety
        self.approval_gate = ApprovalGate()
        self.validator = ParameterValidator(
            config.get("safety_limits", "configs/safety_limits.toml")
        )
        self.audit = AuditLog()

        # Plan executor
        self.plan_executor = PlanExecutor(self.adapter, self.validator)

        # Tools
        self.tools = ToolRegistry()
        self._register_tools(config)

        # Sub-agents
        self.dispatcher = SubAgentDispatcher(
            self.llm, self.tools, self.context
        )
        self.tools.register(SpawnSubagentTool(self.dispatcher))

        # Core loop — no special cases, just LLM + tools + context
        self.react_loop = ReActLoop(
            self.llm, self.tools, self.context
        )
        self.conversation = Conversation()

    def _register_tools(self, config):
        self.tools.register(GetParameterTool(self.adapter))
        self.tools.register(SearchDeviceTool(self.adapter))
        self.tools.register(ReadLogTool(self.adapter))
        self.tools.register(LaunchDisplayTool())
        self.tools.register(ProposePlanTool(self.approval_gate, self.plan_executor))
        self.tools.register(ExecuteCodeTool(config.get("sandbox", {})))
        self.tools.register(RunExistingScriptTool(config.get("sandbox", {})))
        self.tools.register(ListScriptsTool(config.get("sandbox", {})))

    async def run(self, user_message: str) -> str:
        return await self.react_loop.run(user_message, self.conversation)

    def switch_provider(self, name: str):
        self.llm = self.llm_router.get(name)
        self.react_loop.llm = self.llm
        self.dispatcher.llm = self.llm

    def clear_conversation(self):
        self.conversation = Conversation()


# accel_agent/main.py
import asyncio
import tomllib


def main():
    with open("configs/default.toml", "rb") as f:
        config = tomllib.load(f)

    agent = OrchestratorAgent(config)
    cli = AgentCLI(agent)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
```

---

## 16. Example Flows

### Write then observe

```
Operator: "Bump corrector COR:H01 by +0.5 mrad, check BPM:01 and BPM:02"

Agent:
  get_parameter("COR:H01:SETPT")    → 1.2 mrad
  get_parameter("BPM:01:X")         → 0.34 mm
  get_parameter("BPM:02:X")         → -0.12 mm

  propose_plan:
    description: "Bump COR:H01 by +0.5 mrad and observe orbit response"
    actions:
      1. WRITE  COR:H01:SETPT → 1.7  (reason: operator requested +0.5 bump)
      2. WAIT   2.0s  (reason: corrector settling)
      3. READ   [BPM:01:X, BPM:02:X]  (label: orbit after bump)
    rollback:
      1. WRITE  COR:H01:SETPT → 1.2  (reason: restore original)

  [Operator approves]
  [Executor runs: write → wait → read]
  [Agent reports before/after comparison]
```

### Parameter scan

```
Operator: "Scan RF phase from -10 to +10 deg, measure beam energy"

Agent:
  spawn_subagent("device_search", "Find RF phase setpoint and
    energy measurement devices")
    → {RF:PHASE:SETPT, DIAG:ENERGY:MEAS}

  get_parameter("RF:PHASE:SETPT")   → 0.0 deg

  propose_plan:
    description: "RF phase scan to measure energy response"
    actions:
      1. SCAN   RF:PHASE:SETPT: -10 → +10 (20 steps, 2.0s dwell)
                reading [DIAG:ENERGY:MEAS]
    rollback:
      1. WRITE  RF:PHASE:SETPT → 0.0  (reason: restore original)

  [Operator reviews scan table, adjusts steps to 10, approves]
  [Executor runs all 10 steps autonomously]
  [Agent receives scan data, writes plot with execute_code]
```

### Batch configuration

```
Operator: "Load the low-beta optics for quads Q1–Q5"

Agent:
  [reads AGENT.md for low-beta settings]
  get_parameter for Q1–Q5  → current values

  propose_plan:
    description: "Switch to low-beta optics configuration"
    actions:
      1. WRITE  RING:Q1:CURRENT → 45.2  (reason: low-beta optics)
      2. WRITE  RING:Q2:CURRENT → -32.1
      3. WRITE  RING:Q3:CURRENT → 28.7
      4. WRITE  RING:Q4:CURRENT → -41.5
      5. WRITE  RING:Q5:CURRENT → 36.9
    rollback:
      1. WRITE  RING:Q1:CURRENT → 40.0  (reason: restore previous)
      2–5. [restore remaining]

  [Operator reviews table, approves all]
  [Executor writes all 5, reads back each]
```

### Multi-step procedure

```
Operator: "Try bumping COR:H01 to +1, check orbit, then to +2, check again"

Agent:
  get_parameter("COR:H01:SETPT")    → 0.0 mrad

  propose_plan:
    description: "Two-step corrector bump with orbit measurement"
    actions:
      1. READ   [BPM:01:X, BPM:02:X]  (label: baseline)
      2. WRITE  COR:H01:SETPT → 1.0
      3. WAIT   2.0s  (reason: settling)
      4. READ   [BPM:01:X, BPM:02:X]  (label: after +1 mrad)
      5. WRITE  COR:H01:SETPT → 2.0
      6. WAIT   2.0s
      7. READ   [BPM:01:X, BPM:02:X]  (label: after +2 mrad)
    rollback:
      1. WRITE  COR:H01:SETPT → 0.0  (reason: restore)
      2. WAIT   2.0s
      3. READ   [BPM:01:X, BPM:02:X]  (label: verify restore)
```

---

## 17. Implementation Roadmap

| Phase | Scope | Timeframe |
|-------|-------|-----------|
| **1 — Foundation** | ReAct loop, LLM abstraction (LiteLLM), tool registry, CLI shell | Week 1–2 |
| **2 — Read tools** | `get_parameter`, `search_device`, `read_log` with simulator adapter | Week 2–3 |
| **3 — Plans + Safety** | `propose_plan`, action plan model, approval gate, parameter validation, audit log | Week 3–4 |
| **4 — Code sandbox** | `execute_code`, `run_script`, `list_scripts`, static analysis, subprocess harness | Week 4–5 |
| **5 — Sub-agents** | Dispatcher, restricted tool sets, `spawn_subagent` | Week 5 |
| **6 — Context** | AGENT.md system, facility-specific context files | Week 5–6 |
| **7 — Multi-model** | Additional LiteLLM models in config, `/provider` switching | Week 6 |
| **8 — Real adapter** | Replace simulator with facility control system | Week 6–7 |
| **9 — OS sandbox** | Add bwrap or Docker isolation for code execution | Week 7 |
| **10 — Hardening** | Error handling, streaming responses, token budgets, integration tests | Week 7+ |
