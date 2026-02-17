"""Top-level orchestrator wiring: LLM, adapter, tools, ReAct loop."""

import os
from pathlib import Path

from accel_agent.llm.router import LLMRouter
from accel_agent.core.context import ContextManager
from accel_agent.core.react_loop import ReActLoop
from accel_agent.core.conversation import Conversation
from accel_agent.core.plan_executor import PlanExecutor
from accel_agent.core.subagent import SubAgentDispatcher
from accel_agent.safety.approval import ApprovalGate
from accel_agent.safety.validators import ParameterValidator
from accel_agent.safety.audit import AuditLog
from accel_agent.tools.registry import ToolRegistry
from accel_agent.tools.get_parameter import GetParameterTool
from accel_agent.tools.search_device import SearchDeviceTool
from accel_agent.tools.read_log import ReadLogTool
from accel_agent.tools.launch_display import LaunchDisplayTool
from accel_agent.tools.propose_plan import ProposePlanTool
from accel_agent.tools.execute_code import ExecuteCodeTool
from accel_agent.tools.run_script import RunExistingScriptTool
from accel_agent.tools.list_scripts import ListScriptsTool
from accel_agent.tools.spawn_subagent import SpawnSubagentTool


class OrchestratorAgent:
    """Wires all components together."""

    def __init__(self, config: dict):
        self.config = config
        self.llm_router = LLMRouter(config.get("llm", {}))
        self.llm = self.llm_router.get()

        self.adapter = self._make_adapter(config.get("adapter", {}))

        agent_md = config.get("agent_md", "AGENT.md")
        self.context = ContextManager(agent_md_path=agent_md)

        self.approval_gate = ApprovalGate()
        limits_path = config.get("safety_limits", "configs/safety_limits.toml")
        self.validator = ParameterValidator(limits_path)
        self.audit = AuditLog()
        self.operator = self._resolve_operator(config)

        self.plan_executor = PlanExecutor(self.adapter, self.validator)
        self.tools = ToolRegistry()
        self._register_tools(config)

        self.dispatcher = SubAgentDispatcher(
            self.llm, self.tools, self.context
        )
        self.tools.register(SpawnSubagentTool(self.dispatcher))

        self.react_loop = ReActLoop(
            llm=self.llm,
            tool_registry=self.tools,
            context_manager=self.context,
        )
        self.conversation = Conversation()

    def _make_adapter(self, adapter_config: dict):
        kind = adapter_config.get("type", "simulator")
        if kind == "simulator":
            from accel_agent.adapters.simulator import SimulatorAdapter
            return SimulatorAdapter()
        if kind == "epics":
            from accel_agent.adapters.epics import EpicsAdapter
            return EpicsAdapter()
        if kind == "tango":
            from accel_agent.adapters.tango import TangoAdapter
            return TangoAdapter()
        from accel_agent.adapters.simulator import SimulatorAdapter
        return SimulatorAdapter()

    def _register_tools(self, config: dict) -> None:
        self.tools.register(GetParameterTool(self.adapter))
        self.tools.register(SearchDeviceTool(self.adapter))
        self.tools.register(ReadLogTool(self.adapter))
        self.tools.register(LaunchDisplayTool())
        self.tools.register(
            ProposePlanTool(
                self.approval_gate, self.plan_executor,
                audit_log=self.audit, operator=self.operator,
            )
        )
        self.tools.register(ExecuteCodeTool(config.get("sandbox", {})))
        self.tools.register(RunExistingScriptTool(config.get("sandbox", {})))
        self.tools.register(ListScriptsTool(config.get("sandbox", {})))

    async def run(self, user_message: str) -> str:
        return await self.react_loop.run(user_message, self.conversation)

    def switch_provider(self, name: str) -> None:
        self.llm = self.llm_router.get(name)
        self.react_loop.llm = self.llm
        self.dispatcher.llm = self.llm

    @staticmethod
    def _resolve_operator(config: dict) -> str:
        return (
            config.get("operator")
            or os.environ.get("ACCEL_AGENT_OPERATOR")
            or os.environ.get("USER")
            or os.environ.get("USERNAME")
            or "unknown"
        )

    def clear_conversation(self) -> None:
        self.conversation = Conversation()
