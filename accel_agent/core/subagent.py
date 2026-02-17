"""Sub-agent spawner and lifecycle (device_search, log_analysis, diagnostics)."""

from accel_agent.core.react_loop import ReActLoop
from accel_agent.core.conversation import Conversation

SUBAGENT_TOOL_SETS = {
    "device_search": ["search_device", "read_log"],
    "log_analysis": ["read_log", "execute_code"],
    "diagnostics": ["read_log", "get_parameter", "search_device"],
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
            + "\n\n"
            + SUBAGENT_PROMPTS.get(task_type, "")
        )
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
