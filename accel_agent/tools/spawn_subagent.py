"""Dispatch sub-agent for contained tasks (device search, log analysis, diagnostics)."""

from accel_agent.tools.base import BaseTool


class SpawnSubagentTool(BaseTool):
    name = "spawn_subagent"
    description = """Dispatch a sub-agent for a contained subtask. Sub-agents run
with a restricted read-only tool set and return structured results.

Task types:
- "device_search": Find device names by pattern across large namespaces
- "log_analysis": Read and analyze log entries, extract patterns
- "diagnostics": Cross-reference parameters and logs to diagnose issues

Use this when a task requires many intermediate lookups that would
clutter the main conversation (e.g., finding all BPMs in a sector)."""

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
                "instruction": {"type": "string", "description": "Detailed instruction for the sub-agent"},
            },
            "required": ["task_type", "instruction"],
        }

    async def execute(self, params):
        return await self.dispatcher.run(
            task_type=params["task_type"],
            instruction=params["instruction"],
        )
