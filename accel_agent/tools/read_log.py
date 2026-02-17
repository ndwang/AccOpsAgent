"""Read e-log / alarm log / archive."""

from datetime import datetime

from accel_agent.tools.base import BaseTool


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
        entries = await self.adapter.read_log(
            source=params["source"],
            time_range=(
                datetime.fromisoformat(params["start_time"]),
                datetime.fromisoformat(params["end_time"]),
            ),
            filters=params.get("filters"),
        )
        return {"entries": entries, "count": len(entries)}
