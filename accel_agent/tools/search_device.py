"""Search device names by pattern."""

from accel_agent.tools.base import BaseTool


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
