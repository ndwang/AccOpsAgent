"""Open plot / control windows (facility-specific)."""

from accel_agent.tools.base import BaseTool


class LaunchDisplayTool(BaseTool):
    name = "launch_display"
    description = (
        "Launch a plot or control display window. "
        "Facility-specific implementation (e.g., Open XAL, Control System Studio)."
    )

    def parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "display_name": {"type": "string", "description": "Display to open"},
                "args": {"type": "object", "description": "Optional display arguments"},
            },
            "required": ["display_name"],
        }

    async def execute(self, params):
        # Placeholder: facility implements opening display
        return {"status": "not_implemented", "display": params.get("display_name")}
