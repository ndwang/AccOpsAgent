"""Read machine parameter / device value."""

from accel_agent.tools.base import BaseTool


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
