"""Run existing analysis scripts in sandbox."""

from pathlib import Path

from accel_agent.tools.base import BaseTool
from accel_agent.tools.execute_code import ExecuteCodeTool


class RunExistingScriptTool(BaseTool):
    name = "run_script"
    description = """Run an existing analysis script from the scripts directory.
Scripts run in the same sandbox as execute_code. Pass arguments via 'args' as sys.argv."""

    def __init__(self, config: dict):
        self.scripts_dir = Path(config.get("scripts_dir", "scripts"))
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
            available = [f.name for f in self.scripts_dir.glob("*.py")] if self.scripts_dir.exists() else []
            return {
                "status": "error",
                "message": f"Script '{params['script_name']}' not found.",
                "available_scripts": available,
            }
        code = path.read_text()
        if params.get("args"):
            code = f"import sys; sys.argv = {repr([params['script_name']] + params['args'])}\n" + code
        return await self.executor.execute({"code": code, "description": params["description"]})
