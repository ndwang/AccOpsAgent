"""List available analysis scripts with descriptions."""

import ast
from pathlib import Path

from accel_agent.tools.base import BaseTool


class ListScriptsTool(BaseTool):
    name = "list_scripts"
    description = "List available analysis scripts with descriptions."

    def __init__(self, config: dict):
        self.scripts_dir = Path(config.get("scripts_dir", "scripts"))

    def parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "filter": {"type": "string", "description": "Keyword filter"},
            },
        }

    async def execute(self, params):
        scripts = []
        if not self.scripts_dir.exists():
            return {"scripts": [], "count": 0}
        for path in sorted(self.scripts_dir.glob("*.py")):
            try:
                doc = ast.get_docstring(ast.parse(path.read_text()))
            except (SyntaxError, OSError):
                doc = None
            scripts.append({"name": path.name, "description": doc or "(no description)"})
        if params.get("filter"):
            kw = params["filter"].lower()
            scripts = [s for s in scripts if kw in s["name"].lower() or kw in (s["description"] or "").lower()]
        return {"scripts": scripts, "count": len(scripts)}
