"""Tool registration and schema generation."""

from accel_agent.tools.base import BaseTool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        return self._tools[name]

    def schemas(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]

    def subset(self, names: list[str]) -> "ToolRegistry":
        sub = ToolRegistry()
        for name in names:
            if name in self._tools:
                sub.register(self._tools[name])
        return sub
