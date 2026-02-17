"""Base tool interface (BaseTool ABC)."""

from abc import ABC, abstractmethod


class BaseTool(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def parameters_schema(self) -> dict:
        """JSON Schema for tool input."""
        ...

    @abstractmethod
    async def execute(self, params: dict):
        """Run the tool. Returns a result for the LLM."""
        ...

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema(),
        }
