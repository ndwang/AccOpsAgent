"""Abstract LLM interface and response types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall] | None
    usage: dict | None = None
    raw: object = None


class BaseLLM(ABC):
    """Abstract interface that each provider implements."""

    @abstractmethod
    async def complete(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        ...
