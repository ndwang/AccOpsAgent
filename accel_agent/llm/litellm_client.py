"""Unified LLM interface via LiteLLM (all providers through one API)."""

import json
from litellm import acompletion

from accel_agent.llm.base import BaseLLM, LLMResponse, ToolCall


class LiteLLMClient(BaseLLM):
    """
    Single client that talks to any provider via LiteLLM.
    Model names use LiteLLM format: anthropic/claude-sonnet-4-20250514, openai/gpt-4o, etc.
    API keys from environment: ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.
    """

    def __init__(self, model: str, max_tokens: int = 8096):
        self.model = model
        self.max_tokens = max_tokens

    async def complete(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        # LiteLLM expects OpenAI-style messages (system can be first message)
        full_messages = [{"role": "system", "content": system}] + list(messages)

        kwargs = {
            "model": self.model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    },
                }
                for t in tools
            ]
            kwargs["tool_choice"] = "auto"

        response = await acompletion(**kwargs)
        return self._parse(response)

    def _parse(self, response) -> LLMResponse:
        # LiteLLM normalizes to OpenAI ChatCompletion shape
        msg = response.choices[0].message
        tool_calls = None
        if getattr(msg, "tool_calls", None):
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    input=json.loads(tc.function.arguments or "{}"),
                )
                for tc in msg.tool_calls
            ]
        usage = None
        if getattr(response, "usage", None):
            usage = {
                "input": getattr(response.usage, "prompt_tokens", 0),
                "output": getattr(response.usage, "completion_tokens", 0),
            }
        return LLMResponse(
            text=getattr(msg, "content", None) or None,
            tool_calls=tool_calls,
            usage=usage,
            raw=response,
        )
