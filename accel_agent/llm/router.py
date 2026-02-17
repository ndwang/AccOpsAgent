"""Model selection via LiteLLM (unified interface)."""

from accel_agent.llm.base import BaseLLM
from accel_agent.llm.litellm_client import LiteLLMClient


class LLMRouter:
    """
    Holds named LiteLLM-backed models for switching (e.g. /provider claude vs gpt4).
    Config: default model name, and [llm.models] name -> LiteLLM model string.
    """

    def __init__(self, config: dict):
        self._clients: dict[str, BaseLLM] = {}
        self.default = config.get("default", "default")
        models_cfg = config.get("models", {})
        max_tokens = config.get("max_tokens", 8096)

        # If no named models, support a single "default" model from top-level model key
        default_model = config.get("model", "anthropic/claude-sonnet-4-20250514")
        if not models_cfg and default_model:
            self._clients["default"] = LiteLLMClient(
                model=default_model, max_tokens=max_tokens
            )
            self.default = "default"

        for name, model in models_cfg.items():
            self._clients[name] = LiteLLMClient(
                model=model, max_tokens=max_tokens
            )

        if self._clients and self.default not in self._clients:
            self.default = next(iter(self._clients))

    def get(self, provider: str | None = None) -> BaseLLM:
        name = provider or self.default
        if name in self._clients:
            return self._clients[name]
        if self._clients:
            return self._clients[self.default]
        raise RuntimeError(
            "No LLM configured. Set [llm] model = \"provider/model\" or [llm.models] in config."
        )
