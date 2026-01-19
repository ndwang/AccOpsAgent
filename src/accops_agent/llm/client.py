"""LLM client wrapper for OpenAI-compatible APIs."""

import logging
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..utils.constants import (
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with OpenAI-compatible LLM APIs.

    This client wraps the OpenAI SDK to provide a consistent interface
    for LLM operations. It supports OpenAI API and any compatible endpoints
    (e.g., local models via llama.cpp, vLLM, etc.).

    Attributes:
        client: OpenAI client instance
        model: Model name to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    ):
        """Initialize LLM client.

        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for API (defaults to OpenAI, or OPENAI_BASE_URL env var)
            model: Model name to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key from parameter or environment
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "No API key provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Get base URL from parameter or environment
        base_url = base_url or os.getenv("OPENAI_BASE_URL")

        # Initialize OpenAI client
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"Initialized LLM client with custom base URL: {base_url}")
        else:
            self.client = OpenAI(api_key=api_key)
            logger.info("Initialized LLM client with OpenAI API")

        logger.info(f"Using model: {model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text completion from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Generated text completion

        Raises:
            RuntimeError: If generation fails
        """
        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )

            generated_text = response.choices[0].message.content
            logger.debug(f"Generated {len(generated_text)} characters")

            return generated_text

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise RuntimeError(f"LLM generation failed: {e}") from e

    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion with message history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Generated text completion

        Raises:
            RuntimeError: If generation fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )

            generated_text = response.choices[0].message.content
            logger.debug(f"Generated {len(generated_text)} characters")

            return generated_text

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise RuntimeError(f"LLM generation failed: {e}") from e

    def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate structured output (JSON mode).

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            response_format: Response format specification

        Returns:
            Generated structured text

        Raises:
            RuntimeError: If generation fails
        """
        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            # Add response_format if provided (requires compatible models)
            if response_format:
                kwargs["response_format"] = response_format

            response = self.client.chat.completions.create(**kwargs)

            generated_text = response.choices[0].message.content
            logger.debug(f"Generated structured output: {len(generated_text)} characters")

            return generated_text

        except Exception as e:
            logger.error(f"Structured generation failed: {e}")
            raise RuntimeError(f"Structured generation failed: {e}") from e
