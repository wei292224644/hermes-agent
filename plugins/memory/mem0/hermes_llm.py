"""mem0 LLM adapter that delegates to hermes's central ``call_llm``.

mem0's local mode normally builds its own LLM client from a static
``{provider, api_key, base_url}`` dict via :class:`mem0.utils.factory.LlmFactory`.
That duplicates — and easily gets out of sync with — hermes's own provider
resolution, which stores credentials in the credential pool / auth store and
base URLs under ``custom_providers``, not in the ``model`` config block.

Instead, this adapter implements mem0's ``LLMBase`` interface and forwards
every generation to :func:`agent.auxiliary_client.call_llm`, the single entry
point all other hermes subsystems (title generation, vision, session search…)
use. That call resolves provider, model, credentials, base URL, and the wire
protocol (chat-completions / codex-responses / anthropic-messages) on its own,
so mem0 reuses the user's configured main model with zero extra config.

Wire-up: :func:`register_hermes_llm` registers this class under the provider
name ``"hermes"`` in mem0's ``LlmFactory``. The plugin sets the local-mode
LLM provider to ``"hermes"`` whenever ``llm_provider == "hermes"``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from mem0.llms.base import LLMBase

logger = logging.getLogger(__name__)

_FACTORY_PROVIDER_NAME = "hermes"


class HermesLLM(LLMBase):
    """mem0 LLM provider backed by hermes's ``call_llm``."""

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Any = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """Generate a response by delegating to hermes ``call_llm``.

        Returns a plain string when no tools are requested (mem0 parses the
        JSON content itself), or a ``{"content", "tool_calls"}`` dict when
        tools are passed — mirroring mem0's built-in provider contract.

        ``response_format`` (mem0 asks for ``{"type": "json_object"}`` during
        fact extraction) is forwarded via ``extra_body`` so OpenAI-compatible
        backends get JSON mode. Backends that reject it (e.g. Anthropic-style
        endpoints) are handled by retrying once without it — the extraction
        prompt already instructs the model to emit JSON.
        """
        try:
            return self._invoke(messages, response_format, tools, tool_choice)
        except Exception as exc:
            if response_format is not None:
                logger.debug(
                    "HermesLLM: retrying without response_format after error: %s", exc
                )
                return self._invoke(messages, None, tools, tool_choice)
            raise

    def _invoke(self, messages, response_format, tools, tool_choice):
        # Imported lazily: the agent stack is heavy and only needed at call time.
        from agent.auxiliary_client import call_llm

        extra_body: Dict[str, Any] = {}
        if response_format is not None:
            extra_body["response_format"] = response_format

        # No task/provider/model: let hermes auto-resolve to the user's main
        # configured model and credentials.
        response = call_llm(
            messages=messages,
            tools=tools or None,
            temperature=getattr(self.config, "temperature", None),
            max_tokens=getattr(self.config, "max_tokens", None),
            extra_body=extra_body or None,
        )
        return self._parse_response(response, tools)

    @staticmethod
    def _parse_response(response, tools):
        message = response.choices[0].message
        if not tools:
            return message.content

        parsed: Dict[str, Any] = {"content": message.content, "tool_calls": []}
        for tool_call in (getattr(message, "tool_calls", None) or []):
            parsed["tool_calls"].append(
                {
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments),
                }
            )
        return parsed


def register_hermes_llm() -> None:
    """Register :class:`HermesLLM` under the ``"hermes"`` provider name.

    Idempotent — safe to call before every ``Memory`` construction. Note that
    mem0's pydantic ``LlmConfig`` validator rejects unknown provider names, so
    the plugin builds the ``MemoryConfig`` with the llm field constructed via
    ``model_construct`` to bypass that check (see ``_get_client``).
    """
    from mem0.utils.factory import LlmFactory
    from mem0.configs.llms.base import BaseLlmConfig

    LlmFactory.provider_to_class[_FACTORY_PROVIDER_NAME] = (
        "plugins.memory.mem0.hermes_llm.HermesLLM",
        BaseLlmConfig,
    )
