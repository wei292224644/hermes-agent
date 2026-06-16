"""Tests for the HermesLLM mem0 adapter.

The adapter lets mem0's local-mode fact extraction reuse hermes's central
``call_llm`` (provider routing, credential pool, OAuth, anthropic/codex
wrapping) instead of building a parallel client from a static config dict.
"""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock


def _fake_response(content, tool_calls=None):
    """Build a minimal OpenAI-style chat completion response object."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def test_generate_response_delegates_to_call_llm_and_returns_content():
    from plugins.memory.mem0.hermes_llm import HermesLLM

    llm = HermesLLM(config={})
    captured = {}

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return _fake_response("extracted facts")

    with patch("agent.auxiliary_client.call_llm", side_effect=fake_call_llm):
        out = llm.generate_response(
            messages=[{"role": "user", "content": "hi"}],
        )

    assert out == "extracted facts"
    # Delegated the messages verbatim; let hermes auto-resolve provider/model.
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert captured.get("provider") in (None,)
    assert captured.get("model") in (None,)


def test_generate_response_forwards_response_format_via_extra_body():
    from plugins.memory.mem0.hermes_llm import HermesLLM

    llm = HermesLLM(config={})
    captured = {}

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return _fake_response("{}")

    with patch("agent.auxiliary_client.call_llm", side_effect=fake_call_llm):
        llm.generate_response(
            messages=[{"role": "user", "content": "x"}],
            response_format={"type": "json_object"},
        )

    assert captured["extra_body"]["response_format"] == {"type": "json_object"}


def test_generate_response_retries_without_response_format_on_failure():
    """A backend that rejects response_format (e.g. Anthropic-style endpoints)
    must not break extraction — retry once without the JSON-mode hint."""
    from plugins.memory.mem0.hermes_llm import HermesLLM

    llm = HermesLLM(config={})
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ValueError("Unsupported parameter: response_format")
        return _fake_response("recovered")

    with patch("agent.auxiliary_client.call_llm", side_effect=fake_call_llm):
        out = llm.generate_response(
            messages=[{"role": "user", "content": "x"}],
            response_format={"type": "json_object"},
        )

    assert out == "recovered"
    assert len(calls) == 2
    # First attempt carried response_format; retry dropped it.
    assert calls[0]["extra_body"]["response_format"] == {"type": "json_object"}
    assert not calls[1].get("extra_body")


def test_generate_response_with_tools_returns_content_and_tool_calls():
    from plugins.memory.mem0.hermes_llm import HermesLLM

    llm = HermesLLM(config={})
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="add_memory", arguments='{"text": "likes tea"}')
    )

    with patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_response("done", tool_calls=[tool_call]),
    ):
        out = llm.generate_response(
            messages=[{"role": "user", "content": "x"}],
            tools=[{"type": "function", "function": {"name": "add_memory"}}],
        )

    assert out["content"] == "done"
    assert out["tool_calls"] == [{"name": "add_memory", "arguments": {"text": "likes tea"}}]


def test_register_hermes_llm_adds_provider_to_factory():
    from plugins.memory.mem0.hermes_llm import register_hermes_llm
    from mem0.utils.factory import LlmFactory

    register_hermes_llm()
    assert "hermes" in LlmFactory.provider_to_class
    class_path, _config_cls = LlmFactory.provider_to_class["hermes"]
    assert class_path.endswith("HermesLLM")


def test_factory_creates_hermes_llm_instance():
    from plugins.memory.mem0.hermes_llm import register_hermes_llm, HermesLLM
    from mem0.utils.factory import LlmFactory

    register_hermes_llm()
    inst = LlmFactory.create("hermes", {})
    assert isinstance(inst, HermesLLM)
