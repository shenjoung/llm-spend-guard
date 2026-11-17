from types import SimpleNamespace

import pytest

anthropic = pytest.importorskip("anthropic")   # optional dependency

from spendguard.anthropic_provider import AnthropicProvider  # noqa: E402


def test_maps_sdk_reply_to_response():
    msg = SimpleNamespace(stop_reason="end_turn", stop_details=None,
                          content=[SimpleNamespace(type="text", text="ok")],
                          usage=SimpleNamespace(input_tokens=12, output_tokens=3))
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: msg))
    r = AnthropicProvider(client=client)({"max_tokens": 10, "messages": []})
    assert (r.text, r.model, r.input_tokens, r.output_tokens) == ("ok", "claude-opus-5-5", 12, 3)
