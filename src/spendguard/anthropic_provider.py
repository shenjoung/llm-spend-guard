"""Adapter that turns the Anthropic SDK into a spendguard Provider."""
from __future__ import annotations

from dataclasses import dataclass, field

from .failover import Response, Retryable


@dataclass
class AnthropicProvider:
    model: str = "claude-opus-5-5"
    client: object = None
    name: str = field(default="")

    def __post_init__(self):
        self.name = self.name or self.model
        if self.client is None:
            import anthropic
            # The chain does its own retries and failover, so the SDK should fail fast.
            self.client = anthropic.Anthropic(max_retries=0)

    def __call__(self, request: dict) -> Response:
        import anthropic
        try:
            msg = self.client.messages.create(model=self.model, **request)
        except (anthropic.RateLimitError, anthropic.InternalServerError,
                anthropic.APIConnectionError) as e:
            raise Retryable(f"{type(e).__name__}: {e}") from e
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"{self.model} declined the request: {msg.stop_details}")
        text = "".join(b.text for b in msg.content if b.type == "text")
        return Response(text=text, model=self.model,
                        input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens)
