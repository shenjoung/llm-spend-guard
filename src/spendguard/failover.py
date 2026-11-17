"""Provider failover: when the main model is failing, use the backup and leave the main one
alone for a cooldown period instead of hammering it on every request."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass
class Response:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class Provider(Protocol):
    name: str
    def __call__(self, request: dict) -> Response: ...


class Retryable(Exception):
    """Raise (or map to) this for errors worth trying elsewhere: 429, 5xx, timeouts."""


class AllProvidersFailed(Exception):
    def __init__(self, errors: dict[str, Exception]):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


@dataclass
class ProviderChain:
    providers: list[Provider]
    cooldown_s: float = 15 * 60
    retries_per_provider: int = 2
    backoff_s: float = 0.5
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    is_retryable: Callable[[Exception], bool] = lambda e: isinstance(e, Retryable)
    _down_until: dict[str, float] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)   # what happened, for logs and tests

    def available(self) -> list[Provider]:
        now = self.clock()
        up = [p for p in self.providers if self._down_until.get(p.name, 0) <= now]
        # If everything is cooling down, still try the one that recovers first.
        return up or sorted(self.providers, key=lambda p: self._down_until.get(p.name, 0))[:1]

    def __call__(self, request: dict) -> Response:
        errors: dict[str, Exception] = {}
        for provider in self.available():
            for attempt in range(self.retries_per_provider + 1):
                try:
                    resp = provider(request)
                    if self._down_until.pop(provider.name, None) is not None:
                        self.events.append(f"recovered {provider.name}")
                    return resp
                except Exception as e:  # noqa: BLE001 - classified below
                    if not self.is_retryable(e):
                        raise                       # a 400 will fail on the backup too
                    errors[provider.name] = e
                    if attempt < self.retries_per_provider:
                        self.sleep(self.backoff_s * 2 ** attempt)
            self._down_until[provider.name] = self.clock() + self.cooldown_s
            self.events.append(f"cooldown {provider.name} {self.cooldown_s:.0f}s")
        raise AllProvidersFailed(errors)
