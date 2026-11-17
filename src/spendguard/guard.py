"""The one entry point every LLM call goes through: budget check, dedupe, failover, ledger.

A cap that is enforced on one code path but not another is not a cap. Routing every call
through Guard.call() is the point of this module.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable
from zoneinfo import ZoneInfo

from .failover import Response
from .ledger import Ledger, Usage


class OverBudget(Exception):
    def __init__(self, user: str, spent: int, cap: int, estimate: int):
        self.user, self.spent, self.cap, self.estimate = user, spent, cap, estimate
        super().__init__(f"{user}: {spent:,} used + ~{estimate:,} estimated > daily cap {cap:,}")


@dataclass
class HeldJob:
    """Work refused before dispatch because it would cross the cap. Nothing was sent or billed."""
    user: str
    agent: str
    request: dict
    reason: str
    held_at: float = field(default_factory=time.time)


def estimate_tokens(request: dict) -> int:
    """Cheap upper-ish estimate: ~4 characters per token for input, plus the output cap."""
    text = json.dumps(request.get("messages", []), ensure_ascii=False)
    return len(text) // 4 + int(request.get("max_tokens", 1024))


@dataclass
class Guard:
    ledger: Ledger
    call_model: Callable[[dict], Response]         # a ProviderChain or any provider
    default_cap: int | None = 2_000_000            # tokens per user per day; None = unlimited
    tz: str = "UTC"
    dedupe_ttl_s: float = 600
    estimator: Callable[[dict], int] = estimate_tokens
    clock: Callable[[], float] = time.time
    held: list[HeldJob] = field(default_factory=list)
    _recent: dict[str, tuple[float, Response]] = field(default_factory=dict)

    def today(self) -> date:
        return datetime.fromtimestamp(self.clock(), ZoneInfo(self.tz)).date()

    def remaining(self, user: str) -> int | None:
        cap = self.ledger.cap_for(user)
        cap = self.default_cap if cap is None else cap
        if cap is None:
            return None
        return max(cap - self.ledger.tokens_today(user, self.today()), 0)

    def call(self, request: dict, *, user: str, agent: str = "default",
             idempotency_key: str | None = None, hold: bool = True) -> Response | HeldJob:
        # 1. Same notification / same job twice within the TTL: return the first result.
        key = idempotency_key or _fingerprint(user, agent, request)
        hit = self._recent.get(key)
        if hit and self.clock() - hit[0] < self.dedupe_ttl_s:
            return hit[1]

        # 2. Budget is checked BEFORE dispatch, so an over-cap job costs nothing.
        left = self.remaining(user)
        if left is not None:
            est = self.estimator(request)
            if est > left:
                spent = self.ledger.tokens_today(user, self.today())
                cap = spent + left
                if not hold:
                    raise OverBudget(user, spent, cap, est)
                job = HeldJob(user, agent, request, str(OverBudget(user, spent, cap, est)))
                self.held.append(job)
                return job

        # 3. Call through failover, 4. record what was actually used.
        resp = self.call_model(request)
        self.ledger.record(Usage(user, agent, resp.model, resp.input_tokens, resp.output_tokens),
                           self.today())
        self._recent[key] = (self.clock(), resp)
        return resp

    def release_held(self) -> list[Response | HeldJob]:
        """Retry held jobs (e.g. after midnight or a cap raise); still-blocked ones stay held."""
        jobs, self.held = self.held, []
        return [self.call(j.request, user=j.user, agent=j.agent) for j in jobs]


def _fingerprint(user: str, agent: str, request: dict) -> str:
    blob = json.dumps({"u": user, "a": agent, "r": request}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()
