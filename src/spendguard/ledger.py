"""Append-only usage ledger in SQLite: who spent what, through which agent, on which model."""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal

# USD per million tokens (input, output). Override per deployment.
DEFAULT_PRICES = {
    "claude-opus-5-5": (Decimal("4"), Decimal("20")),
    "claude-sonnet-5-5": (Decimal("2"), Decimal("10")),
    "claude-haiku-5-5": (Decimal("0.10"), Decimal("0.50")),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY,
    at TEXT NOT NULL,          -- UTC ISO timestamp
    day TEXT NOT NULL,         -- budget day, in the guard's time zone
    user TEXT NOT NULL,
    agent TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_user_day ON usage(user, day);
CREATE TABLE IF NOT EXISTS caps (
    user TEXT PRIMARY KEY,
    daily_tokens INTEGER NOT NULL,
    set_by TEXT NOT NULL,
    set_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Usage:
    user: str
    agent: str
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Ledger:
    def __init__(self, path: str = ":memory:", prices: dict | None = None):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self.prices = prices or DEFAULT_PRICES

    def cost(self, u: Usage) -> Decimal:
        p_in, p_out = self.prices.get(u.model, (Decimal(0), Decimal(0)))
        return (p_in * u.input_tokens + p_out * u.output_tokens) / Decimal(1_000_000)

    def record(self, u: Usage, day: date) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO usage(at, day, user, agent, model, input_tokens, output_tokens, cost_usd)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), day.isoformat(), u.user, u.agent, u.model,
                 u.input_tokens, u.output_tokens, str(self.cost(u))))

    def tokens_today(self, user: str, day: date) -> int:
        row = self._db.execute(
            "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) FROM usage WHERE user=? AND day=?",
            (user, day.isoformat())).fetchone()
        return int(row[0])

    # --- caps: anyone can read, only an admin can write ---------------------------------
    def cap_for(self, user: str) -> int | None:
        row = self._db.execute("SELECT daily_tokens FROM caps WHERE user=?", (user,)).fetchone()
        return row[0] if row else None

    def set_cap(self, user: str, daily_tokens: int, *, by: str, admins: set[str]) -> None:
        if by not in admins:
            raise PermissionError(f"{by} is not an administrator")
        if daily_tokens < 0:
            raise ValueError("cap must be >= 0")
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO caps(user, daily_tokens, set_by, set_at) VALUES (?,?,?,?)"
                " ON CONFLICT(user) DO UPDATE SET daily_tokens=excluded.daily_tokens,"
                " set_by=excluded.set_by, set_at=excluded.set_at",
                (user, daily_tokens, by, datetime.now(timezone.utc).isoformat()))

    def breakdown(self, by: str, since: date | None = None) -> list[tuple[str, int, Decimal]]:
        """Tokens and cost grouped by 'user', 'agent' or 'model', most expensive first."""
        if by not in {"user", "agent", "model"}:
            raise ValueError("by must be user, agent or model")
        params: tuple = (since.isoformat(),) if since else ()
        totals: dict[str, list] = {}
        # cost is summed in Python as Decimal so it never drifts like a float SUM would
        rows = self._db.execute(
            f"SELECT {by}, input_tokens + output_tokens, cost_usd FROM usage"
            + (" WHERE day >= ?" if since else ""), params).fetchall()
        for key, tokens, cost in rows:
            t = totals.setdefault(key, [0, Decimal(0)])
            t[0] += tokens
            t[1] += Decimal(cost)
        return sorted(((k, v[0], v[1]) for k, v in totals.items()), key=lambda r: -r[2])
