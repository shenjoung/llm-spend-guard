# llm-spend-guard

Once a team has more than a handful of LLM-powered tools, three things go wrong:

1. **A runaway loop burns the budget.** One agent retries forever and spends in an
   afternoon what the team was meant to spend in a week — usually through a code
   path nobody remembered to put a limit on.
2. **The provider has a bad hour.** Every request fails, every tool hammers the same
   failing endpoint, and nothing falls back.
3. **The same thing is sent twice.** A retry or a double-fired webhook posts the same
   message or runs the same job again.

`spendguard` puts every LLM call behind one entry point that handles all three.

```python
from spendguard import Guard, Ledger, ProviderChain
from spendguard.anthropic_provider import AnthropicProvider

guard = Guard(
    ledger=Ledger("usage.db"),
    call_model=ProviderChain([AnthropicProvider("claude-opus-5-5"),
                              AnthropicProvider("claude-sonnet-5-5")], cooldown_s=900),
    default_cap=2_000_000,          # tokens per person per day
    tz="Asia/Taipei",               # when "today" resets
)

result = guard.call({"max_tokens": 1024, "messages": [{"role": "user", "content": "..."}]},
                    user="ann", agent="mail-triage", idempotency_key="email:8812")
```

## What each call goes through

| Step | Behaviour |
|---|---|
| **Dedupe** | Same `idempotency_key` (or identical request) within 10 minutes returns the first result instead of calling again. |
| **Budget, before dispatch** | Estimated tokens are checked against what the person has left today. Work that would cross the cap is **held** (`HeldJob`) — nothing is sent, nothing is billed. `hold=False` raises `OverBudget` instead. |
| **Failover** | Retryable errors (429, 5xx, network) are retried with backoff, then the next provider is used. A failing provider is skipped for a cooldown (default 15 min) instead of being hit on every request; non-retryable errors (a bad request) are raised immediately, because they would fail on the backup too. |
| **Ledger** | Actual input/output tokens and exact `Decimal` cost are recorded per person, agent and model in SQLite. |

```python
guard.ledger.breakdown("agent")          # [('mail-triage', 1_240_000, Decimal('7.32')), ...]
guard.ledger.set_cap("ann", 5_000_000, by="lee", admins={"lee"})   # only admins can change caps
guard.release_held()                     # retry held work after a cap change or a new day
```

## Design notes

- **One entry point.** A cap that is enforced on one code path and not another is not
  a cap; the test `test_runaway_job_cannot_exceed_cap_on_any_path` replays the classic
  incident (a loop calling until 53.6M tokens against a 30M daily cap) and checks the
  ledger never passes the cap.
- **Hold, don't drop.** Held jobs keep their request, so nothing a user asked for is
  silently lost.
- **Fail fast inside, retry outside.** The Anthropic adapter sets the SDK's own
  `max_retries=0` so retries and failover happen in one place with one policy.
- **Injectable clock and sleep**, so cooldowns and day boundaries are tested without waiting.

## Install and test

```bash
pip install -e ".[anthropic,dev]"
python -m pytest
```

11 tests cover the budget (hold before dispatch, the runaway replay, per-user caps,
admin-only changes, time-zone day boundaries, releasing held work), failover (cooldown,
recovery, non-retryable errors, all providers down), dedupe and the ledger. The
Anthropic adapter test runs only when the `anthropic` package is installed.

## License

MIT
