from datetime import date
from decimal import Decimal

import pytest

from spendguard import (AllProvidersFailed, Guard, HeldJob, Ledger, OverBudget, ProviderChain,
                        Response, Retryable, Usage)


class FakeClock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t
    def __call__(self):
        return self.t
    def advance(self, s):
        self.t += s


class FakeProvider:
    def __init__(self, name, fail_times=0, error=Retryable("503"), out=100):
        self.name, self.fail_times, self.error, self.out, self.calls = name, fail_times, error, out, 0
    def __call__(self, request):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        return Response(text=f"from {self.name}", model=self.name, input_tokens=50, output_tokens=self.out)


def chain(*providers, clock=None, **kw):
    return ProviderChain(list(providers), clock=clock or FakeClock(), sleep=lambda s: None, **kw)


REQ = {"max_tokens": 200, "messages": [{"role": "user", "content": "hi"}]}


# --- budget ---------------------------------------------------------------------------
def test_over_cap_work_is_held_before_dispatch_and_costs_nothing():
    led, p = Ledger(), FakeProvider("claude-opus-5-5")
    g = Guard(led, chain(p), default_cap=1_000, clock=FakeClock())
    led.record(Usage("ann", "a", "claude-opus-5-5", 600, 300), g.today())   # 900 used
    out = g.call(REQ, user="ann")
    assert isinstance(out, HeldJob) and p.calls == 0
    assert "daily cap 1,000" in out.reason
    with pytest.raises(OverBudget):
        g.call(REQ, user="ann", hold=False)


def test_runaway_job_cannot_exceed_cap_on_any_path():
    """The incident this exists for: a loop that kept calling until 53.6M tokens against 30M."""
    p = FakeProvider("claude-opus-5-5", out=1_000_000)
    g = Guard(Ledger(), chain(p), default_cap=30_000_000, clock=FakeClock(), dedupe_ttl_s=0,
              estimator=lambda r: 1_000_050)
    results = [g.call({**REQ, "n": i}, user="bot") for i in range(60)]
    assert sum(isinstance(r, Response) for r in results) == 29
    assert g.ledger.tokens_today("bot", g.today()) <= 30_000_000


def test_per_user_cap_overrides_default_and_only_admins_set_it():
    led = Ledger()
    with pytest.raises(PermissionError):
        led.set_cap("ann", 10, by="ann", admins={"lee"})
    led.set_cap("ann", 5_000_000, by="lee", admins={"lee"})
    g = Guard(led, chain(FakeProvider("m")), default_cap=100, clock=FakeClock())
    assert g.remaining("ann") == 5_000_000 and g.remaining("bob") == 100


def test_budget_day_follows_the_configured_time_zone():
    clock = FakeClock(1_800_000_000.0)          # 2027-01-15 08:00 UTC = 16:00 in Taipei
    g = Guard(Ledger(), chain(FakeProvider("m")), clock=clock, tz="Asia/Taipei")
    assert g.today() == date(2027, 1, 15)
    clock.advance(9 * 3600)                     # 01:00 next day in Taipei, still the 15th in UTC
    assert g.today() == date(2027, 1, 16)


def test_held_jobs_run_after_cap_is_raised():
    led = Ledger()
    g = Guard(led, chain(FakeProvider("m")), default_cap=10, clock=FakeClock())
    assert isinstance(g.call(REQ, user="ann"), HeldJob)
    led.set_cap("ann", 10_000, by="lee", admins={"lee"})
    (out,) = g.release_held()
    assert isinstance(out, Response) and not g.held


# --- failover -------------------------------------------------------------------------
def test_failover_uses_backup_and_cools_down_primary():
    clock = FakeClock()
    main, backup = FakeProvider("main", fail_times=99), FakeProvider("backup")
    c = chain(main, backup, clock=clock, retries_per_provider=1, cooldown_s=900)
    assert c(REQ).model == "backup" and main.calls == 2              # 1 try + 1 retry
    assert c(REQ).model == "backup" and main.calls == 2              # skipped while cooling down
    clock.advance(901)
    main.fail_times = 0
    assert c(REQ).model == "main" and "recovered main" in c.events


def test_non_retryable_errors_are_not_masked_by_failover():
    main, backup = FakeProvider("main", fail_times=1, error=ValueError("bad request")), FakeProvider("b")
    with pytest.raises(ValueError):
        chain(main, backup)(REQ)
    assert backup.calls == 0


def test_all_down_raises_with_every_error():
    with pytest.raises(AllProvidersFailed) as e:
        chain(FakeProvider("a", fail_times=9), FakeProvider("b", fail_times=9))(REQ)
    assert set(e.value.errors) == {"a", "b"}


# --- dedupe and ledger ----------------------------------------------------------------
def test_duplicate_request_within_ttl_is_not_sent_twice():
    clock, p = FakeClock(), FakeProvider("m")
    g = Guard(Ledger(), chain(p), clock=clock, dedupe_ttl_s=600)
    g.call(REQ, user="ann", idempotency_key="notify:order-42")
    g.call(REQ, user="ann", idempotency_key="notify:order-42")
    assert p.calls == 1
    clock.advance(601)
    g.call(REQ, user="ann", idempotency_key="notify:order-42")
    assert p.calls == 2


def test_breakdown_by_model_uses_exact_decimal_cost():
    led = Ledger()
    d = date(2026, 11, 18)
    led.record(Usage("ann", "mail", "claude-opus-5-5", 1_000_000, 100_000), d)       # 4 + 2
    led.record(Usage("bob", "mail", "claude-haiku-5-5", 1_000_000, 1_000_000), d)   # 0.1 + 0.5
    rows = led.breakdown("model")
    assert rows[0] == ("claude-opus-5-5", 1_100_000, Decimal("6"))
    assert rows[1][2] == Decimal("0.6")
    assert led.breakdown("agent")[0][0] == "mail"
    with pytest.raises(ValueError):
        led.breakdown("user; DROP TABLE usage")
