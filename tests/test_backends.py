"""Contract tests for the two pluggable backends.

Both `LLMEngine` and `HindsightStore` are the parts of this system that cannot
run without credentials: one needs an API key, the other needs a Hindsight
instance. Left untested they are the two places a bug hides until the morning
of the demo.

So both are tested at their seam instead of against the real thing. The model
client and the Hindsight client are injected, and the tests assert on the calls
that would be made: method, arguments, metadata, and how the response is
parsed back. No key, no network, no server.

The Hindsight assertions are pinned to the real `hindsight-client` surface
(`retain`, `recall`, `reflect`, `list_memories`, `list_banks`,
`clear_memories`) as documented at https://hindsight.vectorize.io. An earlier
version of this file asserted a guessed REST contract -- `POST /memories`,
`POST /recall` -- that does not exist; the guess was replaced after reading the
docs, which is the only reason these tests are worth having.

What these do NOT prove: that Anthropic or Hindsight accepts this. A real
contract test needs a real service. `tests/test_agent.py` covers the behaviour
that sits on top of both.
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest

from memory_agent import SupportAgent
from memory_agent.engine import LLMEngine, ScriptedEngine, build_engine
from memory_agent.models import Memory, MemoryKind, Outcome, RetrievedMemory
from memory_agent.store import (HindsightStore, InMemoryStore, ResilientStore,
                                build_store)


# --------------------------------------------------------------- LLM engine


class FakeMessages:
    """Stands in for `client.messages`. Records every call."""

    def __init__(self, reply: str = "Re-applying the update, which fixed it last time.") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class Message:
            pass

        message = Message()
        message.content = [type("Block", (), {"text": self.reply})()]
        return message


class FakeClient:
    def __init__(self, reply: str = "Re-applying the update.") -> None:
        self.messages = FakeMessages(reply)


def _recalled() -> list[RetrievedMemory]:
    return [
        RetrievedMemory(
            memory=Memory(customer_id="CUST-1", kind=MemoryKind.ATTEMPT,
                          text="Update application", outcome=Outcome.WORKED),
            score=0.67,
            reason="same product, matched on: app",
        ),
        RetrievedMemory(
            memory=Memory(customer_id="CUST-1", kind=MemoryKind.ATTEMPT,
                          text="Restart application", outcome=Outcome.FAILED),
            score=0.41,
            reason="same product, matched on: app",
        ),
    ]


def test_llm_prompt_carries_recall_with_its_outcome_and_score():
    """The model is given what happened and how relevant it is. Without the
    outcome label it cannot tell a fix from a dead end, which is the whole
    point of the system."""
    prompt = LLMEngine().build_prompt(
        "the app crashes again on a large PDF", _recalled(), True
    )
    assert "[worked] Update application (relevance 0.67)" in prompt
    assert "[failed] Restart application (relevance 0.41)" in prompt
    assert "Returning customer: True" in prompt


def test_llm_prompt_says_so_when_there_is_no_recall():
    prompt = LLMEngine().build_prompt("how do I export to CSV?", [], False)
    assert "(no relevant history)" in prompt
    assert "Returning customer: False" in prompt


def test_llm_prompt_carries_what_was_just_learned():
    """The step the customer just reported is not in recall yet, because recall
    runs first. If the prompt omits it, the model re-suggests a failed step or
    escalates a problem that was just solved."""
    learned = [Memory(customer_id="CUST-1", kind=MemoryKind.ATTEMPT,
                      text="Clear application cache", outcome=Outcome.WORKED)]
    prompt = LLMEngine().build_prompt("clearing the cache fixed it", [], False, learned)
    assert "Just learned from that same message:" in prompt
    assert "[worked] Clear application cache" in prompt


def test_llm_prompt_says_nothing_new_rather_than_an_empty_block():
    prompt = LLMEngine().build_prompt("hello", [], False)
    assert "(nothing new)" in prompt


def test_llm_engine_returns_the_model_text_and_sends_no_history_it_invented():
    client = FakeClient(reply="Skip the restart; update the app.")
    engine = LLMEngine(client=client)
    text = engine.reply("crashes again on large PDF", _recalled(), True)

    assert text == "Skip the restart; update the app."
    call = client.messages.calls[0]
    assert call["model"] == engine.model
    assert call["system"]
    # The customer's own words are the only source of what happened.
    assert "crashes again on large PDF" in call["messages"][0]["content"]


def test_llm_engine_is_selected_when_a_key_is_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-a-real-key")
    assert build_engine().name == "llm"


def test_scripted_engine_is_the_default_without_a_key(monkeypatch):
    """The demo must run for a judge who has not configured anything."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert build_engine().name == "scripted"


def test_llm_engine_sends_no_temperature_to_a_client_that_dropped_it():
    """Anthropic removed `temperature` in SDK 1.9.0. Passing it anyway raises
    TypeError on every call, which the whole test suite misses because the
    other tests inject a fake client."""
    client = FakeClient()
    LLMEngine(client=client).reply("hi", [], False)
    assert "temperature" not in client.messages.calls[0]


def test_llm_engine_still_sends_temperature_when_the_client_accepts_it():
    class LegacyMessages:
        def __init__(self):
            self.calls = []

        def create(self, *, temperature=None, **kwargs):
            self.calls.append({"temperature": temperature, **kwargs})
            return type("M", (), {"content": [type("B", (), {"text": "ok"})()]})()

    messages = LegacyMessages()
    client = type("C", (), {"messages": messages})()
    LLMEngine(client=client, temperature=0.2).reply("hi", [], False)
    assert messages.calls[0]["temperature"] == 0.2


def test_a_failed_model_call_degrades_instead_of_crashing_the_demo():
    """A 400 for an exhausted credit balance, or a network blip mid-demo, must
    not reach the customer as a stack trace. The answer gets plainer; the
    conversation keeps going."""
    class BrokenMessages:
        def create(self, **kwargs):
            raise RuntimeError("credit balance is too low")

    client = type("C", (), {"messages": BrokenMessages()})()
    engine = LLMEngine(client=client, fallback=ScriptedEngine())

    text = engine.reply("crashes on large PDF", [], False)

    assert "no history on file" in text
    assert "credit balance is too low" in engine.degraded_reason


def test_a_failed_model_call_still_raises_when_there_is_no_fallback():
    client = type("C", (), {"messages": type(
        "M", (), {"create": lambda self, **k: (_ for _ in ()).throw(RuntimeError("down"))}
    )()})()
    with pytest.raises(RuntimeError):
        LLMEngine(client=client).reply("hi", [], False)


def test_the_keyed_engine_carries_a_fallback_so_a_demo_cannot_die(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-a-real-key")
    engine = build_engine()
    assert engine.name == "llm"
    assert engine._fallback is not None


# ---------------------------------------------------------- Hindsight store


def _hindsight(responder) -> tuple[HindsightStore, list[httpx.Request]]:
    """A HindsightStore wired to a canned transport, plus the requests seen."""
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responder(request)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    return HindsightStore(base_url="http://hindsight.test", bank="support",
                          client=client), seen


class _Banks:
    """The async-only ``client.banks`` namespace, as the real client exposes it."""

    def __init__(self, outer: "FakeHindsight") -> None:
        self._outer = outer

    async def list_banks(self, **kwargs):
        self._outer.calls.append(("banks.list_banks", kwargs))
        return SimpleNamespace(banks=[SimpleNamespace(bank_id=b)
                                      for b in self._outer.memories])


class FakeHindsight:
    """Stand-in for ``hindsight_client.Hindsight``, recording every call.

    Pinned to the real client's surface after reading the installed package
    rather than the docs, which is how three mistakes were caught: ``retain``
    types ``metadata`` as ``dict[str, str]`` and ``timestamp`` as a
    ``datetime``, and both ``list_banks`` and ``clear_bank_memories`` exist
    only on the async namespaces. An earlier version of this file asserted a
    guessed REST contract -- ``POST /memories``, ``POST /recall`` -- that does
    not exist at all.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.memories: dict[str, list[dict]] = {}
        self.raise_on: set[str] = set()
        self.banks = _Banks(self)

    def _call(self, name: str, **kwargs):
        self.calls.append((name, kwargs))
        if name in self.raise_on:
            raise RuntimeError(f"hindsight {name} failed")
        return kwargs

    def retain(self, **kwargs):
        self._call("retain", **kwargs)
        self.memories.setdefault(kwargs["bank_id"], []).append(
            {"text": kwargs["content"], "metadata": dict(kwargs.get("metadata") or {}),
             "id": kwargs.get("document_id"), "created_at": kwargs.get("timestamp")}
        )
        return {"ok": True}

    def recall(self, **kwargs):
        self._call("recall", **kwargs)
        rows = self.memories.get(kwargs["bank_id"], [])
        return SimpleNamespace(results=[SimpleNamespace(text=r["text"],
                                                        metadata=r["metadata"],
                                                        score=0.9 - i * 0.1)
                                        for i, r in enumerate(rows[:3])])

    def reflect(self, **kwargs):
        self._call("reflect", **kwargs)
        return SimpleNamespace(text="reflected")

    def list_memories(self, **kwargs):
        self._call("list_memories", **kwargs)
        return SimpleNamespace(items=list(self.memories.get(kwargs["bank_id"], [])))

    def delete_bank(self, bank_id: str, **kwargs):
        self._call("delete_bank", bank_id=bank_id, **kwargs)
        self.memories.pop(bank_id, None)

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


def _store(fake: FakeHindsight) -> HindsightStore:
    return HindsightStore(base_url="http://hindsight.test", client=fake)


def test_each_customer_gets_their_own_hindsight_bank():
    """Banks are fully isolated inside Hindsight, so per-customer isolation is
    the service's guarantee rather than a filter the adapter must remember."""
    store = _store(FakeHindsight())
    assert store.bank_for("CUST-1042") == "support-cust-1042"
    assert store.bank_for("CUST-9999") != store.bank_for("CUST-1042")


def test_hindsight_writes_a_memory_through_retain():
    fake = FakeHindsight()
    memory = Memory(customer_id="CUST-1", kind=MemoryKind.ATTEMPT,
                    text="Update application", outcome=Outcome.WORKED)
    _store(fake).add(memory)

    name, kwargs = fake.calls[0]
    assert name == "retain"
    assert kwargs["bank_id"] == "support-cust-1"
    assert kwargs["content"] == "Update application"
    # Structured fields ride along as metadata so recall never has to re-parse
    # prose to learn an outcome -- this is what keeps fixes separable from
    # failures against a service that returns plain fact text.
    assert kwargs["metadata"]["outcome"] == "worked"
    assert kwargs["metadata"]["kind"] == "attempt"
    assert kwargs["metadata"]["customer_id"] == "CUST-1"
    # The installed client types metadata as dict[str, str] and timestamp as a
    # datetime, so neither may be sent as a list or an ISO string.
    assert all(isinstance(v, str) for v in kwargs["metadata"].values())
    assert isinstance(kwargs["timestamp"], datetime)
    # Synchronous retain: the caller expects it recallable on the next turn.
    assert kwargs["retain_async"] is False


def test_symptom_terms_survive_the_string_metadata_round_trip():
    """They go over the wire comma-joined, so the split is the only thing
    keeping a generic step retrievable by the words the customer used."""
    fake = FakeHindsight()
    store = _store(fake)
    store.add(Memory(customer_id="CUST-1", kind=MemoryKind.ATTEMPT, text="Restart application",
                     symptom_terms=["crash", "pdf", "upload"]))
    assert store.all("CUST-1")[0].symptom_terms == ["crash", "pdf", "upload"]


def test_hindsight_reads_back_a_memory_the_server_stored():
    fake = FakeHindsight()
    store = _store(fake)
    memory = Memory(customer_id="CUST-1", kind=MemoryKind.INCIDENT,
                    text="crashes on large PDF upload")
    store.add(memory)

    recalled = store.all("CUST-1")

    # An empty query is a listing, not a semantic search -- asking Hindsight to
    # recall on "" returns nothing useful.
    assert fake.names() == ["retain", "list_memories"]
    assert recalled[0].kind is MemoryKind.INCIDENT
    assert recalled[0].customer_id == "CUST-1"
    assert recalled[0].id == memory.id


def test_hindsight_delegates_recall_to_the_server():
    fake = FakeHindsight()
    store = _store(fake)
    store.add(Memory(customer_id="CUST-1", kind=MemoryKind.ATTEMPT,
                     text="Update application", outcome=Outcome.WORKED))

    results = store.search("CUST-1", "crashes on large PDF")

    _, kwargs = [c for c in fake.calls if c[0] == "recall"][0]
    assert kwargs["bank_id"] == "support-cust-1"
    assert kwargs["query"] == "crashes on large PDF"
    assert results[0][0].text == "Update application"
    assert results[0][0].outcome is Outcome.WORKED
    assert 0 < results[0][1] <= 1


def test_hindsight_lists_customers_by_bank_prefix():
    fake = FakeHindsight()
    store = _store(fake)
    store.add(Memory(customer_id="CUST-1", kind=MemoryKind.ATTEMPT, text="Restart"))
    store.add(Memory(customer_id="CUST-2", kind=MemoryKind.ATTEMPT, text="Reinstall"))

    assert store.customers() == ["cust-1", "cust-2"]


def test_hindsight_forget_clears_one_customers_bank():
    fake = FakeHindsight()
    store = _store(fake)
    store.add(Memory(customer_id="CUST-1", kind=MemoryKind.ATTEMPT, text="Restart"))
    store.add(Memory(customer_id="CUST-2", kind=MemoryKind.ATTEMPT, text="Reinstall"))

    assert store.forget("CUST-1") == 1
    assert "delete_bank" in fake.names()
    # The other customer's bank is untouched.
    assert store.all("CUST-2")[0].text == "Reinstall"


def test_hindsight_surfaces_a_server_error_instead_of_returning_nothing():
    """A failing service must not read as 'this customer has no memory'. That
    would look like a recall failure and quietly change what the agent says."""
    fake = FakeHindsight()
    fake.raise_on = {"retain"}
    with pytest.raises(RuntimeError, match="retain failed"):
        _store(fake).add(Memory(customer_id="CUST-1", kind=MemoryKind.ATTEMPT, text="x"))


def test_a_record_missing_metadata_degrades_instead_of_raising():
    """One absent field must not lose the whole recall."""
    fake = FakeHindsight()
    fake.memories["support-cust-1"] = [{"text": "Update application", "metadata": {}}]
    memories = _store(fake).all("CUST-1")
    assert memories[0].text == "Update application"
    assert memories[0].kind is MemoryKind.NOTE


def test_the_adapter_only_calls_methods_the_installed_client_has():
    """Pin the adapter to the real package, not to a hand-written double.

    The docs and the installed `hindsight-client` disagree in three places --
    `list_banks` and `clear_bank_memories` are async-only namespace methods,
    `metadata` is `dict[str, str]`, and `timestamp` is a `datetime`. A fake
    that mirrors the adapter proves nothing, so this asserts against whatever
    `pip install hindsight-client` actually gives us. It skips rather than
    fails when the optional dependency is absent, so a judge can still run the
    suite with no extras.
    """
    client_mod = pytest.importorskip("hindsight_client")
    from memory_agent import store as store_mod

    for method in ("retain", "recall", "reflect", "list_memories", "delete_bank"):
        assert hasattr(client_mod.Hindsight, method), f"client lost {method}"
    assert hasattr(client_mod.Hindsight(base_url="http://x").banks, "list_banks")

    source = inspect.getsource(store_mod.HindsightStore)
    # keyword arguments the adapter passes, checked against the real signature
    import re

    for match in re.finditer(r"self\._client\.(\w+)\(", source):
        name = match.group(1)
        assert hasattr(client_mod.Hindsight, name), f"adapter calls missing {name}"

    retain_params = inspect.signature(client_mod.Hindsight.retain).parameters
    assert str(retain_params["metadata"].annotation) == "dict[str, str] | None"
    assert str(retain_params["timestamp"].annotation) == "datetime.datetime | None"


def test_retain_is_sent_with_the_types_the_client_declares():
    """The string-metadata and datetime-timestamp rules, enforced end to end."""
    fake = FakeHindsight()
    _store(fake).add(Memory(customer_id="CUST-1", kind=MemoryKind.ATTEMPT,
                            text="Restart application", symptom_terms=["crash"]))
    _, kwargs = fake.calls[0]
    assert isinstance(kwargs["timestamp"], datetime)
    assert all(isinstance(v, str) for v in kwargs["metadata"].values())


def test_hindsight_store_refuses_to_build_without_a_url(monkeypatch):
    monkeypatch.delenv("HINDSIGHT_URL", raising=False)
    with pytest.raises(ValueError, match="HINDSIGHT_URL"):
        HindsightStore()


def test_hindsight_is_selected_when_configured(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_URL", "http://hindsight.test")
    monkeypatch.setattr("hindsight_client.Hindsight", lambda **kw: FakeHindsight())
    store = build_store()
    assert isinstance(store, ResilientStore)
    assert isinstance(store.primary, HindsightStore)


def test_an_unreachable_hindsight_degrades_instead_of_breaking_the_page():
    """A demo that dies because an instance is asleep, or the wifi is a hotel's,
    is worse than one that says so. Every call must land somewhere."""
    class Dead:
        def __getattr__(self, name):
            def boom(*a, **k):
                raise OSError("Connection refused")
            return boom

    store = ResilientStore(Dead(), InMemoryStore())
    assert store.degraded_reason is None

    store.add(Memory(customer_id="CUST-1", kind=MemoryKind.ATTEMPT, text="Restart"))

    assert "Connection refused" in store.degraded_reason
    assert store.all("CUST-1")[0].text == "Restart"
    assert store.search("CUST-1", "restart")
    assert "unavailable" in store.name
    # The switch is one-way on purpose: flipping back mid-session would split a
    # customer's history across two backends, and the agent would then answer
    # from a memory that never existed as far as it can tell.
    store.degraded_reason = None
    assert store.all("CUST-1")[0].text == "Restart"


def test_a_client_error_is_not_hidden_behind_the_fallback():
    """4xx means this adapter sent something wrong. Falling back would produce
    a working-looking demo that silently never reaches Hindsight."""
    class Rejecting:
        def __getattr__(self, name):
            def boom(*a, **k):
                error = RuntimeError("422 unprocessable")
                error.status_code = 422
                raise error
            return boom

    store = ResilientStore(Rejecting(), InMemoryStore())
    with pytest.raises(RuntimeError, match="422"):
        store.add(Memory(customer_id="CUST-1", kind=MemoryKind.ATTEMPT, text="x"))
    assert store.degraded_reason is None

def test_local_store_is_the_default_without_configuration(monkeypatch):
    """Same reason as the scripted engine: a judge runs it with no env set."""
    monkeypatch.delenv("HINDSIGHT_URL", raising=False)
    from memory_agent.store import InMemoryStore

    assert isinstance(build_store(), InMemoryStore)


# ---------------------------------------------- the agent over a remote store


def test_agent_works_unchanged_over_the_remote_store():
    """The point of the MemoryStore protocol: `SupportAgent` is written against
    the interface, so swapping backends changes no agent code. This runs the
    full conversation against a fake Hindsight service.

    The fake deliberately ignores lexical scoring, unlike InMemoryStore,
    because a real service does its own retrieval. The agent's contribution --
    extract, store, resolve incidents, report what it learned -- is unchanged
    either way.
    """
    store = _store(FakeHindsight())
    agent = SupportAgent(store=store)

    agent.ask("CUST-1", "My Acme PDF Suite crashes whenever I upload a large PDF")
    reply = agent.ask("CUST-1", "I restarted it but that did not help")
    reply = agent.ask("CUST-1", "Updated the app and it fixed it")

    stored = store.all("CUST-1")
    attempts = {m.text: m.outcome for m in stored if m.kind is MemoryKind.ATTEMPT}
    assert attempts == {
        "Restart application": Outcome.FAILED,
        "Update application": Outcome.WORKED,
    }
    # The fix the customer just reported is acknowledged, not escalated.
    assert "escalate" not in reply.text.lower()
    # And a second customer on the same service sees nothing of the first:
    # separate banks, so CUST-1's fix is not even in the store being searched.
    fresh = agent.ask("CUST-2", "the app crashes again")
    assert fresh.is_returning_customer is False
    assert all(r.memory.customer_id == "CUST-2" for r in fresh.used_memories)
    assert [m.customer_id for m in store.all("CUST-1")] == ["CUST-1"] * len(store.all("CUST-1"))
    assert all("CUST-1" not in (m.customer_id,) for m in store.all("CUST-2"))
