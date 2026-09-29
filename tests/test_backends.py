"""Contract tests for the two pluggable backends.

Both `LLMEngine` and `HindsightStore` are the parts of this system that cannot
run in the demo: one needs an API key, the other needs a memory server. Left
untested they are the two places a bug hides until the morning of the demo.

So both are tested at their seam instead of against the real thing. The model
client and the HTTP transport are injected, and the tests assert on what
actually goes over the wire: which endpoint, which method, which body, and how
the response is parsed. No key, no network, no server.

What these do NOT prove: that Anthropic or Hindsight accepts this. A real
contract test needs a real service. `tests/test_agent.py` covers the behaviour
that sits on top of both.
"""

from __future__ import annotations

import json

import httpx
import pytest

from memory_agent import SupportAgent
from memory_agent.engine import LLMEngine, build_engine
from memory_agent.models import Memory, MemoryKind, Outcome, RetrievedMemory
from memory_agent.store import HindsightStore, build_store


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


def test_hindsight_writes_a_memory_to_the_documented_endpoint():
    store, seen = _hindsight(lambda r: httpx.Response(201, json={"ok": True}))
    memory = Memory(customer_id="CUST-1", kind=MemoryKind.ATTEMPT,
                    text="Update application", outcome=Outcome.WORKED)
    store.add(memory)

    request = seen[0]
    assert request.method == "POST"
    assert request.url.path == "/memories"
    body = json.loads(request.content)
    assert body["bank"] == "support"
    assert body["customer_id"] == "CUST-1"
    assert body["memory"]["text"] == "Update application"
    assert body["memory"]["outcome"] == "worked"


def test_hindsight_reads_back_a_memory_the_server_stored():
    record = Memory(customer_id="CUST-1", kind=MemoryKind.INCIDENT,
                    text="crashes on large PDF upload").to_dict()
    store, seen = _hindsight(lambda r: httpx.Response(200, json={"memories": [record]}))

    memories = store.all("CUST-1")

    assert seen[0].url.path == "/memories"
    assert seen[0].url.params["customer_id"] == "CUST-1"
    assert seen[0].url.params["bank"] == "support"
    assert [m.text for m in memories] == ["crashes on large PDF upload"]
    # Round-tripping is the coupling point: a shape the server does not return
    # would otherwise blow up deep inside the agent, not at the boundary.
    assert memories[0].kind is MemoryKind.INCIDENT


def test_hindsight_delegates_recall_to_the_server():
    record = Memory(customer_id="CUST-1", kind=MemoryKind.ATTEMPT,
                    text="Update application", outcome=Outcome.WORKED).to_dict()
    store, seen = _hindsight(
        lambda r: httpx.Response(200, json={"results": [{"memory": record, "score": 0.82}]})
    )

    results = store.search("CUST-1", "crashes on large PDF", limit=3)

    body = json.loads(seen[0].content)
    assert seen[0].method == "POST"
    assert seen[0].url.path == "/recall"
    assert body == {"bank": "support", "customer_id": "CUST-1",
                    "query": "crashes on large PDF", "limit": 3}
    assert results[0][1] == 0.82
    assert results[0][0].text == "Update application"


def test_hindsight_lists_customers_per_bank():
    store, seen = _hindsight(lambda r: httpx.Response(200, json={"customers": ["CUST-1"]}))
    assert store.customers() == ["CUST-1"]
    assert seen[0].url.path == "/banks/support/customers"


def test_hindsight_update_targets_one_memory_by_id():
    """Resolving an incident is an in-place edit, so it must not append."""
    store, seen = _hindsight(lambda r: httpx.Response(200, json={"ok": True}))
    memory = Memory(customer_id="CUST-1", kind=MemoryKind.INCIDENT,
                    text="crashes on large PDF", outcome=Outcome.WORKED)
    store.update(memory)

    assert seen[0].method == "PUT"
    assert seen[0].url.path == f"/memories/{memory.id}"


def test_hindsight_forget_deletes_a_customer():
    store, seen = _hindsight(lambda r: httpx.Response(200, json={"removed": 7}))
    assert store.forget("CUST-1") == 7
    assert seen[0].method == "DELETE"
    assert seen[0].url.params["customer_id"] == "CUST-1"


def test_hindsight_surfaces_a_server_error_instead_of_returning_nothing():
    """A 500 must not read as 'this customer has no memory'. That would look
    like a recall failure and quietly change what the agent says."""
    store, _ = _hindsight(lambda r: httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(httpx.HTTPStatusError):
        store.all("CUST-1")


def test_hindsight_store_refuses_to_build_without_a_url(monkeypatch):
    monkeypatch.delenv("HINDSIGHT_URL", raising=False)
    with pytest.raises(ValueError, match="HINDSIGHT_URL"):
        HindsightStore()


def test_hindsight_is_selected_when_configured(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_URL", "http://hindsight.test")
    assert isinstance(build_store(), HindsightStore)


def test_local_store_is_the_default_without_configuration(monkeypatch):
    """Same reason as the scripted engine: a judge runs it with no env set."""
    monkeypatch.delenv("HINDSIGHT_URL", raising=False)
    from memory_agent.store import InMemoryStore

    assert isinstance(build_store(), InMemoryStore)


# ---------------------------------------------- the agent over a remote store


def test_agent_works_unchanged_over_the_remote_store(monkeypatch):
    """The point of the MemoryStore protocol: `SupportAgent` is written against
    the interface, so swapping backends changes no agent code. This runs the
    full conversation against a canned Hindsight server.

    The store double here deliberately ignores lexical scoring, unlike
    InMemoryStore, because a real service does its own retrieval. The agent's
    contribution -- extract, store, resolve incidents, report what it learned --
    is unchanged either way.
    """
    records: dict[str, list[dict]] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        cid = body.get("customer_id") or request.url.params.get("customer_id", "")
        if request.method == "POST" and request.url.path == "/memories":
            records.setdefault(cid, []).append(body["memory"])
            return httpx.Response(201, json={"ok": True})
        if request.method == "GET" and request.url.path == "/memories":
            return httpx.Response(200, json={"memories": records.get(cid, [])})
        if request.method == "POST" and request.url.path == "/recall":
            return httpx.Response(200, json={"results": []})
        if request.method == "DELETE":
            removed = len(records.pop(cid, []))
            return httpx.Response(200, json={"removed": removed})
        return httpx.Response(200, json={})

    client = httpx.Client(transport=httpx.MockTransport(handle))
    store = HindsightStore(base_url="http://hindsight.test", client=client)
    agent = SupportAgent(store=store)

    agent.ask("CUST-1", "My Acme PDF Suite crashes whenever I upload a large PDF")
    reply = agent.ask("CUST-1", "I restarted it but that did not help")
    reply = agent.ask("CUST-1", "Updated the app and it fixed it")

    stored = [Memory.from_dict(r) for r in records["CUST-1"]]
    attempts = {m.text: m.outcome for m in stored if m.kind is MemoryKind.ATTEMPT}
    assert attempts == {
        "Restart application": Outcome.FAILED,
        "Update application": Outcome.WORKED,
    }
    # The fix the customer just reported is acknowledged, not escalated.
    assert "escalate" not in reply.text.lower()
    # And a second customer on the same service sees nothing of the first.
    assert agent.ask("CUST-2", "the app crashes again").is_returning_customer is False
