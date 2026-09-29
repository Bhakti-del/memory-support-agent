"""Memory storage backends.

Two interchangeable implementations share the ``MemoryStore`` protocol:

* ``InMemoryStore``  - zero-dependency local store, used by the demo/tests.
* ``HindsightStore`` - client for a real Hindsight memory service (the memory
  system this hackathon requires), used when ``HINDSIGHT_URL`` is set. Same
  interface, so the agent code never changes.

Hindsight stores each customer's memories in their own bank and retrieves them
with semantic + keyword + graph + temporal search. The local store is a
fallback so the repo runs with no service, no key and no network.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Protocol

from .models import Memory, MemoryKind, Outcome, now, tokenize


class MemoryStore(Protocol):
    def add(self, memory: Memory) -> Memory: ...
    def update(self, memory: Memory) -> Memory: ...
    def all(self, customer_id: str) -> list[Memory]: ...
    def customers(self) -> list[str]: ...
    def forget(self, customer_id: str) -> int: ...
    def search(self, customer_id: str, query: str, limit: int = 5) -> list[tuple[Memory, float]]: ...


def _score(memory: Memory, query_tokens: set[str]) -> float:
    """Lexical overlap score. Real Hindsight does this with embeddings."""
    if not query_tokens:
        return 0.0
    haystack = set(tokenize(memory.text)) | set(memory.symptom_terms)
    overlap = query_tokens & haystack
    if not overlap:
        return 0.0
    return len(overlap) / len(query_tokens)


class InMemoryStore:
    """Simple append-only JSON store, grouped by customer."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._by_customer: dict[str, list[Memory]] = {}
        if self.path and self.path.exists():
            self._load()

    def add(self, memory: Memory) -> Memory:
        self._by_customer.setdefault(memory.customer_id, []).append(memory)
        self._persist()
        return memory

    def update(self, memory: Memory) -> Memory:
        """Replace a memory in place. Used to resolve an open incident."""
        memories = self._by_customer.get(memory.customer_id, [])
        for index, existing in enumerate(memories):
            if existing.id == memory.id:
                memories[index] = memory
                self._persist()
                return memory
        return self.add(memory)

    def all(self, customer_id: str) -> list[Memory]:
        return list(self._by_customer.get(customer_id, []))

    def customers(self) -> list[str]:
        return sorted(self._by_customer)

    def forget(self, customer_id: str) -> int:
        """Drop every memory for a customer. Returns how many were removed."""
        removed = len(self._by_customer.pop(customer_id, []))
        if removed:
            self._persist()
        return removed

    def search(self, customer_id: str, query: str, limit: int = 5) -> list[tuple[Memory, float]]:
        q = set(tokenize(query))
        scored = [(m, _score(m, q)) for m in self.all(customer_id)]
        hits = [s for s in scored if s[1] > 0]
        hits.sort(key=lambda pair: (pair[1], pair[0].created_at), reverse=True)
        return hits[:limit]

    def _persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            cid: [m.to_dict() for m in mems] for cid, mems in self._by_customer.items()
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._by_customer = {
            cid: [Memory.from_dict(m) for m in mems] for cid, mems in raw.items()
        }


class HindsightStore:
    """Adapter for a real Hindsight memory server (https://hindsight.vectorize.io).

    One memory **bank** per customer. Banks are fully isolated inside
    Hindsight, so per-customer isolation is enforced by the service rather than
    by a filter this adapter has to remember to apply — CUST-1's fix can never
    surface for CUST-2 because it is not in their bank.

    Recall is delegated to Hindsight's own retriever (semantic + BM25 + graph
    + temporal, fused by RRF), which is the point of using it: it matches
    "won't open" against "fails to launch" where the local store's token
    overlap scores zero.

    The client is injectable so the contract is testable with no server and no
    network, and so a demo can run against a stub when no instance is up.
    """

    def __init__(self, base_url: str | None = None, bank_prefix: str = "support",
                 client: object | None = None) -> None:
        self.base_url = (base_url or os.getenv("HINDSIGHT_URL", "")).rstrip("/")
        if not self.base_url:
            raise ValueError("HindsightStore requires HINDSIGHT_URL")
        self.bank_prefix = os.getenv("HINDSIGHT_BANK_PREFIX", bank_prefix)
        self.api_key = os.getenv("HINDSIGHT_API_KEY") or None
        if client is None:
            from hindsight_client import Hindsight

            client = Hindsight(base_url=self.base_url, api_key=self.api_key)
        self._client = client

    def bank_for(self, customer_id: str) -> str:
        """Hindsight bank id for a customer. The prefix keeps demo banks
        distinguishable from anything else in a shared instance."""
        return f"{self.bank_prefix}-{customer_id.lower()}"

    def add(self, memory: Memory) -> Memory:
        """Retain a memory.

        ``metadata`` carries the structured fields the agent needs back on
        recall (kind, outcome, customer, memory id) so nothing has to be
        re-parsed out of prose. The client's ``metadata`` is typed
        ``dict[str, str]``, so the symptom terms go over the wire comma-joined
        and are split again in ``_to_memory``.

        ``retain_async=False`` because the caller expects the memory to be
        recallable on the very next turn, which is the whole point of the
        demo: report what was learned, then be able to use it immediately.
        """
        self._client.retain(
            bank_id=self.bank_for(memory.customer_id),
            content=memory.text,
            context=f"{memory.kind.value} memory for {memory.customer_id}",
            timestamp=_as_datetime(memory.created_at),
            document_id=memory.id,
            metadata={
                "memory_id": memory.id,
                "customer_id": memory.customer_id,
                "kind": memory.kind.value,
                "outcome": memory.outcome.value if memory.outcome else "",
                "symptom_terms": ",".join(memory.symptom_terms),
                "session_id": memory.session_id or "",
                "product": memory.product or "",
            },
            retain_async=False,
        )
        return memory

    def update(self, memory: Memory) -> Memory:
        """Resolve an incident.

        Hindsight's model is append-and-consolidate rather than row-update: a
        fact is superseded by retaining the newer one, and its consolidation
        resolves the two into an observation that carries the history. The
        client does expose ``memory.update_memory``, but it is async-only and
        this adapter is sync; retaining the resolved state is both simpler and
        closer to how Hindsight wants conflict resolution to work.
        """
        return self.add(memory)

    def all(self, customer_id: str) -> list[Memory]:
        return [m for m, _ in self.search(customer_id, "", limit=100)]

    def customers(self) -> list[str]:
        """Every customer with a bank in this deployment, by bank prefix.

        ``list_banks`` exists only on the async ``client.banks`` namespace, so
        it is driven to completion here. This is a control-plane call made
        once per UI render, not on the hot path.
        """
        banks = self._run_async(self._client.banks.list_banks())
        found = []
        for bank in _iter_items(banks, "banks"):
            name = _attr(bank, "bank_id") or _attr(bank, "id") or str(bank)
            if name.startswith(f"{self.bank_prefix}-"):
                found.append(name[len(self.bank_prefix) + 1:])
        return sorted(found)

    def forget(self, customer_id: str) -> int:
        """Clear one customer's bank. Returns how many memories were removed.

        Deletes the bank rather than clearing its memories, because that is the
        one sync call that does it, and because a bank is recreated
        automatically on the next retain. Counting first means the caller still
        learns how much was there.
        """
        before = self._count(customer_id)
        if before:
            self._client.delete_bank(self.bank_for(customer_id))
        return before

    def search(self, customer_id: str, query: str, limit: int = 5) -> list[tuple[Memory, float]]:
        """Recall from the customer's bank.

        An empty query means "list what we know", which is ``all()``'s job and
        not a semantic search — asking Hindsight to recall on an empty string
        would return nothing useful.
        """
        if not query.strip():
            return [(m, 0.0) for m in self._list(customer_id)]
        response = self._client.recall(
            bank_id=self.bank_for(customer_id),
            query=query,
            max_tokens=4096,
            budget="mid",
        )
        return [(_to_memory(r), _score_of(r)) for r in _results_of(response)][:limit]

    def reflect(self, customer_id: str, query: str, context: str | None = None) -> str:
        """Hindsight's agentic reasoning over a customer's own memory.

        Not used in the demo path — the support reply is composed from
        structured recall so every claim stays traceable to a memory — but it
        is the documented way to let Hindsight answer with its own
        consolidation, and it is exposed for experiments.
        """
        answer = self._client.reflect(
            bank_id=self.bank_for(customer_id), query=query, context=context
        )
        return _attr(answer, "text") or ""

    def _list(self, customer_id: str) -> list[Memory]:
        return [_to_memory(m) for m in _iter_items(
            self._client.list_memories(bank_id=self.bank_for(customer_id), limit=200),
            "items",
        )]

    def _count(self, customer_id: str) -> int:
        return len(self._list(customer_id))

    @staticmethod
    def _run_async(coroutine):
        """Drive a coroutine from sync code.

        The client's control-plane calls are async-only. If a loop is already
        running (an ASGI app, a notebook) the coroutine is run on a private
        loop in a worker thread rather than deadlocking on the current one.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coroutine).result()


class ResilientStore:
    """A store that keeps answering when its backend stops answering.

    Hindsight is a network service, and a demo that dies because an instance
    is asleep, rate-limited, or on a hotel wifi is worse than a demo that says
    so. Once a call to the real store raises for a reason that is not the
    caller's fault, every later call goes to a local store instead and
    ``degraded_reason`` records why.

    The switch is one-way on purpose. Flipping back mid-session would split a
    customer's history across two backends, and the agent would then answer
    from a memory that never existed as far as it can tell.
    """

    _TRANSIENT = (OSError, TimeoutError, ConnectionError, asyncio.TimeoutError)

    def __init__(self, primary: MemoryStore, fallback: MemoryStore) -> None:
        self.primary = primary
        self.fallback = fallback
        self.degraded_reason: str | None = None

    @property
    def name(self) -> str:
        if self.degraded_reason:
            return f"{_label(self.primary)} (unavailable, using local)"
        return _label(self.primary)

    def _call(self, method: str, *args, **kwargs):
        if self.degraded_reason is None:
            try:
                return getattr(self.primary, method)(*args, **kwargs)
            except self._TRANSIENT as exc:
                # A refused connection, DNS failure, or timeout means the
                # service is not there -- not that the memory is missing.
                self.degraded_reason = f"{type(exc).__name__}: {exc}"[:200]
            except Exception as exc:  # noqa: BLE001 - see note below
                # A client-side error from the SDK (bad auth shape, malformed
                # response) means this adapter is wrong, and falling back would
                # hide that behind a working-looking demo. Re-raise.
                if _is_client_error(exc):
                    raise
                self.degraded_reason = f"{type(exc).__name__}: {exc}"[:200]
        return getattr(self.fallback, method)(*args, **kwargs)

    def add(self, memory: Memory) -> Memory:
        return self._call("add", memory)

    def update(self, memory: Memory) -> Memory:
        return self._call("update", memory)

    def all(self, customer_id: str) -> list[Memory]:
        return self._call("all", customer_id)

    def customers(self) -> list[str]:
        return self._call("customers")

    def forget(self, customer_id: str) -> int:
        return self._call("forget", customer_id)

    def search(self, customer_id: str, query: str, limit: int = 5):
        return self._call("search", customer_id, query, limit)


def _is_client_error(exc: Exception) -> bool:
    """4xx means the request was wrong, so do not paper over it."""
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        return 400 <= status < 500
    code = getattr(exc, "code", None)
    return isinstance(code, str) and code.isdigit() and 400 <= int(code) < 500


def _label(store: object) -> str:
    return type(store).__name__.replace("Store", "") or "memory"


def _iter_items(payload: object, key: str) -> list:
    """Hindsight responses are pydantic models or plain dicts depending on
    version; unwrap ``{key: [...]}`` or take the model as a sequence."""
    if isinstance(payload, dict):
        return list(payload.get(key) or [])
    for name in (key, "items", "results"):
        found = getattr(payload, name, None)
        if isinstance(found, list):
            return list(found)
    if isinstance(payload, (list, tuple)):
        return list(payload)
    return []


def _attr(obj: object, name: str, default: object = None) -> object:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _results_of(response: object) -> list:
    for name in ("results", "items", "memories"):
        found = getattr(response, name, None)
        if isinstance(found, list):
            return found
    if isinstance(response, dict):
        for name in ("results", "items", "memories"):
            found = response.get(name)
            if isinstance(found, list):
                return found
    return []


def _score_of(result: object) -> float:
    for name in ("score", "relevance", "rank_score"):
        value = _attr(result, name)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _to_memory(record: object) -> Memory:
    """Rebuild a ``Memory`` from a Hindsight record.

    Hindsight returns the fact text plus the metadata we attached at retain
    time. If the metadata is missing or partial, the structured fields fall back
    to the raw text rather than raising — a partially-recovered memory is still
    worth showing, and losing the whole recall to one absent field would be a
    worse failure than a default.
    """
    if isinstance(record, str):
        return Memory(text=record)
    meta = _attr(record, "metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    text = _attr(record, "text") or _attr(record, "content") or ""
    created = _attr(record, "created_at") or _attr(record, "timestamp")
    if isinstance(created, datetime):
        created = created.isoformat()
    terms = meta.get("symptom_terms") or ""
    memory = Memory(
        customer_id=meta.get("customer_id") or _attr(record, "customer_id") or "",
        kind=_coerce(MemoryKind, meta.get("kind") or _attr(record, "kind"), MemoryKind.NOTE),
        outcome=_coerce(Outcome, meta.get("outcome") or _attr(record, "outcome"), None),
        text=text,
        product=meta.get("product") or _attr(record, "product"),
        symptom_terms=[t for t in str(terms).split(",") if t],
        session_id=meta.get("session_id") or _attr(record, "session_id") or None,
        created_at=created if isinstance(created, str) else now(),
    )
    memory.id = meta.get("memory_id") or _attr(record, "id") or memory.id
    return memory


def _as_datetime(value: str | None) -> datetime | None:
    """``retain`` wants a datetime; our ids and timestamps are ISO strings."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _coerce(enum_cls, value, default):
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            try:
                return enum_cls[value.upper()]
            except KeyError:
                return default
    return default


def build_store(path: str | Path | None = None) -> Iterable[MemoryStore]:
    """Pick the real service if configured, otherwise the local store.

    When Hindsight is configured it is wrapped in a ``ResilientStore``, so a
    configured-but-unreachable instance degrades to local JSON instead of
    taking the page down. The local store is the fallback, not a secret: a
    judge with no Hindsight instance still gets a working demo, and the header
    says which one is actually live.
    """
    if os.getenv("HINDSIGHT_URL"):
        return ResilientStore(  # type: ignore[return-value]
            HindsightStore(), InMemoryStore(path)
        )
    return InMemoryStore(path)  # type: ignore[return-value]
