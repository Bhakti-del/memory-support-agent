"""Memory storage backends.

Two interchangeable implementations share the ``MemoryStore`` protocol:

* ``InMemoryStore``  - zero-dependency local store, used by the demo/tests.
* ``HindsightStore`` - HTTP client for a real Hindsight memory service, used
  when ``HINDSIGHT_URL`` is set. Same interface, so the agent code never changes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Protocol

import httpx

from .models import Memory, tokenize


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
    """Adapter for a real Hindsight memory server.

    Enabled by setting HINDSIGHT_URL. It persists to the service instead of
    local JSON, and delegates recall to the service's retriever.
    """

    def __init__(self, base_url: str | None = None, bank: str = "support",
                 client: httpx.Client | None = None) -> None:
        self.base_url = (base_url or os.getenv("HINDSIGHT_URL", "")).rstrip("/")
        if not self.base_url:
            raise ValueError("HindsightStore requires HINDSIGHT_URL")
        self.bank = os.getenv("HINDSIGHT_BANK", bank)
        # The client is injectable so the wire contract can be tested against a
        # canned transport, with no server and no network.
        self._client = client if client is not None else httpx.Client(timeout=10.0)

    def add(self, memory: Memory) -> Memory:
        self._client.post(
            f"{self.base_url}/memories",
            json={"bank": self.bank, "customer_id": memory.customer_id,
                  "memory": memory.to_dict()},
        ).raise_for_status()
        return memory

    def update(self, memory: Memory) -> Memory:
        self._client.put(
            f"{self.base_url}/memories/{memory.id}",
            json={"bank": self.bank, "customer_id": memory.customer_id,
                  "memory": memory.to_dict()},
        ).raise_for_status()
        return memory

    def all(self, customer_id: str) -> list[Memory]:
        resp = self._client.get(
            f"{self.base_url}/memories", params={"bank": self.bank, "customer_id": customer_id}
        )
        resp.raise_for_status()
        return [Memory.from_dict(m) for m in resp.json().get("memories", [])]

    def customers(self) -> list[str]:
        resp = self._client.get(f"{self.base_url}/banks/{self.bank}/customers")
        resp.raise_for_status()
        return resp.json().get("customers", [])

    def forget(self, customer_id: str) -> int:
        resp = self._client.request(
            "DELETE",
            f"{self.base_url}/memories",
            params={"bank": self.bank, "customer_id": customer_id},
        )
        resp.raise_for_status()
        return resp.json().get("removed", 0)

    def search(self, customer_id: str, query: str, limit: int = 5) -> list[tuple[Memory, float]]:
        resp = self._client.post(
            f"{self.base_url}/recall",
            json={"bank": self.bank, "customer_id": customer_id,
                  "query": query, "limit": limit},
        )
        resp.raise_for_status()
        return [(Memory.from_dict(m["memory"]), float(m.get("score", 0.0)))
                for m in resp.json().get("results", [])]


def build_store(path: str | Path | None = None) -> Iterable[MemoryStore]:
    """Pick the real service if configured, otherwise the local store."""
    if os.getenv("HINDSIGHT_URL"):
        return HindsightStore()  # type: ignore[return-value]
    return InMemoryStore(path)  # type: ignore[return-value]
