"""Core domain models for the memory-enabled support agent."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class MemoryKind(str, Enum):
    ATTEMPT = "attempt"          # a troubleshooting step that was tried
    INCIDENT = "incident"        # a resolved/recurring problem
    PROFILE = "profile"          # stable customer environment / preferences
    NOTE = "note"                # free-form observation


class Outcome(str, Enum):
    WORKED = "worked"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    UNKNOWN = "unknown"


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "it", "to", "of", "and",
    "my", "i", "again", "still", "when", "every", "keeps", "keep", "does",
    "do", "not", "no", "on", "in", "for", "with", "this", "that", "have",
    "has", "be", "can", "will", "would", "there", "me", "you", "your", "we",
    "us", "they", "them", "some", "any", "just", "very", "much", "than",
    "then", "but", "so", "if", "or", "as", "at", "by", "from", "get", "got",
    "please", "help", "thanks", "hello", "hi", "hey", "problem", "issue",
}

# Words whose meaning shifts with context but that carry the product identity.
# "app crashes" and "application crashes" are the same complaint, so they are
# normalized rather than dropped.
CANONICAL = {
    "app": "app", "apps": "app", "application": "app", "applications": "app",
    "software": "app", "program": "app", "tool": "app", "client": "app",
    "pc": "app", "laptop": "app", "computer": "app", "machine": "app",
}

_SUFFIXES = ("ingly", "edly", "ing", "ied", "ies", "ed", "es", "s")


def _stem(word: str) -> str:
    """Light suffix stripping so 'crashes'/'crashing'/'crash' collapse."""
    if word in CANONICAL or len(word) <= 4:
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase, drop stopwords, canonicalize product words, and stem.

    Stemming is what lets a customer's own phrasing ("it crashes when I upload
    a big PDF") match a stored step ("Update application"), without anyone
    having to hand-label the symptom.
    """
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    out: list[str] = []
    for token in cleaned.split():
        if len(token) <= 2 or token in STOPWORDS:
            continue
        out.append(CANONICAL.get(token) or _stem(token))
    return out


@dataclass
class Memory:
    """One durable fact the agent learned about a customer."""

    customer_id: str
    kind: MemoryKind
    text: str
    outcome: Outcome = Outcome.UNKNOWN
    product: str | None = None          # e.g. "Acme PDF Suite"
    symptom_terms: list[str] = field(default_factory=list)
    session_id: str | None = None
    created_at: str = field(default_factory=now)
    id: str = field(default_factory=lambda: new_id("mem"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Memory":
        return cls(
            id=raw["id"],
            customer_id=raw["customer_id"],
            kind=MemoryKind(raw["kind"]),
            text=raw["text"],
            outcome=Outcome(raw["outcome"]),
            product=raw.get("product"),
            symptom_terms=raw.get("symptom_terms", []),
            session_id=raw.get("session_id"),
            created_at=raw.get("created_at", now()),
        )


@dataclass
class RetrievedMemory:
    memory: Memory
    score: float
    reason: str


@dataclass
class SupportReply:
    text: str
    used_memories: list[RetrievedMemory] = field(default_factory=list)
    known_failures: list[str] = field(default_factory=list)
    known_fixes: list[str] = field(default_factory=list)
    is_returning_customer: bool = False
    learned: list[Memory] = field(default_factory=list)
    engine: str = "scripted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "engine": self.engine,
            "is_returning_customer": self.is_returning_customer,
            "known_failures": self.known_failures,
            "known_fixes": self.known_fixes,
            "learned": [
                {"text": m.text, "outcome": m.outcome.value, "kind": m.kind.value}
                for m in self.learned
            ],
            "used_memories": [
                {
                    "id": r.memory.id,
                    "kind": r.memory.kind.value,
                    "outcome": r.memory.outcome.value,
                    "text": r.memory.text,
                    "score": round(r.score, 3),
                    "reason": r.reason,
                }
                for r in self.used_memories
            ],
        }


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)
