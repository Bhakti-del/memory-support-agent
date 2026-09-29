"""Reasoning engines.

`ScriptedEngine` is deterministic and needs no API key, so the memory loop is
always demonstrable. `LLMEngine` is the drop-in upgrade for a real model.
"""

from __future__ import annotations

import os
from typing import Protocol, Sequence

from .models import Memory, MemoryKind, Outcome, RetrievedMemory, tokenize

SYSTEM_PROMPT = (
    "You are a senior support engineer. Use the recalled customer history to "
    "avoid repeating steps that already failed, and to re-surface fixes that "
    "worked before. Be concise and concrete."
)


class Engine(Protocol):
    name: str

    def reply(self, question: str, recalled: list[RetrievedMemory],
              is_returning: bool, just_learned: Sequence[Memory] = ()) -> str: ...


def _stem_set(text: str) -> set[str]:
    """Tokenize, then truncate to a 4-char stem so inflections collapse.

    "Updating the application" and "Update application" both yield
    {updat, appl} and therefore look the same to the dedupe below. Four
    characters rather than five: at five, "restart" and "reinstall" collapse
    together and a real step gets silently dropped.
    """
    return {t[:4] for t in tokenize(text)}


def _dedupe(evidence: list[RetrievedMemory]) -> list[str]:
    """List the distinct steps in a set of attempts, preserving recall order.

    The same step can be recorded more than once — the customer reports it in
    one turn and again later — and showing "Restart application" twice reads as
    a bug.

    Matching is by stem set, so "Restart application" and "restarted the app"
    collapse. Distinct steps are never merged, however similar the wording,
    because dropping a step means the agent may repeat something the customer
    already ruled out.

    Incidents are skipped defensively: a problem the customer had is not a fix,
    even once it is resolved, so it must never reach either list.
    """
    seen: list[set[str]] = []
    kept: list[str] = []
    for recalled in evidence:
        if recalled.memory.kind is not MemoryKind.ATTEMPT:
            continue
        stems = _stem_set(recalled.memory.text)
        if not stems or stems in seen:
            continue
        seen.append(stems)
        kept.append(recalled.memory.text)
    return kept


class ScriptedEngine:
    """Deterministic reasoner. Demonstrates the memory loop with zero setup."""

    name = "scripted"

    def reply(self, question: str, recalled: list[RetrievedMemory],
              is_returning: bool, just_learned: Sequence[Memory] = ()) -> str:
        # A message that reports a working step is the customer handing us the
        # solution. Recall deliberately runs before ingest so a customer never
        # sees their own sentence as history, which means that step is not in
        # `recalled` yet -- so without this branch the agent would answer a
        # solved problem with "everything failed, I'll escalate", which is
        # exactly backwards.
        just_solved = [
            m.text for m in just_learned
            if m.kind is MemoryKind.ATTEMPT and m.outcome is Outcome.WORKED
        ]
        if just_solved:
            return (
                f"Good to hear that worked. I've recorded {' and '.join(just_solved)} "
                f"as the fix for this problem, so I'll try that first if the issue "
                f"comes back."
            )

        # Only attempts are fixes or failures. An incident is a problem the
        # customer had; recording that it was solved does not make the problem
        # a fix, and listing it as one reads as nonsense.
        evidence = [r for r in recalled if r.memory.kind is MemoryKind.ATTEMPT]
        worked = [r for r in evidence if r.memory.outcome is Outcome.WORKED]
        failed = [r for r in evidence if r.memory.outcome is Outcome.FAILED]

        if not worked and not failed:
            if is_returning:
                return (
                    "Welcome back. I have your environment on file, but nothing "
                    "relevant to this specific problem yet, so I'll treat it as "
                    "new.\n\nStarting points: capture the exact error, confirm "
                    "the affected version, and try a clean restart."
                )
            return (
                f"I've logged this issue for you. I have no history on file, so I'll "
                f"work through standard diagnostics.\n\n"
                f"Starting points: capture the exact error, confirm the affected "
                f"version, and try a clean restart."
            )

        known_fixes = _dedupe(worked)
        known_failures = _dedupe(failed)

        lines: list[str] = []
        if known_fixes:
            lines.append("From your previous sessions, this worked:")
            lines += [f"  - {f}" for f in known_fixes]
        if known_failures:
            lines.append("These already failed, so I'll skip them:")
            lines += [f"  - {f}" for f in known_failures]

        lines.append("")
        if known_fixes:
            lines.append(
                f"Recommended next step: re-apply the fix above. If it regressed, "
                f"tell me and I'll open an escalation with your prior session attached."
            )
        else:
            lines.append(
                f"Every step I have on record for this failed, so I'd stop guessing "
                f"and escalate. If you can tell me the exact error text and your "
                f"version, I'll attach them to the escalation."
            )

        head = "Welcome back." if is_returning else "Based on your history,"
        return f"{head}\n\n" + "\n".join(lines)


class LLMEngine:
    """Real model call. Enabled when an API key is present.

    The client is injectable so the request path can be tested without a key
    and without a network. The model is never asked to guess what happened
    before; everything it knows about the customer arrives through memory.
    """

    name = "llm"

    def __init__(self, model: str = "claude-sonnet-5", temperature: float = 0.2,
                 client: object | None = None) -> None:
        self.model = model
        self.temperature = temperature
        self._client = client

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def build_prompt(self, question: str, recalled: list[RetrievedMemory],
                     is_returning: bool,
                     just_learned: Sequence[Memory] = ()) -> str:
        """Everything the model is told, as one user turn. Pure, so testable."""
        history = "\n".join(
            f"- [{r.memory.outcome.value}] {r.memory.text} (relevance {r.score:.2f})"
            for r in recalled
        ) or "(no relevant history)"
        learned = "\n".join(
            f"- [{m.outcome.value}] {m.text}" for m in just_learned
        ) or "(nothing new)"
        return (
            f"Customer question: {question}\n\n"
            f"Recalled history for this customer:\n{history}\n\n"
            f"Just learned from that same message:\n{learned}\n\n"
            f"Returning customer: {is_returning}\n\n"
            f"If the customer reported a step that worked, confirm it and say it "
            f"is now the recorded fix. Do not suggest escalating a problem they "
            f"have just said is resolved."
        )

    def reply(self, question: str, recalled: list[RetrievedMemory],
              is_returning: bool, just_learned: Sequence[Memory] = ()) -> str:
        message = self._get_client().messages.create(
            model=self.model,
            max_tokens=700,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self.build_prompt(
                question, recalled, is_returning, just_learned)}],
        )
        return message.content[0].text


def build_engine() -> Engine:
    if os.getenv("ANTHROPIC_API_KEY"):
        return LLMEngine()
    return ScriptedEngine()
