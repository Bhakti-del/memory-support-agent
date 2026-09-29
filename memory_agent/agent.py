"""The memory-enabled support agent: recall -> reason -> remember."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace

from .engine import Engine, build_engine
from .models import (
    Memory,
    MemoryKind,
    Outcome,
    RetrievedMemory,
    SupportReply,
    new_id,
    tokenize,
)
from .store import MemoryStore, build_store

WORKED_PATTERNS = re.compile(
    r"\b(that worked|it worked|this worked|fixed it|did work|worked\b|resolved|"
    r"solved|seems to be working|works now|back to normal)\b",
    re.I,
)
FAILED_PATTERNS = re.compile(
    r"\b(did not work|didn'?t work|does not work|doesn'?t work|did not help|"
    r"didn'?t help|did not fix|didn'?t fix|no luck|still crashing|still fails?|"
    r"failed|same error|no change|did nothing|neither helped|neither worked|"
    r"none of (them|these|those) (helped|worked)|no help|no good|no dice|"
    r"problem persists|persists|still there|still happening|still an issue|"
    r"nothing (worked|helped|fixed))\b",
    re.I,
)
# Phrases that contain a success word but mean the opposite. Checked before
# WORKED_PATTERNS, otherwise "nothing worked" reads as a success.
NEGATED_WORKED = re.compile(
    r"\b(nothing (worked|helped|fixed)|never worked|did not work|didn'?t work|"
    r"does not work|doesn'?t work|no longer works?)\b",
    re.I,
)
STEP_PATTERNS = re.compile(
    r"\b(restart|reboot|clear(ed|ing)? (the )?cache|update[ds]?|upgraded|reinstall|"
    r"reinstalled|roll(ed)? back|rollback|disable[ds]?|enable[ds]?|reindex|"
    r"checked (the )?logs?|reset|reboot(ed)?|install(ed)?|patch(ed)?|"
    r"incognito|private window|another browser|second monitor|"
    r"free up space|checked disk space|moved the file|converted|split the file)\b",
    re.I,
)

# Verbs a customer uses when reporting what they already tried. Ordered longest
# first so "cleared the cache" wins over the bare "clear".
STEP_LEXICON: tuple[tuple[str, str], ...] = (
    (r"restart(ed|ing)? (the )?(app|application|computer|machine|laptop|pc)?", "Restart application"),
    (r"rebooted|rebooting", "Restart application"),
    (r"cleared? (the )?(app(lication)? )?cache", "Cleared application cache"),
    (r"clear(ed|ing)? (the )?cache", "Cleared application cache"),
    (r"updat(ed|ing)|upgraded", "Update application"),
    (r"reinstall(ed|ing)", "Reinstall application"),
    (r"roll(ed)? back|rollback", "Roll back to previous version"),
    (r"re-?index(ed|ing)?", "Reindex"),
    (r"checked (the )?logs?", "Checked logs"),
    (r"freed up space|free up space|checked disk space", "Free up disk space"),
    (r"in private window|private window|incognito", "Tried a private window"),
    (r"another browser|different browser", "Tried a different browser"),
    (r"second monitor", "Tried a second monitor"),
    (r"moved the file", "Moved the file"),
    (r"convert(ed|ing)|split the file", "Converted or split the file"),
    (r"disabl(ed|ing)", "Disabled an add-on"),
    (r"enabl(ed|ing)", "Enabled an add-on"),
    (r"reset", "Reset settings"),
    (r"patch(ed|ing)", "Applied a patch"),
    (r"install(ed|ing)", "Reinstalled the app"),
)



# Clause boundaries inside a single message. Customers chain attempts with
# these: "I restarted it, cleared the cache and updating finally worked".
CLAUSE_SPLIT = re.compile(
    r"[.;!?]|\b(?:and then|then|after that|also|but|so|i also|i then|"
    r"tried|attempted|finally)\b",
    re.I,
)
PRODUCT_PATTERN = re.compile(
    r"\b((?:[A-Z][A-Za-z0-9]*\s+)*?[A-Z][A-Za-z0-9]*\s+"
    r"(?:Suite|Studio|Manager|Desktop|Cloud))\b"
)
# Sentence-initial words that are capitalised for grammar, not as a product name.
NOT_PRODUCT = {"My", "The", "Our", "Your", "Its", "Their", "This", "That", "When",
               "After", "Since", "Please", "Hi", "Hello", "Every", "Both", "Also"}

# Phrases that describe a symptom rather than a fix. Used to enrich a step's
# symptom terms from the conversation the step was taken in.
SYMPTOM_HINT = re.compile(
    r"\b(crash|crashes|crashing|crashed|error|errors|fail|fails|failing|failed|"
    r"hang|hangs|hanging|frozen|freeze|freezes|blank|freeze|timeout|time out|"
    r"cannot|can't|unable|slow|laggy|stuck|broken|not opening|won't open|"
    r"wont open|white screen|segfault)\b",
    re.I,
)


@dataclass
class CustomerContext:
    """What we infer about a customer from their message, before retrieval."""

    customer_id: str
    product: str | None = None
    symptoms: list[str] = field(default_factory=list)

    @property
    def query(self) -> str:
        return " ".join(self.symptoms)


class SupportAgent:
    def __init__(self, store: MemoryStore | None = None, engine: Engine | None = None) -> None:
        self.store = store if store is not None else build_store()
        self.engine = engine if engine is not None else build_engine()

    # ---------- public API ----------

    def ask(self, customer_id: str, message: str, session_id: str | None = None) -> SupportReply:
        """Answer a customer message using recalled memory, then store it."""
        session_id = session_id or new_id("ses")
        context = self._parse_context(message)

        # Recall against what we knew *before* this message, then ingest. The
        # order matters both ways: mining first would let a customer be shown
        # their own sentence back as history, and waiting until after recall
        # would leave steps mined from this message unsearchable.
        # A profile fact alone is not a relationship, so it does not make a
        # customer "returning".
        history = self.store.all(customer_id)
        is_returning = any(m.kind is not MemoryKind.PROFILE for m in history)
        recalled = self._recall(customer_id, context, history)

        learned = self._ingest(customer_id, message, context, session_id)
        if any(m.kind is MemoryKind.ATTEMPT and m.outcome is Outcome.WORKED for m in learned):
            # The problem just got solved, so it is no longer open.
            self._resolve_open_incident(customer_id)
        text = self.engine.reply(message, recalled, is_returning, learned)

        # Same rule as the engine: only attempts are fixes or failures. An
        # incident is a problem the customer had, so a resolved one showing up
        # under "known fixes" would quote their own complaint back at them.
        evidence = [r for r in recalled if r.memory.kind is MemoryKind.ATTEMPT]

        reply = SupportReply(
            text=text,
            used_memories=recalled,
            known_failures=[r.memory.text for r in evidence if r.memory.outcome is Outcome.FAILED],
            known_fixes=[r.memory.text for r in evidence if r.memory.outcome is Outcome.WORKED],
            is_returning_customer=is_returning,
            learned=learned,
            engine=self.engine.name,
        )

        return reply

    def observe(self, customer_id: str, message: str,
                session_id: str | None = None) -> list[Memory]:
        """Mine troubleshooting attempts out of free text and store them.

        A customer reporting their own history is the normal case:

            "I restarted it, cleared the cache, but updating the app fixed it"

        becomes three ATTEMPT memories with outcomes. Each clause is read for a
        step and an outcome; a step with no verdict nearby is stored as
        in_progress rather than dropped, so a pending attempt is not lost.

        Returns the memories written, so a caller can show what was understood.
        """
        session_id = session_id or new_id("ses")
        stored: list[Memory] = []
        seen: set[tuple[str, Outcome]] = set()

        for step_label, outcome in self._extract_observations(message):
            if (step_label, outcome) in seen:
                continue
            seen.add((step_label, outcome))
            stored.append(self.report_step(customer_id, step_label, outcome.value,
                                           session_id=session_id))
        return stored

    def _extract_observations(self, message: str) -> list[tuple[str, Outcome]]:
        """Read (step, outcome) pairs out of one message.

        Verdicts are attached by position, not per clause. A clause can hold
        both a failure and a success -- "cleared the cache, still crashing,
        updated the app and it fixed it" -- and a single verdict per clause
        makes the first one swallow the rest, recording a working step as a
        failure. So each step takes the nearest verdict that follows it,
        falling back to the nearest one before it, then to the next clause's
        verdict when its own clause says nothing ("...but neither helped").
        """
        pairs: list[tuple[str, Outcome]] = []
        clauses = _clauses(message)
        clause_verdicts = [self._verdict(c) for c in clauses]

        for index, clause in enumerate(clauses):
            steps = self._step_spans(clause)
            if not steps:
                continue
            marks = _verdict_spans(clause)
            for label, start, end in steps:
                verdict = Outcome.IN_PROGRESS
                # Nearest verdict after the step: "updated the app and it
                # fixed it" is judged by the fix, not the crash before it.
                later = [m for m in marks if m[0] >= end]
                if later:
                    verdict = min(later, key=lambda m: m[0])[2]
                else:
                    earlier = [m for m in marks if m[1] <= start]
                    if earlier:
                        verdict = max(earlier, key=lambda m: m[1])[2]
                    else:
                        verdict = clause_verdicts[index]
                        if (verdict is Outcome.IN_PROGRESS
                                and index + 1 < len(clauses)):
                            verdict = clause_verdicts[index + 1]
                pairs.append((label, verdict))

        # A step mentioned twice takes its most informative verdict, and a step
        # with no verdict anywhere stays open rather than being dropped.
        best: dict[str, Outcome] = {}
        order: list[str] = []
        for label, verdict in pairs:
            if label not in best:
                order.append(label)
            if verdict is not Outcome.IN_PROGRESS or label not in best:
                best[label] = verdict
        return [(label, best[label]) for label in order]

    @staticmethod
    def _step_spans(clause: str) -> list[tuple[str, int, int]]:
        """Each step a clause describes, with where it sits in the text."""
        matches: list[tuple[int, int, str]] = []
        for pattern, label in STEP_LEXICON:
            for match in re.finditer(pattern, clause, re.I):
                matches.append((match.start(), match.end(), label))
        return _resolve_spans(matches)

    @staticmethod
    def _step_labels(clause: str) -> list[str]:
        """Every step a clause describes, in the order they appear."""
        return [label for label, _, _ in SupportAgent._step_spans(clause)]

    def _step_label(self, clause: str) -> str | None:
        labels = self._step_labels(clause)
        return labels[0] if labels else None

    @staticmethod
    def _verdict(clause: str) -> Outcome:
        if NEGATED_WORKED.search(clause) or FAILED_PATTERNS.search(clause):
            return Outcome.FAILED
        if WORKED_PATTERNS.search(clause):
            return Outcome.WORKED
        return Outcome.IN_PROGRESS

    def report_step(self, customer_id: str, step: str, outcome: str,
                    symptom: str | None = None, product: str | None = None,
                    session_id: str | None = None) -> Memory:
        """Record a troubleshooting attempt and whether it worked.

        If no explicit symptom is given, inherit one from the conversation the
        step was taken in: the customer's most recent open incident, else the
        most recent incident of any outcome. Without this, "Update application"
        is unretrievable, because a generic step shares no words with the
        customer's description of the problem.
        """
        normalized = outcome.strip().lower()
        mapped = {
            "worked": Outcome.WORKED, "fixed": Outcome.WORKED, "success": Outcome.WORKED,
            "failed": Outcome.FAILED, "fail": Outcome.FAILED, "not_worked": Outcome.FAILED,
            "in_progress": Outcome.IN_PROGRESS, "trying": Outcome.IN_PROGRESS,
        }.get(normalized, Outcome.UNKNOWN)

        session_id = session_id or new_id("ses")
        terms = tokenize(symptom) if symptom else self._inherit_symptom(customer_id, session_id)
        if not terms:
            terms = tokenize(step)

        memory = Memory(
            customer_id=customer_id,
            kind=MemoryKind.ATTEMPT,
            text=step.strip().rstrip("."),
            outcome=mapped,
            product=product or self._latest_product(customer_id),
            symptom_terms=terms,
            session_id=session_id,
        )
        return self.store.add(memory)

    def _inherit_symptom(self, customer_id: str, session_id: str) -> list[str]:
        """Symptom terms from the incident this step belongs to."""
        incidents = [m for m in self.timeline(customer_id) if m.kind is MemoryKind.INCIDENT]
        if not incidents:
            return []
        same_session = [m for m in incidents if m.session_id == session_id]
        open_ones = [m for m in (same_session or incidents)
                     if m.outcome is Outcome.IN_PROGRESS]
        chosen = (open_ones or same_session or incidents)[-1]
        return list(chosen.symptom_terms)

    def _latest_product(self, customer_id: str) -> str | None:
        """The product this customer was last seen using, if any."""
        for memory in reversed(self.timeline(customer_id)):
            if memory.product:
                return memory.product
        return None

    def close_session(self, customer_id: str, summary: str, session_id: str | None = None,
                      product: str | None = None) -> Memory:
        """Consolidate a finished conversation into a durable incident memory."""
        outcome = Outcome.WORKED if WORKED_PATTERNS.search(summary) else Outcome.UNKNOWN
        memory = Memory(
            customer_id=customer_id,
            kind=MemoryKind.INCIDENT,
            text=summary.strip(),
            outcome=outcome,
            product=product,
            symptom_terms=tokenize(summary),
            session_id=session_id or new_id("ses"),
        )
        return self.store.add(memory)

    def set_profile(self, customer_id: str, fact: str) -> Memory:
        return self.store.add(Memory(
            customer_id=customer_id,
            kind=MemoryKind.PROFILE,
            text=fact.strip(),
            product=None,
            symptom_terms=tokenize(fact),
        ))

    def timeline(self, customer_id: str) -> list[Memory]:
        return sorted(self.store.all(customer_id), key=lambda m: m.created_at)

    def forget(self, customer_id: str) -> int:
        """Erase a customer's memory. A demo aid, and the honest answer to
        "forget I called" — the records are not just marked hidden."""
        return self.store.forget(customer_id)

    # ---------- internals ----------

    def _parse_context(self, message: str) -> CustomerContext:
        return CustomerContext(
            customer_id="",
            product=self._detect_product(message),
            symptoms=tokenize(message),
        )

    @staticmethod
    def _detect_product(message: str) -> str | None:
        """Find a product name like "Acme PDF Suite" in the message.

        Drops sentence-initial filler ("My", "The") that happens to be
        capitalised, and returns the last match, since that is usually the one
        closest to the complaint.
        """
        for match in PRODUCT_PATTERN.finditer(message):
            candidate = match.group(1).strip()
            while candidate.split()[0] in NOT_PRODUCT and len(candidate.split()) > 1:
                candidate = " ".join(candidate.split()[1:])
            if candidate and candidate.split()[0] not in NOT_PRODUCT:
                return candidate
        return None

    def _recall(self, customer_id: str, context: CustomerContext,
                history: list[Memory]) -> list[RetrievedMemory]:
        query_tokens = set(context.symptoms)
        recalled: list[RetrievedMemory] = []

        idf = self._idf(history)
        for memory in history:
            if memory.kind is MemoryKind.PROFILE:
                continue  # profile facts are not troubleshooting evidence
            haystack = set(tokenize(memory.text)) | set(memory.symptom_terms)
            overlap = query_tokens & haystack
            if not overlap:
                continue
            score = self._score(overlap, haystack, query_tokens, idf)
            reason = self._why(memory, overlap, context)
            recalled.append(RetrievedMemory(memory=memory, score=score, reason=reason))

        recalled.sort(key=lambda r: (r.score, r.memory.created_at), reverse=True)
        return recalled[:5]

    @staticmethod
    def _idf(history: list[Memory]) -> dict[str, float]:
        """Inverse document frequency over one customer's memories.

        Without this, a term every memory shares ("pdf") counts the same as one
        only a single memory has ("segfault").
        """
        total = max(len(history), 1)
        counts: dict[str, int] = {}
        for memory in history:
            for term in set(tokenize(memory.text)) | set(memory.symptom_terms):
                counts[term] = counts.get(term, 0) + 1
        return {term: math.log(1 + total / n) for term, n in counts.items()}

    @staticmethod
    def _score(overlap: set[str], haystack: set[str], query_tokens: set[str],
               idf: dict[str, float]) -> float:
        """Weighted overlap, normalized by both sides.

        Dividing by the query alone made long memories win; dividing by the
        memory alone made one-word memories win. Both terms are needed.
        """
        def weight(terms: set[str]) -> float:
            return sum(idf.get(t, 1.0) for t in terms)

        matched = weight(overlap)
        denominator = weight(query_tokens) + weight(haystack) - matched
        return matched / denominator if denominator > 0 else 0.0

    @staticmethod
    def _why(memory: Memory, overlap: set[str], context: CustomerContext) -> str:
        if context.product and memory.product == context.product:
            return f"same product ({memory.product}), matched on: {', '.join(sorted(overlap))}"
        return f"matched on: {', '.join(sorted(overlap))}"

    def _ingest(self, customer_id: str, message: str, context: CustomerContext,
                session_id: str) -> list[Memory]:
        """Store everything a message teaches us, and report what was learned.

        Order matters: the complaint is recorded first so that steps mined from
        the same message can inherit its symptom terms, and the attempts are
        written so they are recallable on the very next turn.
        """
        stored = self._remember_complaint(customer_id, message, context, session_id)
        stored += self.observe(customer_id, message, session_id=session_id)
        return stored

    def _remember_complaint(self, customer_id: str, message: str,
                            context: CustomerContext, session_id: str) -> list[Memory]:
        """Store a new complaint as an open incident, if this message is one.

        Two reasons not to: the message only reports what was already tried
        ("I restarted it and it didn't help"), or an incident is already open,
        so this is the same problem continuing rather than a new one.

        A message that *does* state a problem is stored even when it also
        reports attempts, because that is how customers actually write: one
        message naming the symptom and three things they already tried.
        """
        reports_step = any(self._step_label(c) for c in _clauses(message))
        states_problem = bool(SYMPTOM_HINT.search(message))
        already_open = any(
            m.kind is MemoryKind.INCIDENT and m.outcome is Outcome.IN_PROGRESS
            for m in self.timeline(customer_id)
        )

        stored: list[Memory] = []
        if not already_open and (states_problem or not reports_step):
            stored.append(self.store.add(Memory(
                customer_id=customer_id,
                kind=MemoryKind.INCIDENT,
                text=message.strip(),
                outcome=Outcome.IN_PROGRESS,
                product=context.product,
                symptom_terms=tokenize(message),
                session_id=session_id,
            )))

        if context.product and not any(
            m.product == context.product and m.kind is MemoryKind.PROFILE
            for m in self.store.all(customer_id)
        ):
            stored.append(self.set_profile(customer_id, f"Uses {context.product}"))
        return stored

    def _resolve_open_incident(self, customer_id: str) -> Memory | None:
        """Mark the open incident resolved once a step is reported as working."""
        open_incidents = [
            m for m in self.timeline(customer_id)
            if m.kind is MemoryKind.INCIDENT and m.outcome is Outcome.IN_PROGRESS
        ]
        if not open_incidents:
            return None
        return self.store.update(replace(open_incidents[-1], outcome=Outcome.WORKED))

    def detect_outcome(self, message: str) -> Outcome:
        """Read the outcome of a step from free text."""
        if WORKED_PATTERNS.search(message):
            return Outcome.WORKED
        if FAILED_PATTERNS.search(message):
            return Outcome.FAILED
        return Outcome.IN_PROGRESS


def _clauses(message: str) -> list[str]:
    return [c.strip() for c in CLAUSE_SPLIT.split(message) if c and c.strip()]


def _resolve_spans(matches: list[tuple[int, int, str]]) -> list[tuple[str, int, int]]:
    """Longest match wins at each position, overlaps dropped, order preserved.

    "reinstalled" must resolve to Reinstall once rather than matching both the
    reinstall and the install rules, and "cleared the application cache" must
    beat the bare "reset" nested inside it.
    """
    matches.sort(key=lambda m: (m[0], -(m[1] - m[0])))
    taken: list[tuple[int, int]] = []
    found: list[tuple[str, int, int]] = []
    for start, end, label in matches:
        if any(start < t_end and end > t_start for t_start, t_end in taken):
            continue
        taken.append((start, end))
        if label not in [label for label, _, _ in found]:
            found.append((label, start, end))
    return found


def _verdict_spans(clause: str) -> list[tuple[int, int, Outcome]]:
    """Every outcome signal in a clause, in text order.

    Position matters: "still crashing ... it fixed it" holds two opposite
    signals, and whichever step sits between them is judged by the nearer one.
    Negation is checked first so "nothing worked" is never read as a success.
    """
    marks: list[tuple[int, int, Outcome]] = []
    for match in NEGATED_WORKED.finditer(clause):
        marks.append((match.start(), match.end(), Outcome.FAILED))
    for match in FAILED_PATTERNS.finditer(clause):
        marks.append((match.start(), match.end(), Outcome.FAILED))
    for match in WORKED_PATTERNS.finditer(clause):
        if not any(start <= match.start() < end for start, end, _ in marks):
            marks.append((match.start(), match.end(), Outcome.WORKED))
    return sorted(marks)
