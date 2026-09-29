"""Guided demo: two customers, one conversation each, no forms.

Everything the agent knows is learned from what the customer types. Nothing is
passed to report_step — there is no step form in this demo, and none in the UI
either.
"""

from __future__ import annotations

from memory_agent import SupportAgent
from memory_agent.models import MemoryKind, Outcome

BAR = "-" * 72

# The first customer talks through the whole problem, then reports the outcome.
FIRST_VISIT = [
    "My Acme PDF Suite crashes whenever I upload a large PDF.",
    "I restarted it but that did not help.",
    "Cleared the cache, still crashing.",
    "Updated the app and it fixed it.",
]

# The second customer states the same history in one message.
FIRST_VISIT_ONE_SHOT = [
    "My Acme PDF Suite crashes on large PDF upload. I restarted it and cleared "
    "the cache but neither helped. Updating the app fixed it.",
]

RETURN_VISIT = "The application is crashing again when I upload a PDF."


def converse(agent: SupportAgent, customer_id: str, messages: list[str],
             label: str) -> None:
    print(f"\n{'=' * 72}\nCUSTOMER: {customer_id}   ({label})\n{'=' * 72}")
    for message in messages:
        print(f"\nCustomer: {message}")
        reply = agent.ask(customer_id, message)
        print(f"Agent: {reply.text}")
        learned = [f"{m.text} -> {m.outcome.value}"
                   for m in reply.learned if m.kind is MemoryKind.ATTEMPT]
        if learned:
            print(f"  [learned from that message: {'; '.join(learned)}]")


def show_recall(agent: SupportAgent, customer_id: str) -> None:
    print(f"\nCustomer: {RETURN_VISIT}")
    reply = agent.ask(customer_id, RETURN_VISIT)
    print(f"Agent: {reply.text}")
    print("\n-- memory used --")
    for used in reply.used_memories:
        print(f"  [{used.memory.outcome.value:<11}] score={used.score:.2f}  "
              f"{used.memory.text[:56]}   ({used.reason})")


def main() -> None:
    agent = SupportAgent()

    converse(agent, "CUST-1042", FIRST_VISIT, "new, reports attempts over 4 turns")
    show_recall(agent, "CUST-1042")

    converse(agent, "CUST-2077", FIRST_VISIT_ONE_SHOT, "new, reports it all at once")
    show_recall(agent, "CUST-2077")

    converse(agent, "CUST-9999", [RETURN_VISIT], "brand new, same complaint")
    print("\n(no recall above: nothing was ever stored for this customer)")

    print(f"\n{BAR}\nSTATS\n{BAR}")
    for cid in agent.store.customers():
        memories = agent.timeline(cid)
        # Only attempts are fixes or failures; an incident is a problem, even
        # after it has been resolved.
        attempts = [m for m in memories if m.kind is MemoryKind.ATTEMPT]
        worked = sum(1 for m in attempts if m.outcome is Outcome.WORKED)
        failed = sum(1 for m in attempts if m.outcome is Outcome.FAILED)
        print(f"{cid}: {len(memories)} memories | {len(attempts)} attempts | "
              f"{worked} fixes | {failed} failures")


if __name__ == "__main__":
    main()
