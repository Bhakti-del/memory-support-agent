"""Streamlit UI: an operator console for the memory-enabled support agent.

Run with:  streamlit run ui.py

The product surface is the three-column console: a customer, a live conversation,
and the memory that conversation produced. Recall is not a black box here — every
answer shows the memories it drew on, and every turn shows what was learned, so
an operator can audit the loop instead of trusting it.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from memory_agent import SupportAgent
from memory_agent.engine import build_engine
from memory_agent.models import MemoryKind, Outcome
from memory_agent.store import HindsightStore, InMemoryStore, build_store

st.set_page_config(page_title="Support Memory Console", layout="wide")

STORE_PATH = Path(os.getenv("MEMORY_STORE_PATH", "memory_store.json"))

KIND_LABEL = {
    MemoryKind.ATTEMPT: "Attempt",
    MemoryKind.INCIDENT: "Incident",
    MemoryKind.PROFILE: "Profile",
    MemoryKind.NOTE: "Note",
}

OUTCOME_LABEL = {
    Outcome.WORKED: "Worked",
    Outcome.FAILED: "Failed",
    Outcome.IN_PROGRESS: "In progress",
    Outcome.UNKNOWN: "Unknown",
}

# Muted, professional status colors. Not the saturated defaults.
OUTCOME_COLOR = {
    Outcome.WORKED: "#15803d",
    Outcome.FAILED: "#b91c1c",
    Outcome.IN_PROGRESS: "#b45309",
    Outcome.UNKNOWN: "#6b7280",
}

CSS = """
<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }
  [data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px;
    padding: 0.7rem 0.9rem;
  }
  [data-testid="stMetricLabel"] p { font-size: 0.72rem; letter-spacing: 0.04em;
    text-transform: uppercase; color: #6b7280; }
  [data-testid="stMetricValue"] { font-size: 1.5rem; color: #111827; }
  .console-title { font-weight: 650; letter-spacing: -0.01em; }
  .console-sub { color: #6b7280; font-size: 0.95rem; }
  .env-strip { color: #4b5563; font-size: 0.82rem; }
  .mem-kind { font-weight: 600; color: #374151; font-size: 0.82rem;
    text-transform: uppercase; letter-spacing: 0.05em; }
  .mem-id { color: #9ca3af; font-size: 0.72rem; }
  [data-testid="stChatMessage"] { border: 1px solid #e5e7eb; border-radius: 10px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def get_agent() -> SupportAgent:
    # build_store is the one place that decides which backend is live, and it
    # wraps Hindsight so an unreachable instance degrades to local JSON.
    store = build_store(STORE_PATH)
    return SupportAgent(store=store, engine=build_engine())


def _badge(outcome: Outcome) -> str:
    color = OUTCOME_COLOR[outcome]
    return (
        f'<span style="color:{color};font-weight:600;">{OUTCOME_LABEL[outcome]}</span>'
    )


agent = get_agent()

# ------------------------------------------------------------------ header
st.title("Support Memory Console")
st.markdown(
    '<div class="console-sub">Memory-enabled customer support · every answer '
    "traced to the memories behind it</div>",
    unsafe_allow_html=True,
)
store_name = "Hindsight" if os.getenv("HINDSIGHT_URL") else "Local JSON"
# A backend that is configured but not answering is a configuration problem,
# not a memory problem, so it is surfaced in the environment strip rather than
# as a page-wide warning. A console should not greet every operator with a
# stack-trace-shaped banner just because an optional backend is misconfigured.
if getattr(agent.engine, "degraded_reason", None):
    engine_name = f"{agent.engine.name} (unavailable, using fallback)"
else:
    engine_name = agent.engine.name
if getattr(agent.store, "degraded_reason", None):
    store_name = "Hindsight (unavailable, using local)"
try:
    customer_count = len(agent.store.customers())
except Exception:  # noqa: BLE001 - the header must never be what breaks the page
    customer_count = 0
st.markdown(
    f'<div class="env-strip">Memory store <b>{store_name}</b> &nbsp;·&nbsp; '
    f'Reasoning engine <b>{engine_name}</b> &nbsp;·&nbsp; '
    f'Customers on file <b>{customer_count}</b></div>',
    unsafe_allow_html=True,
)

st.divider()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Customer")
    customer_id = st.text_input("Customer ID", value="CUST-1042", label_visibility="collapsed")
    known = agent.store.customers()
    if known:
        st.caption("On file: " + ", ".join(known))
    else:
        st.caption("No customers have memory yet.")

    st.divider()
    st.header("Session")
    if st.button("New conversation", use_container_width=True):
        st.session_state["chat"] = []
        st.rerun()
    if st.button("Clear this customer's memory", use_container_width=True):
        agent.forget(customer_id)
        st.session_state["chat"] = []
        st.rerun()

    st.divider()
    with st.expander("Consolidate a session"):
        summary_text = st.text_area(
            "Summary", placeholder="Large PDF upload caused a crash…",
            label_visibility="collapsed",
        )
        if st.button("Save as incident", use_container_width=True) and summary_text.strip():
            agent.close_session(customer_id, summary_text)
            st.rerun()

# ------------------------------------------------------------------ console
left, right = st.columns([3, 2])

with left:
    st.subheader("Conversation")
    chat = st.session_state.setdefault("chat", [])
    if not chat:
        st.info(
            "No conversation yet. Describe the problem or what has already been "
            "tried — the agent learns from the message and remembers it."
        )
    for turn in chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["text"])
            if turn.get("learned"):
                st.caption("Learned: " + ", ".join(turn["learned"]))
            elif turn.get("memories"):
                st.caption(
                    "Memory used: " + ", ".join(m["text"] for m in turn["memories"])
                )

    prompt = st.chat_input("Describe the problem, or what you already tried…")
    if prompt:
        reply = agent.ask(customer_id, prompt)
        st.session_state.chat += [
            {"role": "user", "text": prompt},
            {
                "role": "assistant",
                "text": reply.text,
                "learned": [
                    f"{m.text} → {m.outcome.value}"
                    for m in reply.learned
                    if m.kind is MemoryKind.ATTEMPT
                ],
                "memories": [
                    {"text": m.memory.text, "outcome": m.memory.outcome.value}
                    for m in reply.used_memories
                ],
            },
        ]
        st.rerun()

with right:
    st.subheader("Customer memory")
    memories = agent.timeline(customer_id)
    if not memories:
        st.info("No memory yet for this customer. Their first session starts from zero.")
    else:
        # Only attempts count as fixes or failures. An incident is a problem
        # the customer had, not a fix, even once it is resolved.
        attempts = [m for m in memories if m.kind is MemoryKind.ATTEMPT]
        st.metric("Memories", len(memories))
        c1, c2 = st.columns(2)
        c1.metric("Known fixes", sum(1 for m in attempts if m.outcome is Outcome.WORKED))
        c2.metric("Known failures", sum(1 for m in attempts if m.outcome is Outcome.FAILED))
        st.divider()
        for m in reversed(memories):
            # A profile fact or note has no outcome. Printing "Unknown" beside
            # it reads as a data defect rather than as a stable fact.
            if m.kind in (MemoryKind.PROFILE, MemoryKind.NOTE):
                header = KIND_LABEL[m.kind]
            else:
                header = f"{KIND_LABEL[m.kind]} · {_badge(m.outcome)}"
            with st.container(border=True):
                st.markdown(
                    f'<div class="mem-kind">{header}</div>', unsafe_allow_html=True
                )
                st.write(m.text)
                st.markdown(
                    f'<div class="mem-id">{m.created_at} · {m.id}</div>',
                    unsafe_allow_html=True,
                )
