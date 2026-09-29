"""UI smoke tests using Streamlit's AppTest runner (no browser needed).

    python -m pytest tests/test_ui.py -q

These exercise the same code path `streamlit run ui.py` does, so they catch
breakage in the UI layer itself, not just the agent underneath it. Nothing here
fills a form: the agent learns from what the user types, so that is what the
tests do.
"""

from __future__ import annotations

from pathlib import Path

import pytest

UI_PATH = Path(__file__).resolve().parent.parent / "ui.py"

CONVERSATION = [
    "My Acme PDF Suite crashes whenever I upload a large PDF",
    "I restarted it but that did not help.",
    "Cleared the cache, still crashing.",
    "Updated the app and it fixed it.",
]


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Fresh UI with an isolated memory store, cleared on every run."""
    from streamlit.testing.v1 import AppTest

    store = tmp_path / "store.json"
    monkeypatch.setenv("MEMORY_STORE_PATH", str(store))

    def _make():
        # cache_resource would otherwise hand back the previous test's agent,
        # pointing at a store file that no longer exists.
        import streamlit as st

        st.cache_resource.clear()
        return AppTest.from_file(str(UI_PATH), default_timeout=20).run()

    return _make


def _say(app, customer: str, *messages: str) -> None:
    """Type messages into the chat box as one customer."""
    at = app()
    at.text_input[0].set_value(customer).run()
    for message in messages:
        at.chat_input[0].set_value(message).run()
    return at


def _chat_text(at) -> str:
    """All text rendered inside chat bubbles (body markdown + inline captions)."""
    parts: list[str] = []
    for message in at.chat_message:
        parts += [m.value for m in message.markdown]
        parts += [c.value for c in message.caption]
    return "\n".join(parts)


def _panel_text(at) -> str:
    # st.write of a string is a markdown element in AppTest, so memory bodies
    # are already covered by at.markdown.
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def test_ui_renders_for_a_new_customer(app):
    at = app()
    assert not at.exception
    assert at.title[0].value == "Support Memory Console"
    # No memory yet, so the panel should say so rather than render empty cards.
    assert any("No memory yet" in i.value for i in at.info)


def test_asking_with_no_memory_gets_a_generic_answer(app):
    at = _say(app, "CUST-1042", CONVERSATION[0])
    assert "no history on file" in _chat_text(at)


def test_conversation_fills_the_memory_panel_with_no_forms(app):
    at = _say(app, "CUST-1042", *CONVERSATION)
    assert not at.exception
    # 3 attempts + 1 incident + 1 product profile
    assert at.metric[0].value == "5"
    assert at.metric[1].value == "1"          # known fixes
    assert at.metric[2].value == "2"          # known failures
    panel = _panel_text(at)
    assert "Update application" in panel
    assert "Failed" in panel
    assert "Worked" in panel


def test_each_reply_reports_what_it_learned(app):
    at = _say(app, "CUST-1042", CONVERSATION[0], "Updated the app and it fixed it.")
    text = _chat_text(at)
    assert "Learned:" in text
    assert "Update application" in text


def test_returning_customer_sees_recalled_memory_in_the_chat(app):
    at = _say(app, "CUST-1042", *CONVERSATION)
    at.chat_input[0].set_value("The app crashes again when I upload a PDF").run()
    assert not at.exception
    text = _chat_text(at)
    assert "Welcome back" in text
    assert "Update application" in text
    assert "Memory used" in text


def test_chat_history_persists_across_reruns(app):
    at = app()
    at.chat_input[0].set_value("first question about a pdf crash").run()
    at.chat_input[0].set_value("second question about a pdf crash").run()
    assert not at.exception
    assert len(at.chat_message) == 4          # 2 user + 2 assistant


def test_memory_does_not_leak_between_customers(app):
    # One AppTest instance throughout: a fresh one rebuilds the agent, which
    # would discard the first customer's memory along with the test's intent.
    at = app()
    at.text_input[0].set_value("CUST-1042").run()
    for message in CONVERSATION:
        at.chat_input[0].set_value(message).run()

    at.text_input[0].set_value("CUST-9999").run()
    assert any("No memory yet" in i.value for i in at.info)

    at.chat_input[0].set_value("The app crashes again when I upload a PDF").run()
    assert "no history on file" in _chat_text(at)
