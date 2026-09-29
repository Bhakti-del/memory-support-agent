import pytest

from memory_agent import SupportAgent
from memory_agent.engine import ScriptedEngine, _dedupe
from memory_agent.models import Memory, MemoryKind, Outcome, RetrievedMemory
from memory_agent.store import InMemoryStore


@pytest.fixture
def agent() -> SupportAgent:
    return SupportAgent(store=InMemoryStore(), engine=ScriptedEngine())


def _recalled(text: str, kind: MemoryKind = MemoryKind.ATTEMPT) -> RetrievedMemory:
    return RetrievedMemory(
        memory=Memory(customer_id="CUST-1", kind=kind, text=text, outcome=Outcome.WORKED),
        score=1.0,
        reason="",
    )


def seed_resolved_session(agent: SupportAgent, customer_id: str = "CUST-1") -> None:
    agent.report_step(customer_id, "Restart application", "failed",
                      symptom="large PDF upload crashes application", product="Acme PDF Suite")
    agent.report_step(customer_id, "Cleared application cache", "failed",
                      symptom="large PDF upload crashes application", product="Acme PDF Suite")
    agent.report_step(customer_id, "Update application", "worked",
                      symptom="large PDF upload crashes application", product="Acme PDF Suite")


# --- success criteria from the problem statement -------------------------

def test_1_stores_information_from_an_earlier_interaction(agent):
    agent.ask("CUST-1", "My Acme PDF Suite crashes on large PDF upload")
    assert agent.timeline("CUST-1")


def test_2_recalls_relevant_memory_in_a_later_interaction(agent):
    seed_resolved_session(agent)
    reply = agent.ask("CUST-1", "The app crashes again when I upload a PDF")
    assert reply.used_memories


def test_3_retrieved_memory_changes_the_response(agent):
    fresh = agent.ask("CUST-2", "The app crashes again when I upload a PDF")
    seed_resolved_session(agent)
    returning = agent.ask("CUST-1", "The app crashes again when I upload a PDF")
    assert fresh.text != returning.text
    assert "Update application" in returning.text


def test_4_distinguishes_failed_and_successful_attempts(agent):
    seed_resolved_session(agent)
    reply = agent.ask("CUST-1", "The app crashes again when I upload a PDF")
    assert reply.known_fixes == ["Update application"]
    assert set(reply.known_failures) == {"Restart application", "Cleared application cache"}


def test_5_returning_customer_gets_a_more_contextual_interaction(agent):
    new_customer = agent.ask("CUST-9", "The app crashes again when I upload a PDF")
    seed_resolved_session(agent)
    returning = agent.ask("CUST-1", "The app crashes again when I upload a PDF")
    assert new_customer.is_returning_customer is False
    assert returning.is_returning_customer is True
    assert "Welcome back" in returning.text


# --- behaviour details ---------------------------------------------------

def test_memory_is_isolated_per_customer(agent):
    seed_resolved_session(agent, "CUST-1")
    other = agent.ask("CUST-2", "The app crashes again when I upload a PDF")
    assert other.used_memories == []
    assert other.is_returning_customer is False


def test_unrelated_question_does_not_recall(agent):
    seed_resolved_session(agent)
    reply = agent.ask("CUST-1", "How do I change my billing address?")
    assert not [m for m in reply.used_memories if m.memory.outcome is Outcome.WORKED]


def test_profile_facts_are_not_treated_as_troubleshooting_evidence(agent):
    agent.set_profile("CUST-1", "Runs macOS 15 on an M-series MacBook")
    agent.report_step("CUST-1", "Update application", "worked", symptom="crash on macOS")
    reply = agent.ask("CUST-1", "crash on macOS")
    kinds = {m.memory.kind for m in reply.used_memories}
    assert MemoryKind.PROFILE not in kinds


def test_close_session_records_outcome_from_the_summary(agent):
    memory = agent.close_session(
        "CUST-1", "Large PDF upload crashed the app. Updating the application fixed it."
    )
    assert memory.kind is MemoryKind.INCIDENT
    assert memory.outcome is Outcome.WORKED


def test_recall_is_capped_and_ordered_by_relevance(agent):
    for i in range(8):
        agent.report_step("CUST-1", f"Unrelated step {i}", "failed", symptom="printer offline")
    seed_resolved_session(agent)
    reply = agent.ask("CUST-1", "crashes when I upload a large PDF")
    assert len(reply.used_memories) <= 5
    scores = [m.score for m in reply.used_memories]
    assert scores == sorted(scores, reverse=True)


def test_dedupe_drops_summary_that_restates_an_atomic_step():
    summary = _recalled("Updating the application fixed it", MemoryKind.INCIDENT)
    assert _dedupe([_recalled("Update application"), summary]) == ["Update application"]


def test_dedupe_never_merges_two_distinct_steps():
    """Dropping a step means the agent may repeat something already ruled out."""
    steps = [_recalled("Restart application"),
             _recalled("Restart the application service"),
             _recalled("Reinstall application")]
    assert len(_dedupe(steps)) == 3


def test_store_persists_across_instances(tmp_path):
    path = tmp_path / "store.json"
    first = SupportAgent(store=InMemoryStore(path), engine=ScriptedEngine())
    first.report_step("CUST-1", "Update application", "worked", symptom="pdf crash")
    second = SupportAgent(store=InMemoryStore(path), engine=ScriptedEngine())
    assert len(second.store.all("CUST-1")) == 1


# --- regressions for the five defects found in review -------------------

def test_recall_works_without_a_hand_labelled_symptom(agent):
    """The defect: retrieval only worked when a human supplied symptom_terms.

    A customer says "it crashed when I uploaded a big PDF". The step recorded
    against that complaint is "Update application", which shares no words with
    the complaint. It must still be recalled.
    """
    agent.ask("CUST-1", "My Acme PDF Suite crashes whenever I upload a large PDF")
    agent.report_step("CUST-1", "Update application", "worked")
    agent.report_step("CUST-1", "Restart application", "failed")

    reply = agent.ask("CUST-1", "it crashed again when I uploaded a big PDF")
    assert "Update application" in reply.known_fixes
    assert "Restart application" in reply.known_failures


def test_step_inherits_symptom_from_its_session(agent):
    agent.ask("CUST-1", "The app hangs when I export to CSV", session_id="ses-1")
    step = agent.report_step("CUST-1", "Reinstall the export plugin", "worked",
                             session_id="ses-1")
    assert "export" in step.symptom_terms
    assert "csv" in step.symptom_terms


def test_explicit_symptom_still_wins_over_inheritance(agent):
    agent.ask("CUST-1", "The app hangs when I export to CSV", session_id="ses-1")
    step = agent.report_step("CUST-1", "Update application", "worked",
                             symptom="pdf upload crash", session_id="ses-1")
    assert "export" not in step.symptom_terms
    assert "pdf" in step.symptom_terms


def test_a_precise_memory_outranks_a_one_word_memory(agent):
    """The defect: scoring divided by the query alone, rewarding long memories."""
    agent.report_step("CUST-1", "PDF", "worked", symptom="pdf")
    agent.report_step("CUST-1",
                      "Update application to fix crash on upload of large PDF documents",
                      "failed", symptom="large pdf upload crash")
    ranked = agent.ask("CUST-1", "large pdf upload crash").used_memories
    assert "PDF" not in [r.memory.text for r in ranked[:1]]


def test_rarer_terms_outrank_terms_every_memory_shares(agent):
    """IDF: a term present in every memory should not score like a rare one."""
    agent.report_step("CUST-1", "Check the PDF export log", "failed",
                      symptom="pdf export fails")
    agent.report_step("CUST-1", "Free up disk space", "failed", symptom="pdf export fails")
    ranked = agent.ask("CUST-1", "pdf export fails with a segfault").used_memories
    assert ranked[0].memory.text == "Check the PDF export log"


def test_profile_only_customer_is_not_treated_as_returning(agent):
    """The defect: a profile row alone made the agent say 'Welcome back'."""
    agent.set_profile("CUST-1", "Runs macOS 15 on an M-series MacBook")
    reply = agent.ask("CUST-1", "How do I export to CSV?")
    assert reply.is_returning_customer is False
    assert "Welcome back" not in reply.text


def test_no_known_fix_means_no_reapply_recommendation(agent):
    """The defect: the reply said 're-apply the fix above' with no fix listed."""
    agent.report_step("CUST-1", "Restart application", "failed", symptom="pdf upload crash")
    agent.report_step("CUST-1", "Cleared cache", "failed", symptom="pdf upload crash")
    reply = agent.ask("CUST-1", "pdf upload crashes again")
    assert "re-apply the fix" not in reply.text
    assert "escalate" in reply.text


# --- learning from conversation, with no step form -----------------------

def test_agent_reads_attempts_out_of_a_message(agent):
    """The whole conversation, no form filling."""
    agent.ask("CUST-1", "My Acme PDF Suite crashes whenever I upload a large PDF")
    agent.ask("CUST-1", "I restarted it but that did not help.")
    agent.ask("CUST-1", "Cleared the cache, still crashing.")
    agent.ask("CUST-1", "Updated the app and it fixed it.")

    attempts = {m.text: m.outcome for m in agent.timeline("CUST-1")
                if m.kind is MemoryKind.ATTEMPT}
    assert attempts == {
        "Restart application": Outcome.FAILED,
        "Cleared application cache": Outcome.FAILED,
        "Update application": Outcome.WORKED,
    }


def test_a_single_message_can_carry_several_attempts(agent):
    agent.ask("CUST-1", "The app crashes on large PDF upload")
    agent.ask("CUST-1", "I tried a private window and then reinstalled, no luck so far")
    attempts = [m.text for m in agent.timeline("CUST-1") if m.kind is MemoryKind.ATTEMPT]
    assert attempts == ["Tried a private window", "Reinstall application"]


def test_attempt_without_a_verdict_is_stored_as_in_progress(agent):
    agent.ask("CUST-1", "The app crashes on large PDF upload")
    agent.ask("CUST-1", "I am reinstalling the application right now")
    attempt = [m for m in agent.timeline("CUST-1") if m.kind is MemoryKind.ATTEMPT][0]
    assert attempt.outcome is Outcome.IN_PROGRESS


def test_step_reporting_is_not_also_recorded_as_a_complaint(agent):
    """'I restarted it, no luck' is evidence about a step, not a new problem."""
    agent.ask("CUST-1", "The app crashes on large PDF upload")
    agent.ask("CUST-1", "I restarted it but that did not help")
    incidents = [m for m in agent.timeline("CUST-1") if m.kind is MemoryKind.INCIDENT]
    assert len(incidents) == 1


def test_a_customer_does_not_see_their_own_message_as_history(agent):
    """Regression: ingesting before recall made turn 1 read as 'Welcome back'."""
    reply = agent.ask("CUST-1", "The app crashes on large PDF upload")
    assert reply.is_returning_customer is False
    assert reply.used_memories == []


def test_reply_reports_what_was_learned(agent):
    reply = agent.ask("CUST-1", "The app crashes on large PDF upload. "
                                "I restarted it but that did not help.")
    learned = {m.text: m.outcome for m in reply.learned if m.kind is MemoryKind.ATTEMPT}
    assert learned == {"Restart application": Outcome.FAILED}


def test_mined_attempt_is_recallable_on_the_next_turn(agent):
    agent.ask("CUST-1", "The app crashes on large PDF upload")
    agent.ask("CUST-1", "I restarted it but that did not help")
    reply = agent.ask("CUST-1", "still crashing when I upload a large PDF")
    assert "Restart application" in reply.known_failures


def test_plain_complaint_creates_no_attempt(agent):
    agent.ask("CUST-1", "The app crashes whenever I upload a large PDF")
    assert [m for m in agent.timeline("CUST-1") if m.kind is MemoryKind.ATTEMPT] == []


def test_product_name_is_detected_without_sentence_filler(agent):
    agent.ask("CUST-1", "My Acme PDF Suite crashes whenever I upload a large PDF")
    profiles = [m.text for m in agent.timeline("CUST-1") if m.kind is MemoryKind.PROFILE]
    assert profiles == ["Uses Acme PDF Suite"]


def test_verdict_in_a_later_clause_applies_to_all_steps(agent):
    """'...but neither helped' judges every step named before it."""
    observations = agent._extract_observations(
        "I restarted it and cleared the cache but neither helped"
    )
    assert observations == [("Restart application", Outcome.FAILED),
                            ("Cleared application cache", Outcome.FAILED)]


def test_a_step_named_twice_is_recorded_once(agent):
    """'reinstalled' must not match both the reinstall and the install rules."""
    observations = agent._extract_observations("I reinstalled the app, no luck")
    assert observations == [("Reinstall application", Outcome.FAILED)]


def test_negated_success_is_not_read_as_success(agent):
    """'nothing worked' contains 'worked'; it must not become a known fix."""
    assert agent._verdict("nothing worked") is Outcome.FAILED
    assert agent._verdict("it worked") is Outcome.WORKED
    assert agent._verdict("the problem persists") is Outcome.FAILED


def test_full_one_shot_message_is_understood(agent):
    observations = agent._extract_observations(
        "My Acme PDF Suite crashes on large PDF upload. I restarted it and cleared "
        "the cache but neither helped. Updating the app fixed it."
    )
    assert observations == [
        ("Restart application", Outcome.FAILED),
        ("Cleared application cache", Outcome.FAILED),
        ("Update application", Outcome.WORKED),
    ]


# --- incidents: the problem, kept separate from the steps ------------------

def test_a_message_naming_the_problem_and_the_fix_still_logs_the_problem(agent):
    """The defect: naming a symptom and a verdict in one message skipped the
    incident, so the mined attempts had no symptom terms to inherit and were
    unsearchable. Customers write exactly this kind of message."""
    reply = agent.ask(
        "CUST-1",
        "My Acme PDF Suite crashes on large PDF upload. I restarted it but that "
        "did not help. Clearing the cache fixed it.",
    )
    incident = next(m for m in reply.learned if m.kind is MemoryKind.INCIDENT)
    attempts = [m for m in reply.learned if m.kind is MemoryKind.ATTEMPT]
    assert {m.text: m.outcome for m in attempts} == {
        "Restart application": Outcome.FAILED,
        "Cleared application cache": Outcome.WORKED,
    }
    # The steps inherit the symptom, so the next visit can find them.
    for attempt in attempts:
        assert set(incident.symptom_terms) <= set(attempt.symptom_terms)


def test_a_step_report_alone_is_not_a_new_problem(agent):
    agent.ask("CUST-1", "The app crashes on large PDF upload")
    reply = agent.ask("CUST-1", "I restarted it but that did not help")
    assert [m for m in reply.learned if m.kind is MemoryKind.INCIDENT] == []


def test_a_reported_fix_is_acknowledged_rather_than_escalated(agent):
    """The defect: recall runs before ingest, so a step reported as working is
    not yet in the recalled set -- and the agent answered a solved problem
    with 'every step failed, so I'd stop guessing and escalate'."""
    agent.ask("CUST-1", "The app crashes on large PDF upload")
    agent.ask("CUST-1", "I restarted it but that did not help")
    agent.ask("CUST-1", "Cleared the cache, still crashing")
    reply = agent.ask("CUST-1", "Updated the app and it fixed it")
    assert "escalate" not in reply.text
    assert "Update application" in reply.text


def test_a_solved_problem_is_closed_so_the_next_one_can_be_logged(agent):
    """Otherwise the first incident stays open forever and every later
    complaint from the same customer is swallowed as the same problem."""
    agent.ask("CUST-1", "The app crashes on large PDF upload")
    agent.ask("CUST-1", "Updated the app and it fixed it")
    agent.ask("CUST-1", "Now the export button has disappeared")
    incidents = [m for m in agent.timeline("CUST-1") if m.kind is MemoryKind.INCIDENT]
    assert [m.outcome for m in incidents] == [Outcome.WORKED, Outcome.IN_PROGRESS]


def test_a_resolved_problem_is_not_offered_back_as_a_fix(agent):
    """The defect: resolving an incident left it in the recalled set with a
    WORKED outcome, so the reply listed the customer's problem text under
    'this worked'."""
    agent.ask("CUST-1", "The app crashes on large PDF upload")
    agent.ask("CUST-1", "Updated the app and it fixed it")
    reply = agent.ask("CUST-1", "The app is crashing again on large PDF upload")
    assert reply.known_fixes == ["Update application"]
    assert "crashes on large PDF upload" not in reply.text
