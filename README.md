# Memory-Enabled Customer Support Agent

A working prototype of the problem statement in
[`../memory_enabled_customer_support_agent_problem_statement.md`](../memory_enabled_customer_support_agent_problem_statement.md).

A support agent that **remembers what was tried and what worked** for each
customer, and uses that recall to skip dead ends and re-surface known fixes.

```text
NEW CUSTOMER                        RETURNING CUSTOMER (same issue)
-------------                       ---------------------------------
"crashes on large PDF"              "crashes again on PDF"
        |                                    |
        v                                    v
no memory on file                  recall: 2 failed, 1 worked
        |                                    |
        v                                    v
"try a restart"                    "skip restart + cache clear,
 (generic)                          update the app - that fixed it last time"
```

---

## Quick start

```bash
cd ~/Downloads/memory-support-agent
source .venv/bin/activate

python demo.py                       # guided narrative demo, no setup
python -m pytest tests -q            # 71 tests, all passing
uvicorn api:app --reload --port 8000 # REST backend + /docs
streamlit run ui.py                  # web UI
```

No API key needed. Everything runs on the deterministic scripted engine.

---

## Architecture

Four layers, each replaceable without touching the others.

```text
                 ┌─────────────────────────────────────────┐
   HTTP / UI ───▶│  LAYER 1  Interface      api.py, ui.py │
                 └────────────────┬────────────────────────┘
                                  │
                 ┌────────────────▼────────────────────────┐
                 │  LAYER 2  Orchestration                │
                 │  SupportAgent        memory_agent/    │
                 │  parse → recall → reason → remember    │
                 │  agent.py                             │
                 └───┬──────────────────────┬─────────────┘
                     │                      │
       ┌─────────────▼──────────┐  ┌────────▼───────────────┐
       │ LAYER 3  Memory store  │  │ LAYER 4  Reasoning      │
       │ store.py               │  │ engine.py               │
       │  InMemoryStore  local  │  │  ScriptedEngine  (demo) │
       │  HindsightStore remote │  │  LLMEngine       (real) │
       └────────────────────────┘  └────────────────────────┘
```

Data flow for one question:

```text
message
  │
  ├─▶ _parse_context    detect product + symptom tokens
  ├─▶ _recall           score every stored memory, keep top 5
  ├─▶ _ingest           mine attempts + complaints out of the text, store them
  ├─▶ engine.reply      answer using recall, split fixes vs. failures
  └─▶ _remember         write the ask back as a new memory
```

`_recall` runs *before* `_ingest`, on purpose. Mining first would let the agent
read a customer's own sentence back to them as history; ingesting after recall
means the steps mined from this message are not yet recallable, which is fine —
they will be on the next turn.

---

## Layer 1 — Interface

Two front ends over one agent instance. Pick either; both call the same methods.

### `api.py` — REST backend

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | store type, engine, known customers |
| `GET` | `/customers` | list customers with memory |
| `GET` | `/customers/{cid}/memories` | full memory timeline |
| `POST` | `/support/ask` | ask a question, get a memory-grounded answer |
| `POST` | `/support/step` | record a troubleshooting step + outcome (optional) |
| `POST` | `/support/session/close` | consolidate a conversation into an incident memory |
| `POST` | `/customers/{cid}/profile` | store a durable environment fact |
| `POST` | `/admin/reset` | wipe the local store (demo helper) |

`/support/step` exists for API clients and backfills, but it is **not** how the
demo works: `/support/ask` already mines steps out of free text, so a client
that sends nothing but a customer's message still gets memory. See "Learning
from the conversation" below.

Interactive schema at `http://localhost:8000/docs`.

`POST /support/ask` response shape:

```json
{
  "text": "Welcome back.\n\nFrom your previous sessions, this worked: ...",
  "engine": "scripted",
  "is_returning_customer": true,
  "known_fixes": ["Update application"],
  "known_failures": ["Restart application", "Cleared application cache"],
  "used_memories": [
    { "id": "mem_1a2b3c", "kind": "attempt", "outcome": "worked",
      "text": "Update application", "score": 0.67, "reason": "same product (Acme PDF Suite), matched on: application, pdf" }
  ],
  "learned": [
    { "id": "mem_4d5e6f", "kind": "attempt", "outcome": "failed",
      "text": "Restart application" }
  ]
}
```

`used_memories[].reason` is deliberate: it makes recall auditable on screen
instead of asking the audience to trust it.

### `ui.py` — Support Memory Console

One operator console, three surfaces:

- **Sidebar** — customer picker, `New conversation`, `Clear this customer's
  memory`, and a collapsed `Consolidate a session` for backfilling an incident by
  hand.
- **Conversation** — the live thread. Each answer carries the memories it drew
  on ("Memory used: …"), and each turn carries what was learned from that
  message ("Learned: Update application → worked"). Recall is auditable on
  screen instead of something the viewer has to take on trust.
- **Customer memory** — the stored record: totals, known fixes, known failures,
  then one card per memory badged by kind and outcome, newest first.

The environment strip under the title names the live memory store and reasoning
engine, so a demo failure is never mistaken for a memory bug.

**There is no step form.** A previous version had one, and it was the wrong
design: it required a human to transcribe each troubleshooting step into a
labelled field before memory could exist, which is exactly the work the system
is supposed to be doing. Memory now grows out of typing.

#### Manual test walkthrough

```bash
rm -f memory_store.json && streamlit run ui.py
```

| # | Action | Expected |
|---|---|---|
| 1 | Leave customer `CUST-1042`, panel reads "No memory yet" | empty state, no cards |
| 2 | Ask "My Acme PDF Suite crashes on large PDF upload" | generic reply, "no history on file" |
| 3 | Ask "I restarted it but that did not help" | learned caption names `Restart application → failed`, one new card |
| 4 | Ask "Cleared the cache, still crashing" | second attempt card |
| 5 | Ask "Updated the app and it fixed it" | "Good to hear that worked", third attempt card |
| 6 | Check the metrics row | 5 memories, 1 known fix, 2 known failures |
| 7 | **Clear the chat input**, re-ask the same question | "Welcome back", lists the fix, lists the two skipped steps, caption names the memories used |
| 8 | Switch customer to `CUST-9999` | panel resets to "No memory yet" |
| 9 | Ask the same question as `CUST-9999` | generic reply again — no leak |

Steps 7 and 9 are the ones that matter. 7 shows recall changing the answer; 9
shows recall is scoped to the customer rather than baked into the script.

Metrics count attempts only. An incident is a problem the customer had, and
counting a resolved one as a "known fix" made the row read as double.

**Reset between runs.** `memory_store.json` is read once at startup and the
agent is `@st.cache_resource`, so clearing the file alone leaves the old records
on screen. Use **Clear this customer's memory**, or stop the server, delete the
file, and start it again. Editing anything under `memory_agent/` also needs a
restart — Streamlit reruns `ui.py` on save but does not re-import the already
loaded core modules.

#### Automated UI tests

```bash
python -m pytest tests/test_ui.py -q
```

Seven tests drive the real widget tree through Streamlit's `AppTest` runner —
same code path as `streamlit run ui.py`, no browser. They cover the empty state,
the generic first answer, the conversation filling the memory panel, the learned
caption, the returning-customer reply, chat persistence across reruns, and
cross-customer isolation.

Two details worth knowing if you extend them: each test gets an isolated
`MEMORY_STORE_PATH` via `tmp_path`, and the fixture calls
`st.cache_resource.clear()` before building the `AppTest`, otherwise Streamlit
hands back the previous test's agent pointing at a deleted store file.

---

## Layer 2 — Orchestration: `memory_agent/agent.py`

`SupportAgent` owns the cycle. Four methods form the public surface:

| Method | Responsibility |
|---|---|
| `ask(cid, message)` | recall → reason → reply → store what was learned |
| `observe(cid, message)` | mine troubleshooting attempts out of free text and store them |
| `report_step(cid, step, outcome)` | record one attempt directly (API/backfill) |
| `close_session(cid, summary)` | consolidate a conversation into an incident memory |
| `set_profile(cid, fact)` | store a stable environment fact |

### Learning from the conversation

Nobody labels anything. The customer types what they tried, in whatever words
they use, and the agent works out what happened:

```text
"My Acme PDF Suite crashes on large PDF upload. I restarted it and cleared
 the cache but neither helped. Updating the app fixed it."

  → INCIDENT  in_progress  "My Acme PDF Suite crashes on large PDF upload. ..."
  → ATTEMPT   failed       Restart application
  → ATTEMPT   failed       Cleared application cache
  → ATTEMPT   worked       Update application
```

The message is split into clauses, and each clause is read twice: once for a
step name (`_step_labels`, matching a `STEP_LEXICON` of verbs like restart,
reinstall, clear cache, update) and once for a verdict (`_verdict`, over
`WORKED_PATTERNS`, `FAILED_PATTERNS` and `NEGATED_WORKED`). Details that turned
out to matter:

- **A verdict can govern earlier clauses.** "restarted it and cleared the cache
  but neither helped" — "neither helped" judges both steps. `_verdict` looks
  forward from a step, not only at the words beside it.
- **Negation beats the keyword.** "nothing worked" contains "worked". It is
  recorded as `FAILED`, because reading it as a fix is the worst possible error
  for a support agent to make.
- **A step with no verdict is not dropped.** It is stored `IN_PROGRESS`, so an
  attempt someone is still running survives the turn.
- **One step is recorded once.** "reinstalled" matches both the reinstall rule
  and the install rule; `_step_labels` resolves overlapping regex spans and keeps
  the longest match.

### What gets remembered, and what does not

This is the part that decides whether the system is a memory system or a
transcript archive. Four rules:

- A *step with an outcome* is always stored, whether it was mined from a message
  or passed to `report_step`.
- A *message that states a problem* is stored as an `INCIDENT` with outcome
  `in_progress`, so an unresolved issue is still recallable later. Naming a
  problem and a verdict in one message still logs it — that is exactly how
  customers write.
- A *message that only reports a step* ("I restarted it, no luck") is not a new
  incident. It is evidence about a step, and a second incident for the same
  problem would make the timeline a lie.
- A *question that matched a known fix* is **not** stored again. The answer is
  already in memory; writing it twice is noise. (`_remember_question`)

### The life of an incident

An incident is open until something closes it. `close_session` closes it, and so
does a step the customer reports as working — `_resolve_open_incident` flips it
to `WORKED`.

This is load-bearing. If the first incident stayed open forever, every later
complaint from the same customer would be swallowed as "the same problem still
open", and the agent would start attaching unrelated steps to it.

Resolution does **not** turn an incident into a fix. It stays an `INCIDENT` kind,
and the engine only ever lists `ATTEMPT` memories under "this worked" — quoting
a resolved problem back as a fix reads as nonsense, and was a real bug here.

**Recall filtering.** Two deliberate exclusions in `_recall`:

- `PROFILE` memories are excluded from the evidence list. "Runs macOS 15" is
  useful context, but it is not evidence that a fix worked, and mixing the two
  makes the "known fixes" list untrustworthy.
- Anything with zero symptom overlap is dropped. Relevance to *this* question is
  the whole point of contextual recall.

**Symptom inheritance.** The hard part of memory is that a stored step is short
and generic — "Update application" — while a customer's description of the
problem is specific — "it crashed when I uploaded a big PDF". They share almost
no words, so recall has to be given a bridge.

Two mechanisms:

- `report_step` inherits symptom terms from the incident the step belongs to:
  the customer's open incident, else the most recent one. A step recorded during
  a conversation inherits that conversation's complaint, so it becomes
  retrievable later without anyone labelling it. An explicit `symptom` argument
  always wins.
- `tokenize` normalizes and stems. "app"/"application"/"software" collapse to
  one term, and suffix stripping makes "crashes"/"crashing"/"crash" the same
  token — so the customer's own phrasing matches the stored step.

**Scoring** is IDF-weighted overlap, normalized by both sides:

```text
score = Σ idf(overlap) / (Σ idf(query) + Σ idf(memory) − Σ idf(overlap))
```

- **IDF**, computed over the customer's own memories, so a term every memory
  shares ("pdf") counts less than a rare one ("segfault").
- **Both sides in the denominator.** Dividing by the query alone let a long
  memory win on volume; dividing by the memory alone let a one-word memory win.
- Ties broken by recency, capped at 5.

Still lexical — a real Hindsight backend does this with embeddings. See "Swapping
in real Hindsight" below.

**Dedupe** (`engine._dedupe`) collapses repeated steps, because showing
"Restart application" twice reads as a bug. Matching is by 4-char stem set, so
"Restart application" and "restarted the app" collapse. Two distinct *steps* are
never merged, however similar: dropping a step means the agent may repeat
something the customer already ruled out. Five characters was tried and reverted
— at five, "restart" and "reinstall" collapse together.

**Outcome detection** (`detect_outcome`, plus the regexes at the top of the
file) reads free text: "that fixed it" → `WORKED`, "still crashing" → `FAILED`.
`close_session` uses this to decide whether an incident resolved.

---

## Layer 3 — Memory store: `memory_agent/store.py`

Both backends satisfy the same `MemoryStore` protocol — `add`, `update`, `all`,
`customers`, `search`. `build_store()` picks one at construction time.

### `InMemoryStore` (default)

JSON, grouped by customer, persisted to `memory_store.json` on every write.
Survives restarts, needs no services, and the file is readable — handy when you
want to show a judge the raw records. `update` exists for the one case that
mutates rather than appends: resolving an open incident.

### `HindsightStore` (set `HINDSIGHT_URL`)

HTTP adapter to a real Hindsight memory service. Same five methods, but recall
is delegated to the server's retriever instead of local lexical scoring, and
writes go to a named bank rather than a file.

```bash
export HINDSIGHT_URL=http://localhost:8765
export HINDSIGHT_BANK=support
```

Expected service surface (the endpoints the adapter calls):

| Endpoint | Body / query | Returns |
|---|---|---|
| `POST /memories` | `{bank, customer_id, memory}` | — |
| `PUT /memories/{id}` | `{bank, customer_id, memory}` | — |
| `GET /memories` | `?bank&customer_id` | `{memories: [...]}` |
| `GET /banks/{bank}/customers` | — | `{customers: [...]}` |
| `POST /recall` | `{bank, customer_id, query, limit}` | `{results: [{memory, score}]}` |

`/health` reports which backend is live, so a demo failure is never mistaken
for a memory bug.

**To connect a real Hindsight service:** match those five routes, and confirm
`Memory.from_dict` accepts its record shape. `models.py` is the only coupling
point. `SupportAgent` calls the protocol, never the implementation.

Both pluggable backends are tested without credentials — see
`tests/test_backends.py`.

---

## Layer 4 — Reasoning engine: `memory_agent/engine.py`

Engines receive the question, the ranked memories, a returning/new flag, and
what was just learned from this same message. They return text. They never touch
storage.

### `ScriptedEngine` (default)

Deterministic. Builds the answer from recall:

```
Welcome back.

From your previous sessions, this worked:
  - Update application
These already failed, so I'll skip them:
  - Restart application
  - Cleared application cache

Recommended next step: re-apply the fix above. ...
```

With no evidence it falls back to generic diagnostics — which is exactly what
makes the before/after contrast visible in a demo.

Two branches exist because a templated answer can be actively wrong:

- **No known fix.** "Every step I have on record for this failed, so I'd stop
  guessing and escalate." The generic "re-apply the fix above" with nothing
  listed above it is nonsense.
- **The message just reported a working step.** "Good to hear that worked. I've
  recorded Update application as the fix…". This branch exists because recall
  runs before ingest, so the step the customer just reported is not in the
  recalled set — and without it the agent answers a *solved* problem with
  "everything failed, I'll escalate."

Only `ATTEMPT` memories are ever listed as fixes or failures; `_dedupe` skips
incidents defensively as well as filtering in the caller.

### `LLMEngine` (set `ANTHROPIC_API_KEY`)

Same signature, real model call. The recalled history is formatted into the
prompt with its outcome label and relevance score, under a system prompt
instructing the model to avoid failed steps and re-surface known fixes. What was
just learned from the same message is passed separately, with an instruction not
to escalate a problem the customer has just said is resolved. The model is never
asked to guess what happened before — everything it knows about the customer
arrives through the memory layer.

Swapping engines changes nothing else in the system.

---

## Data model: `memory_agent/models.py`

```python
Memory(
    id, customer_id, session_id, created_at,
    kind      = ATTEMPT | INCIDENT | PROFILE | NOTE,
    text      = "Update application",
    outcome   = WORKED | FAILED | IN_PROGRESS | UNKNOWN,
    product   = "Acme PDF Suite",     # optional grouping key
    symptom_terms = ["large", "pdf", "upload", "crashes"],
)
```

`symptom_terms` is the retrieval bridge: the step text is short and generic
("Update application"), so symptom terms are what let a future message about
"large PDF upload" match it.

`kind` drives behaviour. `ATTEMPT` is the only kind that can be a fix or a
failure. `INCIDENT` is a problem the customer had and whether it is still open.
`PROFILE` is context. `NOTE` is a free-form escape hatch.

`SupportReply` is the response contract — reply text, the ranked `used_memories`,
split `known_fixes` / `known_failures`, `is_returning_customer`, and `learned`:
everything this message added to memory, so the UI can say what it picked up
instead of the customer wondering whether the agent was listening.

---

## Tests

71 total, no network, ~0.8s. `tests/test_agent.py` covers the agent,
`tests/test_backends.py` covers the two pluggable backends, `tests/test_ui.py`
covers the UI (see the Streamlit section above).

### `tests/test_agent.py` — 41 tests

The first five map one-to-one onto the problem statement's success criteria:

| # | Test | Proves |
|---|---|---|
| 1 | `test_1_stores_information_from_an_earlier_interaction` | memory is written |
| 2 | `test_2_recalls_relevant_memory_in_a_later_interaction` | memory is retrieved |
| 3 | `test_3_retrieved_memory_changes_the_response` | recall changes the answer |
| 4 | `test_4_distinguishes_failed_and_successful_attempts` | outcomes are separated |
| 5 | `test_5_returning_customer_gets_a_more_contextual_interaction` | new ≠ returning |

The rest cover the failure modes that make demos fall over:

- `test_memory_is_isolated_per_customer` — a hardcoded demo would leak CUST-1's
  fix to CUST-2. This is the single most important one.
- `test_unrelated_question_does_not_recall` — "billing address" must not pull in
  PDF crash history.
- `test_profile_facts_are_not_treated_as_troubleshooting_evidence`
- `test_recall_is_capped_and_ordered_by_relevance`
- `test_close_session_records_outcome_from_the_summary`
- `test_dedupe_drops_summary_that_restates_an_atomic_step`
- `test_store_persists_across_instances`

#### Learning from conversation, with no step form

Proving the agent reads plain English, since that is what replaced the form:

`test_agent_reads_attempts_out_of_a_message`,
`test_a_single_message_can_carry_several_attempts`,
`test_attempt_without_a_verdict_is_stored_as_in_progress`,
`test_step_reporting_is_not_also_recorded_as_a_complaint`,
`test_a_customer_does_not_see_their_own_message_as_history`,
`test_mined_attempt_is_recallable_on_the_next_turn`,
`test_plain_complaint_creates_no_attempt`,
`test_product_name_is_detected_without_sentence_filler`,
`test_verdict_in_a_later_clause_applies_to_all_steps`,
`test_a_step_named_twice_is_recorded_once`,
`test_negated_success_is_not_read_as_success`,
`test_full_one_shot_message_is_understood`.

#### Regression tests for defects found in review

The first twelve tests all passed while recall was still broken, because every
one of them supplied a hand-written `symptom`. These are named after what they
prevent from regressing:

| Test | Prevents |
|---|---|
| `test_recall_works_without_a_hand_labelled_symptom` | retrieval depending on a human typing the symptom |
| `test_step_inherits_symptom_from_its_session` | a step becoming unretrievable after it is recorded |
| `test_explicit_symptom_still_wins_over_inheritance` | inheritance overriding a deliberate label |
| `test_a_precise_memory_outranks_a_one_word_memory` | long memories winning on volume |
| `test_rarer_terms_outrank_terms_every_memory_shares` | ignoring IDF entirely |
| `test_profile_only_customer_is_not_treated_as_returning` | "Welcome back" with no relevant history |
| `test_no_known_fix_means_no_reapply_recommendation` | recommending a fix that was never listed |

The incident lifecycle went wrong three separate ways, and each has a test:

| Test | Prevents |
|---|---|
| `test_a_message_naming_the_problem_and_the_fix_still_logs_the_problem` | mined steps with no symptom to inherit, making them unsearchable |
| `test_a_reported_fix_is_acknowledged_rather_than_escalated` | answering a solved problem with "I'll escalate" |
| `test_a_solved_problem_is_closed_so_the_next_one_can_be_logged` | one open incident swallowing every later complaint |
| `test_a_resolved_problem_is_not_offered_back_as_a_fix` | quoting a resolved problem back under "this worked" |

A fourth defect, found by demoing rather than by testing: verdicts were read
per clause, so a message holding both a failure and a success in one clause gave
the first signal to every step. `"cleared the cache, still crashing, updated the
app and it fixed it"` stored the working update as a failure, and the agent then
told the customer every step had failed — including the one that fixed it. Each
step now takes the nearest outcome signal that follows it, so punctuation is not
required between a failure and a fix.

| Test | Prevents |
|---|---|
| `test_a_fix_after_a_failure_in_the_same_clause_is_not_swallowed` | a working step stored as a failure |
| `test_an_unpunctuated_rundown_keeps_the_fix_at_the_end` | the same, written the way a customer actually types |
| `test_a_step_between_two_signals_is_judged_by_the_nearer_one` | position ignored when signals straddle a step |

### `tests/test_ui.py` — 7 tests

Drives the real widget tree via Streamlit's `AppTest`. Empty state, generic
first answer, the conversation filling the memory panel with no form involved,
the learned caption, the returning-customer reply, chat persistence across
reruns, and cross-customer isolation.

### `tests/test_backends.py` — 23 tests

The two things this system cannot run in a demo are the only things that used to
be untested: a model call and a memory server. Both are now tested at their seam,
with the client and the HTTP transport injected.

- **LLMEngine** — `build_prompt` is a pure function, so the tests assert on what
  the model is actually told: recalled history carries its outcome label *and*
  its relevance score (without the label the model cannot tell a fix from a dead
  end), what was just learned is included (recall runs first, so the step the
  customer just reported is not in recall yet), and empty states render as
  `(no relevant history)` / `(nothing new)` rather than a blank block.
- **HindsightStore** — `httpx.MockTransport` records the request and returns a
  canned response, so the tests pin the wire contract: `POST /memories`,
  `GET /memories`, `PUT /memories/{id}`, `DELETE /memories`,
  `GET /banks/{bank}/customers`, `POST /recall`; the request bodies; and
  `Memory.from_dict` round-tripping the server's record shape.
- A **500 must raise.** A server error that returned an empty list would read as
  "this customer has no memory" and quietly change what the agent says.
- One **end-to-end** test runs a whole conversation against a canned Hindsight
  server, which is the actual claim of the `MemoryStore` protocol: swapping
  backends changes no agent code.
- **SDK drift is a real failure mode, and it had teeth.** Anthropic removed
  `temperature` from `messages.create` in SDK 1.9.0, so every live call raised
  `TypeError` — and no test caught it, because the other tests inject a fake
  client that accepts any kwarg. The engine now inspects the installed client's
  signature and sends the parameter only when it is accepted; one test pins a
  client that rejects it, one pins one that requires it, and one runs the whole
  suite's fixture against the real installed package.
- **An unreachable model must not take down the page.** A failed call —
  exhausted credits, network blip, rate limit — falls back to the scripted
  engine and records `degraded_reason`; the UI surfaces it in the environment
  strip rather than as a page-wide warning. A support console should degrade
  visibly and keep answering from memory, not crash on a billing error.

**What these do not prove:** that Anthropic or Hindsight actually accepts this
contract. A real integration test needs a real service and real credentials.
This is the honest limit — `HINDSIGHT_URL` and `ANTHROPIC_API_KEY` are
unverified against live systems until you set them.

To confirm the tests have teeth, break something and watch them fail:

```bash
perl -0pi -e 's|/recall|/search2|g' memory_agent/store.py
.venv/bin/python -m pytest tests/test_backends.py -q   # 1 failure
git checkout memory_agent/store.py                    # or restore your copy
```

```bash
python -m pytest tests -q                              # everything
python -m pytest tests/test_agent.py -q                # agent only
python -m pytest tests/test_backends.py -q             # model + memory-server seams
python -m pytest tests/test_ui.py -q                   # UI only
python -m pytest -k "returning or isolate" -v          # the two that matter
```

---

## Demo script

1. **`CUST-1042`, new customer**, same crash. Four turns: the complaint, then
   "I restarted it but that did not help", "Cleared the cache, still crashing",
   "Updated the app and it fixed it". Generic answer first, memory panel filling
   as it goes. Then the same customer returns with the crash again.
2. **`CUST-2077`, the whole story in one message.** "…crashes on large PDF
   upload. I restarted it and cleared the cache but neither helped. Updating the
   app fixed it." Then a return visit.
3. **`CUST-9999`, brand new, identical question.** Generic answer again — proving
   recall is per-customer, not a hardcoded response.

Customer 1 and customer 2 are the same problem written two different ways, which
is the point: memory is written by reading the conversation, not by being handed
a form. Customer 3 is what separates a memory system from a scripted one. Do not
skip it.

Nothing in `demo.py` passes a `symptom` or a `step` argument anywhere. Worth
leaving that way: it shows retrieval working on the customer's own words rather
than on a label you supplied.

---

## Known limitations

Worth stating before a judge does.

- **Lexical recall, not semantic.** Stemming and IDF fixed "uploaded" matching
  "upload", but "the file hangs on upload" still will not match "large PDF
  upload causes a crash" — no shared token survives. Real Hindsight embeddings
  fix this; the local store does not.
- **The real backends are unverified against live systems.** `LLMEngine` and
  `HindsightStore` are contract-tested against injected clients
  (`tests/test_backends.py`), which pins what goes over the wire and how
  responses are parsed — but no key and no Hindsight server have been run
  against. Set `ANTHROPIC_API_KEY` / `HINDSIGHT_URL` and exercise them before
  the demo; the Hindsight route shapes in particular are this repo's assumption,
  not a documented API.
- **Step extraction is a lexicon, not understanding.** `STEP_LEXICON` recognises
  restart, reinstall, clear cache, update, reinstall, factory reset and similar.
  "I nuked the preferences folder" is not in it, and that attempt is lost. A real
  deployment puts an LLM behind `observe()`; the scripted path exists to make
  the memory loop demonstrable without one.
- **Symptom inheritance assumes ordering.** A step inherits from the customer's
  open incident. Record steps before any complaint is logged, and the association
  is wrong. Passing an explicit `symptom` avoids it, which is why the parameter
  still exists.
- **Scripted replies are templated.** Honest about recall, thin on actual
  troubleshooting reasoning. `LLMEngine` is the real answer.
- **No memory decay or contradiction handling.** If a fix worked in March and
  regressed in June, both records persist and the agent has no way to prefer
  the newer one — a recency tiebreak exists in scoring, but nothing marks the
  old fix as stale.
- **Single-process memory.** The Streamlit and API apps each hold their own
  `InMemoryStore` instance against the same JSON file. Concurrent writes can
  clobber. A database or real Hindsight service fixes it.
- **No auth, no PII handling.** Customer IDs are bare strings.
