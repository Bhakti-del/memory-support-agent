# Building a Memory-Enabled Customer Support Agent with Hindsight

Every time you contact customer support, there is a good chance the agent you're talking to has no idea you called last week.

You explain the same problem. You go through the same steps. You wait through the same troubleshooting process. And eventually, you're told to try restarting the application — the exact thing you already said didn't work.

**This isn't a people problem. It's a memory problem.**

## The Problem with Stateless Customer Support

Most support systems are effectively stateless from the perspective of the agent handling the current conversation. Even when a customer has a long history with a product, the useful information may be buried across previous tickets, inconsistent notes, or long conversation histories.

This creates a familiar pattern:

- **Repeated steps:** The customer is asked to restart an application even though they already tried it.
- **Lost context:** A solution that worked previously isn't immediately available to the next interaction.
- **Escalation dead ends:** A problem gets escalated and the next agent starts the diagnosis again.

The obvious cost is customer frustration, but there is another cost: **duplicated agent work**.

When an agent has to re-diagnose a problem that has already been investigated for the same customer, time is spent repeating work instead of progressing toward a solution.

The instinctive solution is often to store more data: longer ticket histories, better CRM notes, or larger summaries.

But storage alone doesn't solve the problem.

The important question is:

> **Can the system retrieve the right information at the moment the agent needs it?**

What is actually needed is a system that can automatically identify:

- what the customer previously tried,
- what failed,
- what worked,
- what is still unresolved,
- and information about the customer's environment.

That information needs to be available **before the agent generates its next response**.

This is the gap our project addresses using **structured, per-customer agent memory powered by Hindsight**.

---

# What Do We Mean by Customer Memory?

Memory in an AI agent isn't simply a copy of the previous conversation.

Our system represents memory as **structured, retrievable facts associated with a specific customer**.

Instead of storing a large raw chat transcript, the system extracts information that can directly affect future support decisions.

We use three main types of memory:

### ATTEMPT

A troubleshooting action together with its outcome.

```text
Cleared application cache → failed
Update application → worked
```

### INCIDENT

The problem the customer is experiencing and whether it has been resolved.

```text
App crashes on large PDF upload → resolved
```

### PROFILE

Stable information about the customer's environment.

```text
Uses Acme PDF Suite
```

These memories are extracted automatically from what the customer says rather than requiring a support agent to manually enter them.

This gives the system actionable information instead of simply giving it another conversation summary.

---

# Why Structured Memory Beats Summaries

A free-text summary might tell an agent:

> "The customer previously experienced crashes and tried several troubleshooting steps."

But that doesn't directly answer:

> **Which steps should we skip, and which one should we try first?**

Structured memory does.

For example:

```text
[failed] Restart application
[failed] Cleared application cache
[worked] Update application
```

Now the agent can immediately reason from previous outcomes.

Without memory, it might say:

> "Let's start with a restart. If that doesn't work, try clearing the cache."

With memory, it can say:

> "From your previous sessions, updating the application fixed this. I'll skip restart and cache clearing because those already failed."

The difference is not simply that the agent remembers the customer.

**It remembers what happened and uses that history to change its next action.**

---

# Why Memory Must Be Per-Customer

A global knowledge base of troubleshooting solutions is useful, but it is not the same thing as customer memory.

Consider two customers experiencing the same PDF crash.

One customer's problem might be fixed by updating the application.

Another customer's problem might require a reinstall.

If their histories are mixed together, the system could retrieve the wrong experience.

That's why our memory store is keyed by `customer_id`.

Every memory retrieval, write, and recall is scoped to the individual customer.

The architecture therefore looks conceptually like:

```text
Customer
   |
   v
customer_id
   |
   v
Customer-specific memories
   |
   +-- Attempts
   +-- Incidents
   +-- Profile
```

This isolation is fundamental to the design rather than an optional feature.

---

# A Real Example: CUST-1042

Let's see what this looks like in an actual support interaction.

Customer:

```text
Customer ID: CUST-1042
Product: Acme PDF Suite
Problem: Crashes on large PDF uploads
```

During the first support session, the customer says:

```text
My Acme PDF Suite crashes whenever I upload a large PDF.

I restarted it but that did not help.

Cleared the cache, still crashing.

Updated the app and it fixed it.
```

The system extracts:

```text
[failed]      Restart application
[failed]      Cleared application cache
[worked]      Update application
[resolved]    App crashes on large PDF upload
[profile]     Uses Acme PDF Suite
```

Later, the same customer returns:

```text
The application is crashing again when I upload a PDF.
```

## Without Memory

A stateless agent has no useful history available.

It might respond:

> "I've logged this issue for you. Let's confirm the affected version and try a clean restart."

The customer now has to explain:

> "I already tried that."

The agent asks what else they tried.

The customer explains the cache-clearing attempt.

Eventually, the conversation reaches the update that previously fixed the problem.

The customer has had to repeat information the system could have used immediately.

## With Memory

The same message arrives with `CUST-1042`'s history available.

The recall system finds:

```text
[worked]  score=0.71  Update application
[failed]  score=0.68  Restart application
[failed]  score=0.65  Cleared application cache
```

The agent can respond:

> "Welcome back.
>
> From your previous sessions, updating the application fixed this.
>
> Restarting the application and clearing the cache had already failed, so I'll skip those steps.
>
> Let's try the previous fix again. If the issue has returned after the update, we can escalate it with your previous session context."

The system has avoided re-diagnosing the same problem and surfaced the previously successful action immediately.

---

# The Memory Recall Loop

Storing memory is only half of the problem.

The other half is retrieving it at the correct moment.

Our system follows this sequence for every customer message:

```text
1. Parse the customer's message
              ↓
2. Retrieve relevant memories
              ↓
3. Pass memories to the reasoning engine
              ↓
4. Generate the response
              ↓
5. Extract new memories from the message
              ↓
6. Store the new memories
```

The ordering is important.

**Recall happens before ingesting the current message.**

If the system stores the current message first and then performs recall, the current message can appear in its own retrieval results.

That creates a subtle bug where something the customer has just said could incorrectly be treated as historical information.

So the architecture deliberately separates:

```text
Past memory
     ↓
Recall
     ↓
Current response
     ↓
New memory
```

rather than mixing the current message into the history before retrieval.

---

# How Hindsight Gives the Agent Long-Term Memory

A local in-process memory store is useful for development, demos, and testing.

But a production support agent has different requirements.

Customers may return days or weeks later. The application may restart. Multiple processes may need access to the same customer history.

This is where **Hindsight** becomes the persistent memory backend.

Instead of coupling the support agent directly to Hindsight, we created a common `MemoryStore` protocol:

```python
class MemoryStore(Protocol):
    def add(self, memory: Memory) -> Memory: ...
    def update(self, memory: Memory) -> Memory: ...
    def all(self, customer_id: str) -> list[Memory]: ...
    def forget(self, customer_id: str) -> int: ...
    def search(
        self,
        customer_id: str,
        query: str,
        limit: int = 5
    ) -> list[tuple[Memory, float]]: ...
```

The support agent only knows about this interface.

It doesn't need to know whether the underlying implementation is local storage or Hindsight.

---

# InMemoryStore vs. HindsightStore

For development and testing, we use `InMemoryStore`.

It stores memories locally and can optionally persist them to a JSON file.

For production, `HindsightStore` communicates with a Hindsight instance over HTTP.

Conceptually:

```text
                 Support Agent
                       |
                       v
                  MemoryStore
                 /           \
                /             \
               v               v
       InMemoryStore      HindsightStore
               |               |
               v               v
        Local storage       Hindsight
```

The Hindsight implementation sends customer-specific memory information to the service and uses the recall endpoint when searching for relevant memories.

Memories are also placed into configurable banks so different products or environments can remain isolated.

---

# Switching to Hindsight

One of the useful properties of this architecture is that switching storage backends doesn't require rewriting the agent.

The store is selected through an environment variable:

```python
def build_store(path=None):
    if os.getenv("HINDSIGHT_URL"):
        return HindsightStore()
    return InMemoryStore(path)
```

When `HINDSIGHT_URL` is available, the application uses Hindsight.

Otherwise, it uses the local implementation.

A production environment can configure:

```bash
export HINDSIGHT_URL=https://your-hindsight-instance.example.com
export HINDSIGHT_BANK=support-prod
```

The rest of the agent remains unchanged.

---

# From Keyword Matching to Semantic Recall

The local store performs lexical matching.

For example, if a customer says:

```text
The application is crashing during PDF upload.
```

the system can compare words such as:

```text
application
crashing
PDF
upload
```

against stored memories.

This works for simple cases, but it can struggle when the customer uses different wording.

For example:

```text
"The software freezes when I upload a large document."
```

could refer to a memory stored as:

```text
"Application crashes during large PDF upload."
```

The concepts are related even though the exact words differ.

Hindsight provides semantic retrieval through embeddings, allowing related memories to be retrieved even when the wording is different.

This gives the project a path from a simple local implementation to a more capable persistent memory system.

---

# Testing the Memory Layer

The architecture was also designed with testing in mind.

The HTTP client used by `HindsightStore` is injected rather than created internally.

This allows tests to replace the real network transport with a mock transport.

For example:

```python
store = HindsightStore(
    base_url="http://fake",
    client=mock_client
)
```

The tests can then verify the exact requests and responses without requiring:

- a running Hindsight server,
- a network connection,
- or an API key.

This makes the storage contract independently testable.

The project also includes tests covering returning-customer behavior and memory isolation between customers.

---

# What Does This Mean for Customer Support?

The value of memory appears directly in the support workflow.

## Faster Resolution for Returning Customers

A returning customer doesn't need to repeatedly explain the same problem.

If a known fix worked previously, the agent can surface it immediately.

## Fewer Repeated Troubleshooting Steps

If previous troubleshooting attempts failed, the agent can avoid repeating them and provide the relevant context when escalation is necessary.

## Knowledge That Compounds

Every interaction can add information to the customer's history.

A fix confirmed once can become the starting point the next time the same problem occurs.

## Auditable AI Behavior

The system can expose which memories were retrieved and why they were relevant.

Instead of only seeing:

```text
AI: Try updating the application.
```

the system can expose information such as:

```text
Known fix: Update application
Previous outcome: worked
Recall score: 0.71
Reason: matching product and symptoms
```

This provides visibility into the context used to produce the response.

---

# Lessons We Learned

Building the system showed us that persistent storage is only one part of the problem.

## 1. Outcome extraction is difficult

Consider:

> "I cleared the cache, but it was still crashing, then updating fixed it."

The system needs to correctly identify:

```text
Clear cache → failed
Update → worked
```

Understanding which outcome belongs to which action is one of the more challenging parts of the memory pipeline.

## 2. Deterministic fallbacks are useful

A `ScriptedEngine` allows the memory system to be developed and tested without depending completely on an external API.

If the model is unavailable, the system can still produce a more structured response instead of failing completely.

## 3. Per-customer retrieval matters

Recall should be evaluated against the appropriate customer's history rather than treating all memories as one global collection.

This helps distinguish common terms from more diagnostic information within that customer's history.

## 4. The Protocol pattern makes the system easier to evolve

Both `InMemoryStore` and `HindsightStore` implement the same interface.

This makes them independently testable and allows the production backend to change without rewriting the support agent.

## 5. Memory timing belongs in the architecture

Recall must happen before ingesting the current message.

The distinction between historical information and the current interaction is fundamental to how the system behaves.

---

# Future Scope

The current system establishes the core memory loop, but there are several directions for extending it.

### Semantic Recall

The local retrieval implementation can rely more heavily on Hindsight's semantic retrieval capabilities so that differently worded descriptions of the same problem can still retrieve relevant memories.

### Cross-Customer Pattern Detection

Customer memories currently remain isolated.

A future extension could aggregate anonymized outcomes across customers to identify broader troubleshooting patterns, such as which fixes work most reliably for a particular symptom.

### Proactive Escalation

If a customer experiences repeated failures for the same problem, the system could automatically flag the case for escalation instead of waiting for the customer to request a human agent.

### Multi-Product Memory Isolation

Hindsight memory banks can support multiple products or environments while keeping their memories separated.

### Voice and Async Support

The extraction pipeline operates on text, which means transcribed voice calls, email conversations, and asynchronous chat could eventually use the same memory pipeline.

---

# The Core Insight

The technical implementation involves memory extraction, structured storage, retrieval, scoring, testing, and Hindsight integration.

But the underlying idea is simple:

> **An agent that remembers what worked can make the next interaction more useful.**

The value isn't necessarily in a single response.

It comes from accumulation.

A customer explains a problem.

The agent learns what was tried.

A solution works.

That outcome becomes memory.

The customer returns later.

The agent recalls the previous experience.

Instead of starting over, it starts from what it already knows.

That's what we're building with a memory-enabled customer support agent: **not an AI that merely remembers conversations, but an agent that turns previous interactions into actionable context for the next one.**

---

## References

- [Hindsight GitHub](https://github.com/vectorize-io/hindsight)
- [Hindsight Documentation](https://hindsight.vectorize.io/)
- [What Is Agent Memory — Vectorize](https://vectorize.io/what-is-agent-memory)
- [httpx](https://www.python-httpx.org)

## Demo

Add your project demonstration video here.

## Project

Add your team's project repository, team information, and other required project links here.
