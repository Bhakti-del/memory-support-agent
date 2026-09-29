"""FastAPI backend for the memory-enabled support agent.

Endpoints
    GET  /health
    GET  /customers
    GET  /customers/{cid}/memories      -> full timeline
    POST /support/ask                  -> answer using recalled memory
    POST /support/step                 -> record a troubleshooting attempt
    POST /support/session/close        -> consolidate a finished conversation
    POST /customers/{cid}/profile      -> store a durable environment fact
    POST /admin/reset                  -> wipe the local store (demo helper)
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from memory_agent import SupportAgent
from memory_agent.engine import build_engine
from memory_agent.models import Memory
from memory_agent.store import HindsightStore, InMemoryStore, build_store

STORE_PATH = Path(os.getenv("MEMORY_STORE_PATH", "memory_store.json"))

app = FastAPI(
    title="Memory-Enabled Customer Support Agent",
    version="0.1.0",
    description="Support agent that recalls past customer experience before answering.",
)


def _make_agent() -> SupportAgent:
    # build_store is the one place that decides which backend is live, and it
    # wraps Hindsight so an unreachable instance degrades to local JSON.
    store = build_store(STORE_PATH)
    return SupportAgent(store=store, engine=build_engine())


agent = _make_agent()


class AskRequest(BaseModel):
    customer_id: str = Field(..., min_length=1, examples=["CUST-1042"])
    message: str = Field(..., min_length=1, examples=["The app crashes again on large PDF upload"])
    session_id: str | None = None


class StepRequest(BaseModel):
    customer_id: str
    step: str = Field(..., min_length=1, examples=["Update application"])
    outcome: str = Field(..., examples=["worked"])
    symptom: str | None = None
    product: str | None = None
    session_id: str | None = None


class CloseSessionRequest(BaseModel):
    customer_id: str
    summary: str
    product: str | None = None
    session_id: str | None = None


class ProfileRequest(BaseModel):
    fact: str = Field(..., min_length=1, examples=["Runs macOS 15 on an M-series MacBook"])


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "store": "hindsight" if os.getenv("HINDSIGHT_URL") else "local",
        "engine": agent.engine.name,
        "customers": agent.store.customers(),
    }


@app.get("/customers")
def list_customers() -> dict:
    return {"customers": agent.store.customers()}


@app.get("/customers/{customer_id}/memories")
def memories(customer_id: str) -> dict:
    timeline = agent.timeline(customer_id)
    return {
        "customer_id": customer_id,
        "count": len(timeline),
        "memories": [m.to_dict() for m in timeline],
    }


@app.post("/support/ask")
def ask(req: AskRequest) -> dict:
    reply = agent.ask(req.customer_id, req.message, req.session_id)
    return reply.to_dict()


@app.post("/support/step")
def record_step(req: StepRequest) -> dict:
    """Record a step explicitly.

    Optional: `/support/ask` mines steps out of the customer's own words. This
    endpoint is for the cases a message cannot cover — a step taken outside the
    chat, or a caller that already has structured data.
    """
    memory: Memory = agent.report_step(
        req.customer_id, req.step, req.outcome,
        symptom=req.symptom, product=req.product, session_id=req.session_id,
    )
    return memory.to_dict()


@app.post("/support/session/close")
def close_session(req: CloseSessionRequest) -> dict:
    memory = agent.close_session(
        req.customer_id, req.summary, product=req.product, session_id=req.session_id
    )
    return memory.to_dict()


@app.post("/customers/{customer_id}/profile")
def add_profile(customer_id: str, req: ProfileRequest) -> dict:
    return agent.set_profile(customer_id, req.fact).to_dict()


@app.post("/admin/reset")
def reset() -> dict:
    """Wipe memory. Local store clears the file; Hindsight banks are cleared
    per customer through the store, not by deleting a local file."""
    global agent
    for customer_id in list(agent.store.customers()):
        agent.store.forget(customer_id)
    if isinstance(agent.store, InMemoryStore):
        STORE_PATH.unlink(missing_ok=True)
    agent = _make_agent()
    return {"status": "reset"}
