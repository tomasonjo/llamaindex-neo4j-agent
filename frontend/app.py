"""Streamlit chat UI for the LlamaIndex Neo4j agent."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Neo4j Agent", page_icon="🔎", layout="wide")

# ---------------------------------------------------------------- session state

def _fetch_remote_sessions() -> list[dict[str, Any]]:
    """Fetch all sessions known to the backend (Neo4j memory)."""
    try:
        r = requests.get(f"{BACKEND_URL}/sessions", timeout=10)
        r.raise_for_status()
        return r.json().get("sessions", [])
    except Exception as exc:
        st.warning(f"Could not list sessions: {exc}")
        return []


def _load_history_from_backend(session_id: str) -> list[dict[str, Any]]:
    """Fetch this session's conversation history from Neo4j via the backend.

    The backend already returns messages in the {role, items:[...]} shape the
    UI renders natively, with tool calls and results interleaved in the right
    order.
    """
    try:
        r = requests.get(
            f"{BACKEND_URL}/sessions/{session_id}/history", timeout=10
        )
        r.raise_for_status()
        return r.json().get("messages", [])
    except Exception as exc:
        st.warning(f"Could not load history for {session_id}: {exc}")
        return []


if "sessions" not in st.session_state:
    # {session_id: {"messages": [...], "loaded": bool}}
    st.session_state.sessions = {}
    # Seed from whatever the backend already has in Neo4j.
    for row in _fetch_remote_sessions():
        sid = row["session_id"]
        st.session_state.sessions[sid] = {"messages": [], "loaded": False}
    if not st.session_state.sessions:
        sid = f"session-{uuid.uuid4().hex[:8]}"
        st.session_state.sessions[sid] = {"messages": [], "loaded": True}
    st.session_state.active_session = next(iter(st.session_state.sessions))


def _new_session() -> str:
    sid = f"session-{uuid.uuid4().hex[:8]}"
    st.session_state.sessions[sid] = {"messages": [], "loaded": True}
    st.session_state.active_session = sid
    return sid


def _ensure_loaded(session_id: str) -> None:
    """Load the session's history from Neo4j once on first access."""
    sess = st.session_state.sessions[session_id]
    if not sess.get("loaded"):
        sess["messages"] = _load_history_from_backend(session_id)
        sess["loaded"] = True


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("💬 Sessions")

    if st.button("➕ New session", use_container_width=True):
        _new_session()
        st.rerun()

    st.divider()

    for sid in list(st.session_state.sessions.keys()):
        is_active = sid == st.session_state.active_session
        if st.button(
            f"{'▶ ' if is_active else '  '}{sid}",
            key=f"sel-{sid}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.active_session = sid
            st.rerun()

    st.divider()
    st.caption("Active session ID")
    st.code(st.session_state.active_session, language=None)
    stream = st.toggle("Stream responses", value=True)
    st.caption(f"Backend: `{BACKEND_URL}`")
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if r.ok:
            st.success("Backend healthy")
        else:
            st.error(f"Backend error: {r.status_code}")
    except Exception as exc:
        st.error(f"Backend unreachable: {exc}")


# ---------------------------------------------------------------- main

st.title("🔎 LlamaIndex Neo4j Agent")
st.caption("Retrieval: remote Neo4j via MCP · Memory: local Neo4j")

_ensure_loaded(st.session_state.active_session)
active = st.session_state.sessions[st.session_state.active_session]
messages: list[dict[str, Any]] = active["messages"]


def _render_item(item: dict[str, Any]) -> None:
    """Render a single ordered message item (text or tool call/result)."""
    kind = item["type"]
    if kind == "text":
        st.markdown(item["text"])
    elif kind == "tool_call":
        with st.expander(f"🔧 {item['name']}"):
            st.json(item.get("args", {}))
    elif kind == "tool_result":
        with st.expander(f"📦 result: {item['name']}"):
            st.code(item.get("output", ""))


# --- render history in order ---
for msg in messages:
    with st.chat_message(msg["role"]):
        for item in msg["items"]:
            _render_item(item)


# --- chat input ---
prompt = st.chat_input("Ask something about the graph…")

if prompt:
    messages.append({"role": "user", "items": [{"type": "text", "text": prompt}]})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "session_id": st.session_state.active_session,
        "message": prompt,
    }

    assistant_items: list[dict[str, Any]] = []

    with st.chat_message("assistant"):
        if stream:
            current_text = ""
            current_text_slot: Any = None  # active st.empty() placeholder or None

            try:
                with requests.post(
                    f"{BACKEND_URL}/chat/stream",
                    json=payload,
                    stream=True,
                    timeout=300,
                ) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line:
                            continue
                        line = line.decode("utf-8")
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])
                        t = event.get("type")

                        if t == "delta":
                            if current_text_slot is None:
                                current_text = ""
                                current_text_slot = st.empty()
                            current_text += event["text"]
                            current_text_slot.markdown(current_text + "▌")

                        elif t == "final":
                            final_text = event.get("text") or current_text
                            if current_text_slot is None:
                                st.markdown(final_text)
                            else:
                                current_text_slot.markdown(final_text)
                                current_text_slot = None
                            if final_text:
                                assistant_items.append(
                                    {"type": "text", "text": final_text}
                                )
                            current_text = ""

                        elif t in ("tool_call", "tool_result"):
                            # seal any pending text block first so ordering is preserved
                            if current_text_slot is not None:
                                current_text_slot.markdown(current_text)
                                if current_text:
                                    assistant_items.append(
                                        {"type": "text", "text": current_text}
                                    )
                                current_text_slot = None
                                current_text = ""
                            item = {
                                "type": t,
                                "name": event["name"],
                                **(
                                    {"args": event.get("args", {})}
                                    if t == "tool_call"
                                    else {"output": event.get("output", "")}
                                ),
                            }
                            _render_item(item)
                            assistant_items.append(item)

                        elif t == "error":
                            st.error(event.get("message", "unknown error"))

                # if the stream ended with an unfinalized partial text
                if current_text_slot is not None and current_text:
                    current_text_slot.markdown(current_text)
                    assistant_items.append({"type": "text", "text": current_text})
            except Exception as exc:
                st.error(f"Request failed: {exc}")

        else:
            try:
                r = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=300)
                r.raise_for_status()
                data = r.json()
                # Non-streaming: events are in order; render them then the response text.
                for ev in data.get("events", []):
                    item = {
                        "type": ev["type"],
                        "name": ev["name"],
                        **(
                            {"args": ev.get("args", {})}
                            if ev["type"] == "tool_call"
                            else {"output": ev.get("output", "")}
                        ),
                    }
                    _render_item(item)
                    assistant_items.append(item)
                st.markdown(data["response"])
                assistant_items.append({"type": "text", "text": data["response"]})
            except Exception as exc:
                st.error(f"Request failed: {exc}")

    messages.append({"role": "assistant", "items": assistant_items})
