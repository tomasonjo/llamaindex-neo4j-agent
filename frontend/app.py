"""Streamlit chat UI for the LlamaIndex Neo4j agent."""

from __future__ import annotations

import json
import os
import uuid

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Neo4j Agent", page_icon="🔎", layout="wide")
st.title("🔎 LlamaIndex Neo4j Agent")
st.caption("Retrieval: remote Neo4j via MCP · Memory: local Neo4j")

with st.sidebar:
    st.subheader("Session")
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"
    st.session_state.session_id = st.text_input(
        "Session ID", value=st.session_state.session_id
    )
    if st.button("🆕 New session"):
        st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()

    st.divider()
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

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for ev in msg.get("events", []):
            if ev["type"] == "tool_call":
                with st.expander(f"🔧 {ev['name']}"):
                    st.json(ev.get("args", {}))
            elif ev["type"] == "tool_result":
                with st.expander(f"📦 result: {ev['name']}"):
                    st.code(ev.get("output", ""))

prompt = st.chat_input("Ask something about the graph…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {"session_id": st.session_state.session_id, "message": prompt}

    with st.chat_message("assistant"):
        if stream:
            placeholder = st.empty()
            events_box = st.container()
            full_text = ""
            events: list[dict] = []
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
                            full_text += event["text"]
                            placeholder.markdown(full_text + "▌")
                        elif t == "final":
                            full_text = event["text"] or full_text
                            placeholder.markdown(full_text)
                        elif t in ("tool_call", "tool_result"):
                            events.append(event)
                            with events_box:
                                if t == "tool_call":
                                    with st.expander(f"🔧 {event['name']}"):
                                        st.json(event.get("args", {}))
                                else:
                                    with st.expander(f"📦 result: {event['name']}"):
                                        st.code(event.get("output", ""))
                        elif t == "error":
                            st.error(event.get("message", "unknown error"))
                placeholder.markdown(full_text)
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                full_text = ""
            st.session_state.messages.append(
                {"role": "assistant", "content": full_text, "events": events}
            )
        else:
            try:
                r = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=300)
                r.raise_for_status()
                data = r.json()
                st.markdown(data["response"])
                for ev in data.get("events", []):
                    if ev["type"] == "tool_call":
                        with st.expander(f"🔧 {ev['name']}"):
                            st.json(ev.get("args", {}))
                    else:
                        with st.expander(f"📦 result: {ev['name']}"):
                            st.code(ev.get("output", ""))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": data["response"],
                        "events": data.get("events", []),
                    }
                )
            except Exception as exc:
                st.error(f"Request failed: {exc}")
