"""LlamaIndex FunctionAgent wired to:

- A retrieval Neo4j exposed through the ``neo4j-mcp-server`` MCP process.
- A local (docker) Neo4j used for agent memory via ``neo4j-agent-memory``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from typing import Any, AsyncIterator, Optional

from llama_index.core.agent.workflow import (
    AgentStream,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.llms.openai import OpenAI
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

from neo4j import AsyncGraphDatabase
from neo4j_agent_memory import MemoryClient, MemorySettings
from neo4j_agent_memory.integrations.llamaindex import Neo4jLlamaIndexMemory

from .config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an agent that can search in Neo4j for relevant information.
Always call the get-schema tool before executing any Cypher queries so you know the
available labels, relationships and properties. Prefer short, efficient queries and
explain results clearly to the user."""


class AgentService:
    """Owns the MCP subprocess, the LLM, the tool list and per-session memories."""

    def __init__(self) -> None:
        self._exit_stack = AsyncExitStack()
        self._memory_client: Optional[MemoryClient] = None
        self._memory_driver = None
        self._mcp_client: Optional[BasicMCPClient] = None
        self._tools: list = []
        self._llm: Optional[OpenAI] = None
        self._memories: dict[str, Neo4jLlamaIndexMemory] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ setup

    async def startup(self) -> None:
        """Launch the MCP server over stdio and open the memory client."""
        # BasicMCPClient launches `neo4j-mcp-server` as a child process and
        # talks to it over stdio when the first argument is a command, not URL.
        self._mcp_client = BasicMCPClient(
            "neo4j-mcp-server",
            args=[
                "--neo4j-uri", settings.retrieval_neo4j_uri,
                "--neo4j-database", settings.retrieval_neo4j_database,
                "--neo4j-username", settings.retrieval_neo4j_username,
                "--neo4j-password", settings.retrieval_neo4j_password,
                "--neo4j-transport-mode", "stdio",
                "--neo4j-read-only", "true",
            ],
        )
        tool_spec = McpToolSpec(client=self._mcp_client)
        self._tools = await tool_spec.to_tool_list_async()
        logger.info("Loaded %d MCP tools", len(self._tools))

        self._llm = OpenAI(model=settings.openai_model, api_key=settings.openai_api_key)

        memory_settings = MemorySettings(
            neo4j={
                "uri": settings.memory_neo4j_uri,
                "username": settings.memory_neo4j_username,
                "password": settings.memory_neo4j_password,
                "database": settings.memory_neo4j_database,
            }
        )
        self._memory_client = await self._exit_stack.enter_async_context(
            MemoryClient(settings=memory_settings)
        )
        logger.info("Memory client connected to %s", settings.memory_neo4j_uri)

        # Direct driver used only for listing existing sessions (read-only).
        self._memory_driver = AsyncGraphDatabase.driver(
            settings.memory_neo4j_uri,
            auth=(settings.memory_neo4j_username, settings.memory_neo4j_password),
        )

    async def shutdown(self) -> None:
        if self._memory_driver is not None:
            await self._memory_driver.close()
        await self._exit_stack.aclose()

    # ---------------------------------------------------------------- internal

    async def _get_memory(self, session_id: str) -> Neo4jLlamaIndexMemory:
        async with self._lock:
            if session_id not in self._memories:
                self._memories[session_id] = Neo4jLlamaIndexMemory.from_defaults(
                    session_id=session_id,
                    memory_client=self._memory_client,
                )
            return self._memories[session_id]

    def _build_workflow(self) -> FunctionAgent:
        return FunctionAgent(
            tools=self._tools,
            llm=self._llm,
            system_prompt=SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------------ public

    async def chat(self, session_id: str, user_msg: str) -> dict[str, Any]:
        """Run the agent once and return the final response plus trace events."""
        memory = await self._get_memory(session_id)
        workflow = self._build_workflow()
        handler = workflow.run(user_msg=user_msg, memory=memory)

        events: list[dict[str, Any]] = []
        async for event in handler.stream_events():
            if isinstance(event, ToolCall):
                events.append({
                    "type": "tool_call",
                    "name": event.tool_name,
                    "args": event.tool_kwargs,
                })
            elif isinstance(event, ToolCallResult):
                events.append({
                    "type": "tool_result",
                    "name": event.tool_name,
                    "output": str(event.tool_output),
                })

        response = await handler
        return {"response": str(response), "events": events}

    async def list_sessions(self) -> list[dict[str, Any]]:
        """Return all known sessions stored in the memory Neo4j.

        neo4j-agent-memory writes ``(:Conversation {session_id, created_at,
        updated_at})`` nodes. We aggregate by ``session_id`` (multiple
        conversations can share one) and return them newest first.
        """
        if self._memory_driver is None:
            return []
        cypher = """
        MATCH (c:Conversation)
        WHERE c.session_id IS NOT NULL
        RETURN c.session_id AS session_id,
               max(c.updated_at) AS updated_at,
               min(c.created_at) AS created_at
        ORDER BY updated_at DESC
        """
        try:
            async with self._memory_driver.session(
                database=settings.memory_neo4j_database
            ) as session:
                result = await session.run(cypher)
                rows = []
                async for record in result:
                    rows.append(
                        {
                            "session_id": record["session_id"],
                            "updated_at": str(record["updated_at"])
                            if record["updated_at"] is not None
                            else None,
                            "created_at": str(record["created_at"])
                            if record["created_at"] is not None
                            else None,
                        }
                    )
            return rows
        except Exception as exc:
            logger.warning("list_sessions failed: %s", exc)
            return []

    async def get_history(self, session_id: str) -> list[dict[str, Any]]:
        """Return the stored conversation for ``session_id`` from Neo4j memory,
        fully reconstructed with tool calls and tool results in order.

        Messages are fetched directly from the memory Neo4j (Conversation ->
        Message). Consecutive assistant/tool messages are grouped into a
        single assistant turn whose ``items`` list interleaves text blocks,
        tool calls and tool results in the order the agent produced them.

        Returned shape::

            [
              {"role": "user",      "items": [{"type": "text", "text": ...}]},
              {"role": "assistant", "items": [
                  {"type": "tool_call",   "name": ..., "args": {...}},
                  {"type": "tool_result", "name": ..., "output": ...},
                  {"type": "text",        "text": ...},
              ]},
              ...
            ]
        """
        if self._memory_driver is None:
            return []

        cypher = """
        MATCH (c:Conversation {session_id: $sid})-[:HAS_MESSAGE]->(m:Message)
        RETURN m.role AS role,
               m.content AS content,
               m.metadata AS metadata,
               m.timestamp AS ts
        ORDER BY m.timestamp
        """
        raw: list[dict[str, Any]] = []
        try:
            async with self._memory_driver.session(
                database=settings.memory_neo4j_database
            ) as session:
                result = await session.run(cypher, sid=session_id)
                async for record in result:
                    raw.append(
                        {
                            "role": record["role"],
                            "content": record["content"] or "",
                            "metadata": record["metadata"],
                        }
                    )
        except Exception as exc:
            logger.warning("get_history failed: %s", exc)
            return []

        # tool_call_id -> tool_name, so tool-result messages can be labelled
        tool_call_names: dict[str, str] = {}
        messages: list[dict[str, Any]] = []
        current_assistant: Optional[dict[str, Any]] = None

        def _meta(raw_meta: Any) -> dict[str, Any]:
            if not raw_meta:
                return {}
            if isinstance(raw_meta, dict):
                return raw_meta
            try:
                return json.loads(raw_meta)
            except Exception:
                return {}

        def _ensure_assistant() -> dict[str, Any]:
            nonlocal current_assistant
            if current_assistant is None:
                current_assistant = {"role": "assistant", "items": []}
                messages.append(current_assistant)
            return current_assistant

        for row in raw:
            role = row["role"]
            content = row["content"]
            meta = _meta(row["metadata"])

            if role == "user":
                current_assistant = None  # start a new assistant turn after this
                messages.append(
                    {"role": "user", "items": [{"type": "text", "text": content}]}
                )

            elif role == "assistant":
                turn = _ensure_assistant()
                tool_calls = meta.get("tool_calls") or []
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function") or {}
                        name = fn.get("name", "tool")
                        args_raw = fn.get("arguments")
                        try:
                            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                        except Exception:
                            args = {"_raw": args_raw}
                        turn["items"].append(
                            {"type": "tool_call", "name": name, "args": args}
                        )
                        if tc.get("id"):
                            tool_call_names[tc["id"]] = name
                if content:
                    turn["items"].append({"type": "text", "text": content})

            elif role == "tool":
                turn = _ensure_assistant()
                name = tool_call_names.get(meta.get("tool_call_id", ""), "tool")
                turn["items"].append(
                    {"type": "tool_result", "name": name, "output": content}
                )
            # silently ignore system/other roles

        return messages

    async def stream_chat(
        self, session_id: str, user_msg: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield streaming events (token deltas, tool calls, tool results, final)."""
        memory = await self._get_memory(session_id)
        workflow = self._build_workflow()
        handler = workflow.run(user_msg=user_msg, memory=memory)

        async for event in handler.stream_events():
            if isinstance(event, AgentStream):
                if event.delta:
                    yield {"type": "delta", "text": event.delta}
            elif isinstance(event, ToolCall):
                yield {
                    "type": "tool_call",
                    "name": event.tool_name,
                    "args": event.tool_kwargs,
                }
            elif isinstance(event, ToolCallResult):
                yield {
                    "type": "tool_result",
                    "name": event.tool_name,
                    "output": str(event.tool_output),
                }

        response = await handler
        yield {"type": "final", "text": str(response)}


agent_service = AgentService()
