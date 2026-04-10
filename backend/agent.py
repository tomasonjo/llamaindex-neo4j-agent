"""LlamaIndex FunctionAgent wired to:

- A retrieval Neo4j exposed through the ``neo4j-mcp-server`` MCP process.
- A local (docker) Neo4j used for agent memory via ``neo4j-agent-memory``.
"""

from __future__ import annotations

import asyncio
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

    async def shutdown(self) -> None:
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
