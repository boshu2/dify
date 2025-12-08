"""
RAG-Enabled Agent Implementation.

Integrates the 12-factor agent with the RAG pipeline to enable
knowledge retrieval during agent execution.
"""
from dataclasses import dataclass, field
from typing import Any, Callable

from app.agents.twelve_factor_agent import (
    Agent,
    AgentConfig,
    AgentState,
    LLMClient,
    ToolDefinition,
)
from app.rag.retriever import RetrievalResult, Retriever
from app.rag.vector_store import Document


@dataclass
class RAGAgentConfig:
    """Configuration for RAG-enabled agent."""
    agent_id: str
    purpose: str
    retriever: Retriever
    llm_client: LLMClient | None = None
    tools: list[ToolDefinition] = field(default_factory=list)
    max_iterations: int = 10
    top_k: int = 5  # Number of documents to retrieve
    include_sources: bool = True  # Include source citations


class RAGAgent:
    """
    12-Factor Agent with integrated RAG capabilities.

    Adds a knowledge_search tool that queries the RAG pipeline
    and injects relevant context into the agent's conversation.
    """

    def __init__(self, config: RAGAgentConfig):
        self.config = config
        self._agent: Agent | None = None
        self._setup_agent()

    def _setup_agent(self) -> None:
        """Initialize the underlying 12-factor agent with RAG tool."""
        # Create the knowledge search tool
        knowledge_tool = ToolDefinition(
            name="knowledge_search",
            description=(
                "Search the knowledge base for relevant information. "
                "Use this when you need facts, documentation, or context "
                "to answer the user's question."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant information",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": self.config.top_k,
                    },
                },
                "required": ["query"],
            },
            handler=self._search_knowledge,
        )

        # Combine RAG tool with user-provided tools
        all_tools = [knowledge_tool] + list(self.config.tools)

        # Create the underlying agent
        agent_config = AgentConfig(
            agent_id=self.config.agent_id,
            purpose=self._build_purpose(),
            tools=all_tools,
            max_iterations=self.config.max_iterations,
            llm_client=self.config.llm_client,
        )
        self._agent = Agent(agent_config)

    def _build_purpose(self) -> str:
        """Build agent purpose with RAG instructions."""
        return f"""{self.config.purpose}

You have access to a knowledge base. When you need factual information
or documentation, use the knowledge_search tool to retrieve relevant content.
Always cite your sources when using retrieved information."""

    async def _search_knowledge(
        self,
        query: str,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Search the knowledge base.

        This is the handler for the knowledge_search tool.
        """
        limit = limit or self.config.top_k

        try:
            result: RetrievalResult = await self.config.retriever.retrieve(
                query=query,
                limit=limit,
            )

            # Format results for the agent
            formatted_results = []
            for search_result in result.documents:
                doc = search_result.document
                formatted_doc: dict[str, Any] = {
                    "content": doc.content,
                    "relevance_score": search_result.score,
                }

                if self.config.include_sources:
                    formatted_doc["source"] = doc.metadata.get("source", "unknown")
                    formatted_doc["id"] = doc.id

                formatted_results.append(formatted_doc)

            return {
                "query": query,
                "num_results": len(formatted_results),
                "results": formatted_results,
                "strategy": result.strategy,
            }

        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "num_results": 0,
                "results": [],
            }

    def launch(self, user_message: str) -> AgentState:
        """Launch a new agent execution."""
        if not self._agent:
            raise RuntimeError("Agent not initialized")
        return self._agent.launch(user_message)

    async def step(self, state: AgentState) -> AgentState:
        """Execute one step of the agent."""
        if not self._agent:
            raise RuntimeError("Agent not initialized")
        return await self._agent.step(state)

    async def run_to_completion(
        self,
        state: AgentState,
        human_callback: Callable | None = None,
    ) -> AgentState:
        """Run agent until completion."""
        if not self._agent:
            raise RuntimeError("Agent not initialized")
        return await self._agent.run_to_completion(state, human_callback)

    def pause(self, state: AgentState) -> AgentState:
        """Pause agent execution."""
        if not self._agent:
            raise RuntimeError("Agent not initialized")
        return self._agent.pause(state)

    def resume(self, state: AgentState) -> AgentState:
        """Resume agent execution."""
        if not self._agent:
            raise RuntimeError("Agent not initialized")
        return self._agent.resume(state)

    def provide_human_response(
        self,
        state: AgentState,
        response: str,
        approved: bool = True,
    ) -> AgentState:
        """Provide human response to agent."""
        if not self._agent:
            raise RuntimeError("Agent not initialized")
        return self._agent.provide_human_response(state, response, approved)


class RAGPipelineBuilder:
    """
    Builder for creating a complete RAG pipeline.

    Simplifies the setup of embeddings, vector store, and retriever.
    """

    def __init__(self):
        self._embedder = None
        self._vector_store = None
        self._retriever = None
        self._documents: list[Document] = []

    def with_embedder(self, embedder) -> "RAGPipelineBuilder":
        """Set the embedding provider."""
        self._embedder = embedder
        return self

    def with_vector_store(self, vector_store) -> "RAGPipelineBuilder":
        """Set the vector store."""
        self._vector_store = vector_store
        return self

    def with_documents(self, documents: list[Document]) -> "RAGPipelineBuilder":
        """Add documents to index."""
        self._documents.extend(documents)
        return self

    async def build_retriever(self, strategy: str = "hybrid") -> Retriever:
        """Build the retriever with configured components."""
        from app.rag.retriever import create_retriever

        retriever = create_retriever(
            strategy=strategy,
            vector_store=self._vector_store,
            embedder=self._embedder,
            documents=self._documents if self._documents else None,
        )

        # If we have documents and a vector store, index them
        if self._documents and self._vector_store and self._embedder:
            await self._index_documents()

        return retriever

    async def _index_documents(self) -> None:
        """Index documents in the vector store."""
        if not self._documents or not self._vector_store or not self._embedder:
            return

        # Embed documents
        texts = [doc.content for doc in self._documents]
        result = await self._embedder.embed(texts)

        # Attach embeddings to documents
        for doc, embedding in zip(self._documents, result.embeddings):
            doc.embedding = embedding

        # Store in vector store
        await self._vector_store.add_documents(self._documents)


async def create_rag_agent(
    agent_id: str,
    purpose: str,
    retriever: Retriever,
    llm_client: LLMClient | None = None,
    tools: list[ToolDefinition] | None = None,
    **kwargs,
) -> RAGAgent:
    """
    Factory function to create a RAG-enabled agent.

    Args:
        agent_id: Unique agent identifier.
        purpose: Agent's purpose/role.
        retriever: Configured retriever for knowledge search.
        llm_client: LLM client for agent reasoning.
        tools: Additional tools for the agent.
        **kwargs: Additional RAGAgentConfig options.

    Returns:
        Configured RAGAgent.
    """
    config = RAGAgentConfig(
        agent_id=agent_id,
        purpose=purpose,
        retriever=retriever,
        llm_client=llm_client,
        tools=tools or [],
        **kwargs,
    )
    return RAGAgent(config)
