from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.agent import Agent
from app.schemas.agent import AgentCreate, AgentUpdate, ChatRequest, ChatResponse
from app.services.provider_service import ProviderService
from app.services.datasource_service import DataSourceService
from app.providers.llm import LLMProviderFactory, LLMMessage
from app.schemas.provider import ProviderType


class AgentService:
    """Service for managing agents and executing chat."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.provider_service = ProviderService(db)
        self.datasource_service = DataSourceService(db)

    async def create(self, data: AgentCreate) -> Agent:
        # Verify provider exists
        provider = await self.provider_service.get_by_id(data.provider_id)
        if not provider:
            raise ValueError(f"Provider not found: {data.provider_id}")

        # Get datasources
        datasources = await self.datasource_service.get_by_ids(data.datasource_ids)

        agent = Agent(
            name=data.name,
            description=data.description,
            system_prompt=data.system_prompt,
            provider_id=data.provider_id,
        )
        agent.datasources = datasources

        self.db.add(agent)
        await self.db.commit()
        await self.db.refresh(agent)
        return agent

    async def get_by_id(self, agent_id: str) -> Agent | None:
        result = await self.db.execute(
            select(Agent)
            .options(selectinload(Agent.provider), selectinload(Agent.datasources))
            .where(Agent.id == agent_id)
        )
        return result.scalar_one_or_none()

    async def get_all(self) -> list[Agent]:
        result = await self.db.execute(
            select(Agent)
            .options(selectinload(Agent.provider), selectinload(Agent.datasources))
            .order_by(Agent.created_at.desc())
        )
        return list(result.scalars().all())

    async def update(self, agent_id: str, data: AgentUpdate) -> Agent | None:
        agent = await self.get_by_id(agent_id)
        if not agent:
            return None

        update_data = data.model_dump(exclude_unset=True)

        # Handle datasource_ids separately
        if "datasource_ids" in update_data:
            datasource_ids = update_data.pop("datasource_ids")
            agent.datasources = await self.datasource_service.get_by_ids(datasource_ids)

        for field, value in update_data.items():
            setattr(agent, field, value)

        await self.db.commit()
        await self.db.refresh(agent)
        return agent

    async def delete(self, agent_id: str) -> bool:
        agent = await self.get_by_id(agent_id)
        if not agent:
            return False

        await self.db.delete(agent)
        await self.db.commit()
        return True

    async def chat(self, agent_id: str, request: ChatRequest) -> ChatResponse:
        """Execute a chat with an agent."""
        agent = await self.get_by_id(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")

        # Build context from datasources
        context_parts = []
        for ds in agent.datasources:
            if ds.content:
                context_parts.append(f"### {ds.name}\n{ds.content}")

        context = "\n\n".join(context_parts)

        # Build system prompt with context
        system_prompt = agent.system_prompt
        if context:
            system_prompt = f"{system_prompt}\n\n## Reference Data:\n{context}"

        # Build messages
        messages = [LLMMessage(role="system", content=system_prompt)]

        # Add conversation history
        for msg in request.conversation_history:
            messages.append(LLMMessage(role=msg.role, content=msg.content))

        # Add current message
        messages.append(LLMMessage(role="user", content=request.message))

        # Create LLM provider and get response
        provider = agent.provider
        llm = LLMProviderFactory.create(
            provider_type=ProviderType(provider.provider_type),
            api_key=provider.api_key,
            model_name=provider.model_name,
            base_url=provider.base_url,
        )

        response = await llm.chat(messages)

        return ChatResponse(
            response=response.content,
            usage=response.usage,
        )
