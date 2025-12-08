from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.agent_service import AgentService
from app.schemas.agent import (
    AgentCreate,
    AgentUpdate,
    AgentResponse,
    ChatRequest,
    ChatResponse,
)

router = APIRouter()


@router.post("/", response_model=AgentResponse)
async def create_agent(
    data: AgentCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new agent."""
    service = AgentService(db)
    try:
        agent = await service.create(data)
        return agent
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=list[AgentResponse])
async def list_agents(
    db: AsyncSession = Depends(get_db),
):
    """List all agents."""
    service = AgentService(db)
    agents = await service.get_all()
    return agents


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific agent."""
    service = AgentService(db)
    agent = await service.get_by_id(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.patch("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    data: AgentUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update an agent."""
    service = AgentService(db)
    agent = await service.update(agent_id, data)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete an agent."""
    service = AgentService(db)
    success = await service.delete(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "deleted"}


@router.post("/{agent_id}/chat", response_model=ChatResponse)
async def chat_with_agent(
    agent_id: str,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """Send a message to an agent and get a response."""
    service = AgentService(db)
    try:
        response = await service.chat(agent_id, request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
