"""Workflow execution routes."""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Any

from app.workflows.types import NodeType, NodeDefinition, Edge
from app.workflows.state import WorkflowDefinition, WorkflowState
from app.workflows.engine import WorkflowEngine

router = APIRouter()


# Schemas
class NodeConfig(BaseModel):
    """Node configuration."""
    id: str
    type: str
    config: dict[str, Any] = {}


class EdgeConfig(BaseModel):
    """Edge configuration."""
    source: str
    target: str
    condition: str | None = None


class WorkflowCreate(BaseModel):
    """Workflow creation request."""
    name: str
    description: str | None = None
    nodes: list[NodeConfig]
    edges: list[EdgeConfig]


class WorkflowExecuteRequest(BaseModel):
    """Workflow execution request."""
    workflow: WorkflowCreate
    variables: dict[str, Any] = {}


class WorkflowExecuteResponse(BaseModel):
    """Workflow execution response."""
    execution_id: str
    status: str
    variables: dict[str, Any]
    node_outputs: dict[str, Any]


class SimpleWorkflowRequest(BaseModel):
    """Simple workflow execution (single node)."""
    node_type: str
    config: dict[str, Any]
    variables: dict[str, Any] = {}


@router.post("/execute", response_model=WorkflowExecuteResponse)
async def execute_workflow(request: WorkflowExecuteRequest):
    """
    Execute a workflow definition.

    Supports all node types:
    - start, end: Flow control
    - llm: LLM completion
    - code: Python code execution
    - http_request: HTTP API calls
    - condition: Branching
    - loop: Iteration
    - transform: Data transformation
    - variable_aggregator: Combine variables
    """
    try:
        # Convert to internal types
        nodes = []
        for n in request.workflow.nodes:
            try:
                node_type = NodeType(n.type)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid node type: {n.type}",
                )
            nodes.append(NodeDefinition(id=n.id, type=node_type, config=n.config))

        edges = [
            Edge(source=e.source, target=e.target, condition=e.condition)
            for e in request.workflow.edges
        ]

        definition = WorkflowDefinition(
            id="temp-workflow",
            name=request.workflow.name,
            description=request.workflow.description,
            nodes=nodes,
            edges=edges,
        )

        # Create engine and execute
        engine = WorkflowEngine()
        state = await engine.execute(definition, variables=request.variables)

        return WorkflowExecuteResponse(
            execution_id=state.execution_id,
            status=state.status.value,
            variables=state.variables,
            node_outputs={
                node_id: exec.output_data
                for node_id, exec in state.node_executions.items()
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution failed: {str(e)}",
        )


@router.post("/execute/node", response_model=dict[str, Any])
async def execute_single_node(request: SimpleWorkflowRequest):
    """
    Execute a single workflow node.

    Useful for testing individual node types:
    - code: Execute Python code
    - http_request: Make HTTP request
    - transform: Transform data
    - loop: Iterate over array
    - variable_aggregator: Combine variables
    """
    try:
        node_type = NodeType(request.node_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid node type: {request.node_type}",
        )

    # Create minimal workflow with single node
    nodes = [
        NodeDefinition(id="start", type=NodeType.START, config={}),
        NodeDefinition(id="node", type=node_type, config=request.config),
        NodeDefinition(id="end", type=NodeType.END, config={}),
    ]
    edges = [
        Edge(source="start", target="node"),
        Edge(source="node", target="end"),
    ]

    definition = WorkflowDefinition(
        id="single-node",
        name="Single Node Execution",
        nodes=nodes,
        edges=edges,
    )

    engine = WorkflowEngine()
    state = await engine.execute(definition, variables=request.variables)

    # Return node output
    node_exec = state.node_executions.get("node")
    if node_exec:
        return {
            "status": "success",
            "output": node_exec.output_data,
            "variables": state.variables,
        }

    return {"status": "completed", "variables": state.variables}


@router.get("/node-types")
async def list_node_types():
    """List all available workflow node types."""
    return {
        "node_types": [
            {
                "type": nt.value,
                "description": _get_node_description(nt),
            }
            for nt in NodeType
        ]
    }


def _get_node_description(node_type: NodeType) -> str:
    """Get description for node type."""
    descriptions = {
        NodeType.START: "Entry point of the workflow",
        NodeType.END: "Exit point of the workflow",
        NodeType.AGENT: "Execute an AI agent",
        NodeType.CONDITION: "Branch based on condition",
        NodeType.PARALLEL: "Execute branches in parallel",
        NodeType.MERGE: "Wait for parallel branches",
        NodeType.LOOP: "Iterate over an array",
        NodeType.HUMAN: "Human-in-the-loop checkpoint",
        NodeType.TRANSFORM: "Transform data (json_parse, json_stringify, template)",
        NodeType.WAIT: "Wait for duration or external event",
        NodeType.HTTP_REQUEST: "Make HTTP API request",
        NodeType.LLM: "Direct LLM completion",
        NodeType.CODE: "Execute Python code",
        NodeType.KNOWLEDGE_RETRIEVAL: "RAG retrieval from knowledge base",
        NodeType.VARIABLE_AGGREGATOR: "Combine multiple variables",
    }
    return descriptions.get(node_type, "No description")
