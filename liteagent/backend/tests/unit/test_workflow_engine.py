"""
Tests for Workflow Engine.

Tests the stateless reducer pattern for workflow orchestration.
"""
import pytest
from datetime import datetime, timezone

from app.workflows.engine import (
    Edge,
    NodeDefinition,
    NodeExecution,
    NodeStatus,
    NodeType,
    WorkflowBuilder,
    WorkflowDefinition,
    WorkflowEngine,
    WorkflowEvent,
    WorkflowReducer,
    WorkflowState,
    WorkflowStatus,
)


# =============================================================================
# Node Definition Tests
# =============================================================================

class TestNodeDefinition:
    """Test NodeDefinition."""

    def test_create_node(self):
        """Test creating a node definition."""
        node = NodeDefinition(
            id="test-node",
            type=NodeType.AGENT,
            config={"purpose": "Test agent"},
        )

        assert node.id == "test-node"
        assert node.type == NodeType.AGENT
        assert node.config["purpose"] == "Test agent"

    def test_node_to_dict(self):
        """Test serializing node to dict."""
        node = NodeDefinition(
            id="node-1",
            type=NodeType.CONDITION,
            config={"condition": "x > 0"},
        )

        data = node.to_dict()

        assert data["id"] == "node-1"
        assert data["type"] == "condition"
        assert data["config"]["condition"] == "x > 0"


# =============================================================================
# Workflow Definition Tests
# =============================================================================

class TestWorkflowDefinition:
    """Test WorkflowDefinition."""

    def test_create_workflow(self):
        """Test creating a workflow definition."""
        nodes = [
            NodeDefinition(id="start", type=NodeType.START),
            NodeDefinition(id="end", type=NodeType.END),
        ]
        edges = [Edge(source="start", target="end")]

        workflow = WorkflowDefinition(
            id="wf-1",
            name="Test Workflow",
            nodes=nodes,
            edges=edges,
        )

        assert workflow.id == "wf-1"
        assert workflow.name == "Test Workflow"
        assert len(workflow.nodes) == 2
        assert len(workflow.edges) == 1

    def test_get_node(self):
        """Test getting node by ID."""
        nodes = [
            NodeDefinition(id="node-1", type=NodeType.AGENT),
            NodeDefinition(id="node-2", type=NodeType.TRANSFORM),
        ]

        workflow = WorkflowDefinition(id="wf", name="Test", nodes=nodes, edges=[])

        node = workflow.get_node("node-1")
        assert node is not None
        assert node.type == NodeType.AGENT

        missing = workflow.get_node("nonexistent")
        assert missing is None

    def test_get_start_node(self):
        """Test getting start node."""
        nodes = [
            NodeDefinition(id="start", type=NodeType.START),
            NodeDefinition(id="process", type=NodeType.AGENT),
            NodeDefinition(id="end", type=NodeType.END),
        ]

        workflow = WorkflowDefinition(id="wf", name="Test", nodes=nodes, edges=[])

        start = workflow.get_start_node()
        assert start is not None
        assert start.id == "start"

    def test_get_outgoing_edges(self):
        """Test getting outgoing edges from a node."""
        edges = [
            Edge(source="a", target="b"),
            Edge(source="a", target="c"),
            Edge(source="b", target="d"),
        ]

        workflow = WorkflowDefinition(id="wf", name="Test", nodes=[], edges=edges)

        outgoing = workflow.get_outgoing_edges("a")
        assert len(outgoing) == 2
        targets = [e.target for e in outgoing]
        assert "b" in targets
        assert "c" in targets


# =============================================================================
# Workflow State Tests
# =============================================================================

class TestWorkflowState:
    """Test WorkflowState."""

    def test_create_state(self):
        """Test creating workflow state."""
        state = WorkflowState(
            workflow_id="wf-1",
            execution_id="exec-1",
        )

        assert state.workflow_id == "wf-1"
        assert state.execution_id == "exec-1"
        assert state.status == WorkflowStatus.PENDING

    def test_variables(self):
        """Test setting and getting variables."""
        state = WorkflowState(workflow_id="wf", execution_id="exec")

        state.set_variable("input", "Hello")
        state.set_variable("count", 42)

        assert state.get_variable("input") == "Hello"
        assert state.get_variable("count") == 42
        assert state.get_variable("missing", "default") == "default"

    def test_node_execution(self):
        """Test node execution tracking."""
        state = WorkflowState(workflow_id="wf", execution_id="exec")

        exec_record = state.get_node_execution("node-1")
        exec_record.status = NodeStatus.RUNNING

        # Getting same node returns same record
        same_record = state.get_node_execution("node-1")
        assert same_record.status == NodeStatus.RUNNING

    def test_serialization(self):
        """Test state serialization and deserialization."""
        state = WorkflowState(
            workflow_id="wf-1",
            execution_id="exec-1",
            status=WorkflowStatus.RUNNING,
        )
        state.set_variable("test", "value")
        state.current_nodes = ["node-1", "node-2"]

        exec_record = state.get_node_execution("node-1")
        exec_record.status = NodeStatus.COMPLETED
        exec_record.output_data = {"result": "success"}

        # Serialize
        data = state.to_dict()

        # Deserialize
        restored = WorkflowState.from_dict(data)

        assert restored.workflow_id == "wf-1"
        assert restored.status == WorkflowStatus.RUNNING
        assert restored.get_variable("test") == "value"
        assert "node-1" in restored.current_nodes
        assert restored.node_executions["node-1"].status == NodeStatus.COMPLETED

    def test_compute_hash(self):
        """Test state hash computation."""
        state1 = WorkflowState(workflow_id="wf", execution_id="exec")
        state1.set_variable("x", 1)
        hash1 = state1.compute_hash()

        state1.set_variable("x", 2)
        hash2 = state1.compute_hash()

        assert hash1 != hash2  # Hash changes with state


# =============================================================================
# Workflow Reducer Tests
# =============================================================================

class TestWorkflowReducer:
    """Test WorkflowReducer."""

    @pytest.fixture
    def simple_workflow(self):
        """Create a simple workflow for testing."""
        return (
            WorkflowBuilder("wf-1", "Simple Workflow")
            .add_start()
            .add_transform("transform", transform="passthrough")
            .add_end()
            .connect("start", "transform")
            .connect("transform", "end")
            .build()
        )

    @pytest.mark.asyncio
    async def test_start_event(self, simple_workflow):
        """Test handling start event."""
        reducer = WorkflowReducer()
        state = WorkflowState(workflow_id="wf-1", execution_id="exec-1")

        event = WorkflowEvent(
            event_type="start",
            data={"input": "Hello"},
        )

        new_state = await reducer.reduce(simple_workflow, state, event)

        assert new_state.status == WorkflowStatus.RUNNING
        assert "start" in new_state.current_nodes
        assert new_state.get_variable("input") == "Hello"

    @pytest.mark.asyncio
    async def test_node_complete_event(self, simple_workflow):
        """Test handling node completion."""
        reducer = WorkflowReducer()
        state = WorkflowState(
            workflow_id="wf-1",
            execution_id="exec-1",
            status=WorkflowStatus.RUNNING,
        )
        state.current_nodes = ["start"]

        event = WorkflowEvent(
            event_type="node_complete",
            node_id="start",
            data={"result": "started"},
        )

        new_state = await reducer.reduce(simple_workflow, state, event)

        # Start node completed, transform node should be running
        assert "transform" in new_state.current_nodes
        assert new_state.get_variable("result") == "started"

    @pytest.mark.asyncio
    async def test_node_failed_event(self, simple_workflow):
        """Test handling node failure."""
        reducer = WorkflowReducer()
        state = WorkflowState(
            workflow_id="wf-1",
            execution_id="exec-1",
            status=WorkflowStatus.RUNNING,
        )
        state.current_nodes = ["transform"]

        event = WorkflowEvent(
            event_type="node_failed",
            node_id="transform",
            data={"error": "Something went wrong"},
        )

        new_state = await reducer.reduce(simple_workflow, state, event)

        assert new_state.status == WorkflowStatus.FAILED
        assert new_state.node_executions["transform"].status == NodeStatus.FAILED
        assert "Something went wrong" in new_state.node_executions["transform"].error

    @pytest.mark.asyncio
    async def test_pause_and_resume(self, simple_workflow):
        """Test pause and resume events."""
        reducer = WorkflowReducer()
        state = WorkflowState(
            workflow_id="wf-1",
            execution_id="exec-1",
            status=WorkflowStatus.RUNNING,
        )

        # Pause
        pause_event = WorkflowEvent(event_type="pause")
        state = await reducer.reduce(simple_workflow, state, pause_event)
        assert state.status == WorkflowStatus.PAUSED

        # Resume
        resume_event = WorkflowEvent(event_type="resume")
        state = await reducer.reduce(simple_workflow, state, resume_event)
        assert state.status == WorkflowStatus.RUNNING


# =============================================================================
# Workflow Engine Tests
# =============================================================================

class TestWorkflowEngine:
    """Test WorkflowEngine."""

    @pytest.fixture
    def simple_workflow(self):
        """Create simple workflow."""
        return (
            WorkflowBuilder("wf-1", "Simple")
            .add_start()
            .add_transform("process", transform="passthrough", input_variable="input", output_variable="output")
            .add_end()
            .connect("start", "process")
            .connect("process", "end")
            .build()
        )

    def test_launch(self, simple_workflow):
        """Test launching workflow."""
        engine = WorkflowEngine()
        state = engine.launch(simple_workflow)

        assert state.workflow_id == "wf-1"
        assert state.execution_id is not None
        assert state.status == WorkflowStatus.PENDING

    @pytest.mark.asyncio
    async def test_start_workflow(self, simple_workflow):
        """Test starting workflow execution."""
        engine = WorkflowEngine()
        state = engine.launch(simple_workflow)

        state = await engine.start(
            simple_workflow,
            state,
            input_data={"input": "Hello World"},
        )

        assert state.status == WorkflowStatus.RUNNING

    @pytest.mark.asyncio
    async def test_run_to_completion(self, simple_workflow):
        """Test running workflow to completion."""
        engine = WorkflowEngine()
        state = engine.launch(simple_workflow)

        state = await engine.start(
            simple_workflow,
            state,
            input_data={"input": "Test Input"},
        )

        state = await engine.run_to_completion(simple_workflow, state)

        assert state.status == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_transform_node_execution(self):
        """Test transform node processes data."""
        workflow = (
            WorkflowBuilder("wf", "Transform Test")
            .add_start()
            .add_transform("upper", transform="template", template="{input} PROCESSED")
            .add_end()
            .connect("start", "upper")
            .connect("upper", "end")
            .build()
        )

        engine = WorkflowEngine()
        state = engine.launch(workflow)

        state = await engine.start(workflow, state, input_data={"input": "Hello"})
        state = await engine.run_to_completion(workflow, state)

        assert state.status == WorkflowStatus.COMPLETED
        assert "PROCESSED" in state.get_variable("output", "")


# =============================================================================
# Workflow Builder Tests
# =============================================================================

class TestWorkflowBuilder:
    """Test WorkflowBuilder."""

    def test_build_simple_workflow(self):
        """Test building a simple workflow."""
        workflow = (
            WorkflowBuilder("wf-1", "My Workflow")
            .add_start()
            .add_end()
            .connect("start", "end")
            .build()
        )

        assert workflow.id == "wf-1"
        assert workflow.name == "My Workflow"
        assert len(workflow.nodes) == 2
        assert len(workflow.edges) == 1

    def test_build_with_agent(self):
        """Test building workflow with agent node."""
        workflow = (
            WorkflowBuilder("wf", "Agent Workflow")
            .add_start()
            .add_agent("processor", purpose="Process data", input_variable="data")
            .add_end()
            .connect("start", "processor")
            .connect("processor", "end")
            .build()
        )

        agent_node = workflow.get_node("processor")
        assert agent_node is not None
        assert agent_node.type == NodeType.AGENT
        assert agent_node.config["purpose"] == "Process data"

    def test_build_with_condition(self):
        """Test building workflow with condition branching."""
        workflow = (
            WorkflowBuilder("wf", "Conditional")
            .add_start()
            .add_condition("check", condition="value > 10")
            .add_transform("high", transform="passthrough")
            .add_transform("low", transform="passthrough")
            .add_end()
            .connect("start", "check")
            .connect("check", "high", condition="branch == 'true'")
            .connect("check", "low", condition="branch == 'false'")
            .connect("high", "end")
            .connect("low", "end")
            .build()
        )

        assert len(workflow.nodes) == 5
        assert len(workflow.edges) == 5

        # Check conditional edges
        check_edges = workflow.get_outgoing_edges("check")
        assert len(check_edges) == 2

    def test_build_with_human_node(self):
        """Test building workflow with human checkpoint."""
        workflow = (
            WorkflowBuilder("wf", "Human Workflow")
            .add_start()
            .add_human("approval", prompt="Please approve this action", options=["Approve", "Reject"])
            .add_end()
            .connect("start", "approval")
            .connect("approval", "end")
            .build()
        )

        human_node = workflow.get_node("approval")
        assert human_node is not None
        assert human_node.type == NodeType.HUMAN
        assert human_node.config["prompt"] == "Please approve this action"
        assert "Approve" in human_node.config["options"]


# =============================================================================
# Human-in-the-Loop Tests
# =============================================================================

class TestHumanInTheLoop:
    """Test human-in-the-loop workflow capabilities."""

    @pytest.mark.asyncio
    async def test_workflow_pauses_at_human_node(self):
        """Workflow should pause at human nodes."""
        workflow = (
            WorkflowBuilder("wf", "Human Test")
            .add_start()
            .add_human("review", prompt="Review this")
            .add_end()
            .connect("start", "review")
            .connect("review", "end")
            .build()
        )

        engine = WorkflowEngine()
        state = engine.launch(workflow)

        state = await engine.start(workflow, state)
        state = await engine.run_to_completion(workflow, state)

        # Should be waiting for human input
        assert state.status == WorkflowStatus.WAITING_HUMAN

    @pytest.mark.asyncio
    async def test_provide_human_response(self):
        """Test providing human response resumes workflow."""
        workflow = (
            WorkflowBuilder("wf", "Human Test")
            .add_start()
            .add_human("review", prompt="Review this")
            .add_end()
            .connect("start", "review")
            .connect("review", "end")
            .build()
        )

        engine = WorkflowEngine()
        state = engine.launch(workflow)

        state = await engine.start(workflow, state)
        state = await engine.run_to_completion(workflow, state)

        assert state.status == WorkflowStatus.WAITING_HUMAN

        # Provide human response
        state = await engine.provide_human_response(
            workflow,
            state,
            node_id="review",
            response="Approved",
        )

        state = await engine.run_to_completion(workflow, state)

        assert state.status == WorkflowStatus.COMPLETED
        assert state.get_variable("human_response") == "Approved"


# =============================================================================
# Conditional Workflow Tests
# =============================================================================

class TestConditionalWorkflows:
    """Test conditional branching in workflows."""

    @pytest.mark.asyncio
    async def test_condition_true_branch(self):
        """Test condition evaluates to true branch."""
        workflow = (
            WorkflowBuilder("wf", "Conditional")
            .add_start()
            .add_condition("check", condition="value > 5")
            .add_transform("high_path", transform="template", template="HIGH")
            .add_transform("low_path", transform="template", template="LOW")
            .add_end()
            .connect("start", "check")
            .connect("check", "high_path", condition="condition_result == True")
            .connect("check", "low_path", condition="condition_result == False")
            .connect("high_path", "end")
            .connect("low_path", "end")
            .build()
        )

        engine = WorkflowEngine()
        state = engine.launch(workflow)

        state = await engine.start(workflow, state, input_data={"value": 10})
        state = await engine.run_to_completion(workflow, state)

        assert state.status == WorkflowStatus.COMPLETED
        # High path should have been taken
        assert state.node_executions.get("high_path") is not None
        assert state.node_executions["high_path"].status == NodeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_condition_false_branch(self):
        """Test condition evaluates to false branch."""
        workflow = (
            WorkflowBuilder("wf", "Conditional")
            .add_start()
            .add_condition("check", condition="value > 5")
            .add_transform("high_path", transform="template", template="HIGH")
            .add_transform("low_path", transform="template", template="LOW")
            .add_end()
            .connect("start", "check")
            .connect("check", "high_path", condition="condition_result == True")
            .connect("check", "low_path", condition="condition_result == False")
            .connect("high_path", "end")
            .connect("low_path", "end")
            .build()
        )

        engine = WorkflowEngine()
        state = engine.launch(workflow)

        state = await engine.start(workflow, state, input_data={"value": 3})
        state = await engine.run_to_completion(workflow, state)

        assert state.status == WorkflowStatus.COMPLETED
        # Low path should have been taken
        assert state.node_executions.get("low_path") is not None
        assert state.node_executions["low_path"].status == NodeStatus.COMPLETED


# =============================================================================
# State Persistence Tests (Factor 5 & 6)
# =============================================================================

class TestStatePersistence:
    """Test workflow state can be persisted and restored."""

    @pytest.mark.asyncio
    async def test_pause_serialize_resume(self):
        """Test pausing, serializing, and resuming workflow."""
        workflow = (
            WorkflowBuilder("wf", "Pausable")
            .add_start()
            .add_transform("step1", transform="passthrough")
            .add_transform("step2", transform="passthrough")
            .add_end()
            .connect("start", "step1")
            .connect("step1", "step2")
            .connect("step2", "end")
            .build()
        )

        engine = WorkflowEngine()
        state = engine.launch(workflow)

        # Start and run one step
        state = await engine.start(workflow, state, input_data={"input": "test"})

        # Pause
        state.status = WorkflowStatus.PAUSED

        # Serialize
        serialized = state.to_dict()

        # "Later" - restore and resume
        restored_state = WorkflowState.from_dict(serialized)
        restored_state.status = WorkflowStatus.RUNNING

        # Continue execution
        final_state = await engine.run_to_completion(workflow, restored_state)

        assert final_state.status == WorkflowStatus.COMPLETED
