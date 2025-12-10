"""
Workflow execution tasks.

Handles async workflow operations:
- Background workflow execution
- Long-running node execution
- Workflow scheduling
"""

from celery import shared_task

from app.celery import app


@app.task(bind=True, queue="workflow")
def execute_workflow(
    self,
    workflow_id: str,
    inputs: dict | None = None,
    user_id: str | None = None,
):
    """
    Execute a workflow asynchronously.

    Args:
        workflow_id: ID of the workflow to execute
        inputs: Input variables for the workflow
        user_id: ID of the user triggering the workflow

    Returns:
        dict with execution results
    """
    from app.services.workflow_service import WorkflowService

    service = WorkflowService()
    result = service.execute(workflow_id, inputs or {}, user_id)

    return {
        "workflow_id": workflow_id,
        "execution_id": result.get("execution_id"),
        "status": result.get("status", "completed"),
        "outputs": result.get("outputs", {}),
    }


@app.task(bind=True, queue="workflow")
def resume_workflow(self, execution_id: str, human_input: dict | None = None):
    """
    Resume a paused workflow (e.g., after human input).

    Args:
        execution_id: ID of the workflow execution to resume
        human_input: Human-provided input

    Returns:
        dict with execution results
    """
    from app.services.workflow_service import WorkflowService

    service = WorkflowService()
    result = service.resume(execution_id, human_input or {})

    return {
        "execution_id": execution_id,
        "status": result.get("status", "completed"),
        "outputs": result.get("outputs", {}),
    }


@app.task(bind=True, queue="workflow")
def cancel_workflow(self, execution_id: str):
    """
    Cancel a running workflow.

    Args:
        execution_id: ID of the workflow execution to cancel

    Returns:
        dict with cancellation status
    """
    from app.services.workflow_service import WorkflowService

    service = WorkflowService()
    result = service.cancel(execution_id)

    return {
        "execution_id": execution_id,
        "status": "cancelled",
        "message": result.get("message"),
    }


@app.task(bind=True, queue="workflow")
def execute_scheduled_workflow(self, schedule_id: str):
    """
    Execute a scheduled workflow.

    Args:
        schedule_id: ID of the workflow schedule

    Returns:
        dict with execution results
    """
    from app.services.schedule_service import ScheduleService

    service = ScheduleService()
    result = service.execute_scheduled(schedule_id)

    return {
        "schedule_id": schedule_id,
        "execution_id": result.get("execution_id"),
        "status": result.get("status", "completed"),
    }
