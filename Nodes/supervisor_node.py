from typing import Annotated, TypedDict, List, Literal, Any
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from Gateway.models import ModelRole
from config import SUPERVISOR_PROMPT


class Task(TypedDict):
    id: str
    description: str
    status: Literal["pending", "in_progress", "completed", "failed"]
    assigned_agent: str | None
    result: str | None
    retry_count: int          


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    llm_gateway: Any
    tasks: List[Task]
    current_task: str | None
    next_agent: str | None
    final_response: str | None
    iteration_count: int
    errors: List[str]
    planner_reasoning: str | None


# Maximum retries per individual task before it is force-failed
MAX_TASK_RETRIES = 2
# Maximum total supervisor iterations across the whole run
MAX_ITERATIONS = 25


async def supervisor_node(state: AgentState):

    if state["iteration_count"] >= MAX_ITERATIONS:
        return {"next_agent": "finish"}

    pending_tasks = [t for t in state["tasks"] if t["status"] == "pending"]

    failed_tasks = [t for t in state["tasks"] if t["status"] == "failed"]

    if not pending_tasks:
        errors = state.get("errors", [])
        if failed_tasks:
            errors = errors + [
                f"Task {t['id']!r} ({t['description'][:60]}) failed and was not completed."
                for t in failed_tasks
            ]
        return {
            "next_agent": "finish",
            "current_task": None,
            "errors": errors,
        }

    task = pending_tasks[0]
    fallback_errors: list[str] = []

    try:
        supervisor_llm = state["llm_gateway"].get_model(ModelRole.SUPERVISOR)

        class SupervisorDecision(BaseModel):
            action: Literal["none", "retry", "reassign", "stop"]
            task_id: str | None = None
            reassign_to: Literal["research", "coding", "finalizer"] | None = None
            note: str | None = None

        supervisor_chain = supervisor_llm.with_structured_output(SupervisorDecision)

        lines = [f"User request: {state['user_query']}", "Tasks:"]
        for t in state["tasks"]:
        
            result_preview = ""
            if t.get("result"):
                result_preview = str(t["result"])[:300].replace("\n", " ")
            lines.append(
                f"  id={t['id']} status={t['status']} "
                f"assigned={t.get('assigned_agent')} "
                f"retries={t.get('retry_count', 0)} "
                f"result_preview={result_preview!r}"
            )

        decision = await supervisor_chain.ainvoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content="\n".join(lines)),
        ])

        if decision.action == "stop":
            return {
                "next_agent": "finish",
                "current_task": None,
                "iteration_count": state["iteration_count"] + 1,
                "errors": state.get("errors", []) + ([decision.note] if decision.note else []),
            }

        if decision.action == "retry" and decision.task_id:
            target = next((t for t in state["tasks"] if t["id"] == decision.task_id), None)
            current_retries = target.get("retry_count", 0) if target else 0

            if current_retries >= MAX_TASK_RETRIES:
                # Exceeded retry budget — force-fail the task
                updated_tasks = [
                    {**t, "status": "failed"} if t["id"] == decision.task_id else t
                    for t in state["tasks"]
                ]
                has_pending = any(x["status"] == "pending" for x in updated_tasks)
                return {
                    "tasks": updated_tasks,
                    "next_agent": "finish" if not has_pending else task["assigned_agent"],
                    "current_task": None if not has_pending else task["id"],
                    "iteration_count": state["iteration_count"] + 1,
                    "errors": state.get("errors", []) + [
                        f"Task {decision.task_id!r} exceeded max retries ({MAX_TASK_RETRIES}) and was force-failed."
                    ],
                }

            updated_tasks = [
                {**t, "status": "pending", "result": None, "retry_count": current_retries + 1}
                if t["id"] == decision.task_id else t
                for t in state["tasks"]
            ]
            return {
                "tasks": updated_tasks,
                "current_task": decision.task_id,
                "next_agent": next(
                    (t["assigned_agent"] for t in updated_tasks if t["id"] == decision.task_id),
                    task["assigned_agent"],
                ),
                "iteration_count": state["iteration_count"] + 1,
            }

        if decision.action == "reassign" and decision.task_id and decision.reassign_to:
            updated_tasks = [
                {**t, "assigned_agent": decision.reassign_to, "status": "pending", "result": None}
                if t["id"] == decision.task_id else t
                for t in state["tasks"]
            ]
            return {
                "tasks": updated_tasks,
                "current_task": decision.task_id,
                "next_agent": decision.reassign_to,
                "iteration_count": state["iteration_count"] + 1,
            }

        # action == "none" or unrecognised — deterministic routing
        return {
            "current_task": task["id"],
            "next_agent": task["assigned_agent"],
            "iteration_count": state["iteration_count"] + 1,
        }

    except Exception as e:
        
        fallback_errors.append(f"Supervisor LLM failed (deterministic fallback): {e}")

    return {
        "current_task": task["id"],
        "next_agent": task["assigned_agent"],
        "iteration_count": state["iteration_count"] + 1,
        "errors": state.get("errors", []) + fallback_errors,
    }


def supervisor_router(state: AgentState) -> str:
    next_agent = state.get("next_agent")

    if next_agent in {"finish", None}:
        return "finalizer"

    if next_agent in {"research", "coding", "finalizer"}:
        return next_agent

    return "finalizer"
