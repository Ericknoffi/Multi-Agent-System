from typing import Annotated, TypedDict, List, Literal, Any
from langchain_core.messages import BaseMessage, add_message

class Task(TypedDict):
    id: str
    description: str
    status: Literal["pending", "in_progress", "completed", "failed"]
    assigned_agent: str | None
    result: str | None


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_message]
    user_query: str
    llm_gateway: Any
    tasks: List[Task]
    current_task: str | None
    next_agent: str | None
    final_response: str | None
    iteration_count: int
    errors: List[str]
    planner_reasoning: str | None


async def supervisor_node(state: AgentState):

    MAX_ITERATIONS = 25

    if(
        state["iteration_count"] >= MAX_ITERATIONS
    ):
        return {
            "next_agent": "finish"
        }
    
    pending_tasks = [
        task for task in state["tasks"] if task["status"] == "pending"
    ]

    if not pending_tasks:
        return {
            "next_agent": "finish",
            "current_task": None
        }
    
    task = pending_tasks[0]

    return {
        "current_task": task["id"],
        "next_agent": task["assigned_agent"],
        "iteration_count": state["iteration_count"] + 1
    }


def supervisor_router(
    state: AgentState,
) -> str:

    next_agent = state.get("next_agent")

    if next_agent not in {
        "research",
        "coding",
        "finalizer",
    }:
        return "finalizer"

    return next_agent