from pydantic import BaseModel
from langchain_core.messages import SystemMessage,HumanMessage
from Gateway.models import ModelRole
from config import RESEARCH_PROMPT
from .supervisor_node import AgentState


class ResearchResult(BaseModel):
    summary: str
    findings: list[str]
    confidence: float


def get_current_task(state):

    task_id = state["current_task"]

    for task in state["tasks"]:
        if task["id"] == task_id:
            return task
    raise ValueError(
        f"Task with id {task_id} not found in state."
    )


async def researcher(state: AgentState):

    researcher_llm = state["llm_gateway"].get_model(ModelRole.RESEARCHER)
    
    task = get_current_task(state)

    llm_response = await(
        researcher_llm.with_structured_output(ResearchResult).ainvoke(
            [
                SystemMessage(
                    content=RESEARCH_PROMPT
                ),
                HumanMessage(
                    content=task["description"]
                )
            ]
        )
    )

    updated_task = []

    for t in state["tasks"]:

        if t["id"] == task["id"]:
            updated_task.append({**t, "status": "completed", "result": llm_response.model_dump()})
        else:
            updated_task.append(t)
    return {
            "tasks": updated_task
    }