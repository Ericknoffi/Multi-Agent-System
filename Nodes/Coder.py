from pydantic import BaseModel
from config import CODER_PROMPT
from Gateway.models import ModelRole
from Gateway.llm_gateway import LLMGateway
from .supervisor_node import AgentState
from langchain_core.messages import SystemMessage,HumanMessage


class CodingResult(BaseModel):
    summary: str
    code: str
    language: str
    explanation: str

def get_current_task(state):

    task_id = state["current_task"]

    for task in state["tasks"]:
        if task["id"] == task_id:
            return task
    raise ValueError(
        f"Task with id {task_id} not found in state."
    )

async def coder(state: AgentState):

    coder_llm = state["llm_gateway"].get_model(ModelRole.CODING)

    task = get_current_task(state)

    response = await(
        coder_llm.with_structured_output(CodingResult).ainvoke(
            [
                SystemMessage(
                    content=CODER_PROMPT
                ),
                HumanMessage(
                    content=task["description"]
                )
            ]
        )
    )

    updated_tasks = []

    for t in state["tasks"]:

        if t["id"] == task["id"]:
            updated_tasks.append({**t, "status": "completed", "result": response.model_dump()})
        else:
            updated_tasks.append(t)

    return {
        "tasks": updated_tasks
    }