from pydantic import BaseModel
from langchain_core.messages import SystemMessage,HumanMessage
from Gateway.models import ModelRole
from .supervisor_node import AgentState
from config import FINALIZER_PROMPT

class FinalResponse(BaseModel):
    answer: str


def get_completed_tasks(tasks):
    return [task for task in tasks if task["status"] == "completed"]


async def finalizer_node(state: AgentState):

    finalizer_llm = state["llm_gateway"].get_model(ModelRole.FINALIZER)

    completed_tasks = get_completed_tasks(
        state["tasks"]
    )

    task_results = []

    for task in completed_tasks:

        task_results.append(
            {
                "task": task["description"],
                "result": task["result"]
            }
        )

    response = await (
        finalizer_llm
        .with_structured_output(
            FinalResponse
        )
        .ainvoke(
            [
                SystemMessage(
                    content=FINALIZER_PROMPT
                ),
                HumanMessage(
                    content=str(task_results)
                )
            ]
        )
    )

    return {
        "final_response":
            response.answer
    }