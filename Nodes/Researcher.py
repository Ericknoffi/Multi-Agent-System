from langchain_core.messages import SystemMessage, HumanMessage
from Gateway.models import ModelRole
from config import RESEARCH_PROMPT
from .supervisor_node import AgentState
from ._utils import get_current_task, build_updated_tasks, get_prior_results


async def researcher(state: AgentState):
    task = get_current_task(state)

    import os
    workspace_abs = os.path.abspath("./workspace")
    prior = get_prior_results(state)
    human_content = (
        f"Absolute workspace directory: {workspace_abs}\n\n"
        f"Original request: {state['user_query']}\n\n"
        + (f"{prior}\n\n" if prior else "")
        + f"Your task: {task['description']}"
    )

    try:
        agent = state["llm_gateway"].get_agent(ModelRole.RESEARCHER)

        response = await agent.ainvoke({
            "messages": [
                SystemMessage(content=RESEARCH_PROMPT),
                HumanMessage(content=human_content),
            ]
        })
        messages = response.get("messages", [])
        last_message = messages[-1] if messages else None
        result = ""

        if last_message:
            content = last_message.content

            if isinstance(content, list):
                result = " ".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict)
                ).strip()
            else:
                result = (content or "").strip()

          
            finish_reason = (
                last_message.response_metadata.get("finish_reason", "")
                if hasattr(last_message, "response_metadata") else ""
            )
            if finish_reason == "length":
                result += "\n[Note: response was truncated due to token limit]"

        if not result:
            result = "Researcher returned an empty response."

        return {
            "tasks": build_updated_tasks(state, task["id"], "completed", result)
        }

    except Exception as e:
        return {
            "tasks": build_updated_tasks(state, task["id"], "failed", None),
            "errors": state.get("errors", []) + [f"Researcher failed on task {task['id']!r}: {e}"],
        }
