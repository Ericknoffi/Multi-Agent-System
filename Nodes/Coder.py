import re
import json
from langchain_core.messages import SystemMessage, HumanMessage
from Gateway.models import ModelRole
from config import CODER_PROMPT
from .supervisor_node import AgentState
from ._utils import get_current_task, build_updated_tasks, get_prior_results


def _parse_coder_json(text: str) -> dict | None:
    """
    Parse the JSON object the coder prompt instructs the model to return.
    Extracts the JSON block using a robust strategy (checking markdown blocks first,
    then parsing from first '{' to last '}', then scanning backwards), allowing
    unescaped control characters/newlines with strict=False.
    """
    # 1. Try to find json code block first
    code_block_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        candidate = code_block_match.group(1).strip()
        try:
            parsed = json.loads(candidate, strict=False)
            if isinstance(parsed, dict) and any(k in parsed for k in ("summary", "run", "files")):
                return parsed
        except Exception:
            pass

    # 2. Try matching from first '{' to last '}'
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace+1]
        try:
            parsed = json.loads(candidate, strict=False)
            if isinstance(parsed, dict) and any(k in parsed for k in ("summary", "run", "files")):
                return parsed
        except Exception:
            pass

    # 3. Try step-by-step fallback scanning (if there are multiple { } blocks)
    first_brace = text.find('{')
    if first_brace != -1:
        idx = text.rfind('}')
        while idx > first_brace:
            candidate = text[first_brace:idx+1]
            try:
                parsed = json.loads(candidate, strict=False)
                if isinstance(parsed, dict) and any(k in parsed for k in ("summary", "run", "files")):
                    return parsed
            except Exception:
                pass
            idx = text.rfind('}', 0, idx)

    return None


def _clean_file_content(content: str) -> str:
    """
    If the file content itself is wrapped in markdown code fences by the model,
    strip them so it does not result in syntax errors on disk.
    """
    content = content.strip()
    match = re.match(r"^```[a-zA-Z0-9_-]*\n?(.*?)\n?```$", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content



async def coder(state: AgentState):
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
        agent = state["llm_gateway"].get_agent(ModelRole.CODING)

        response = await agent.ainvoke({
            "messages": [
                SystemMessage(content=CODER_PROMPT),
                HumanMessage(content=human_content),
            ]
        })

        messages = response.get("messages", [])
        last_message = messages[-1] if messages else None
        raw_text = ""

        if last_message:
            content = last_message.content
            if isinstance(content, list):
                raw_text = " ".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict)
                ).strip()
            else:
                raw_text = (content or "").strip()

            finish_reason = (
                last_message.response_metadata.get("finish_reason", "")
                if hasattr(last_message, "response_metadata") else ""
            )
            if finish_reason == "length":
                raw_text += "\n[Note: response was truncated due to token limit]"

        if not raw_text:
            raw_text = "Coder returned an empty response."


        parsed = _parse_coder_json(raw_text)

        # Automatically write the files to the workspace if returned in the JSON payload
        if isinstance(parsed, dict) and "files" in parsed:
            for f in parsed["files"]:
                filename = f.get("filename")
                content = f.get("content", "")
                if filename:
                    # Strip leading slashes to prevent absolute path errors on Windows
                    filename = filename.lstrip("/\\")
                    path = os.path.join(workspace_abs, filename)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    cleaned_content = _clean_file_content(content)
                    with open(path, "w", encoding="utf-8") as file_obj:
                        file_obj.write(cleaned_content)
                    print(f"[Coder Node] Automatically saved {filename} to workspace")

        final_result = parsed if parsed is not None else raw_text

        return {
            "tasks": build_updated_tasks(state, task["id"], "completed", final_result)
        }

    except Exception as e:
        return {
            "tasks": build_updated_tasks(state, task["id"], "failed", None),
            "errors": state.get("errors", []) + [f"Coder failed on task {task['id']!r}: {e}"],
        }
