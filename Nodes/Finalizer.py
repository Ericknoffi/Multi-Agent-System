from pydantic import BaseModel
from typing import Literal, List
from .supervisor_node import AgentState


class CodeFile(BaseModel):
    filename: str
    content: str


class FinalResponse(BaseModel):
    title: str
    format_type: Literal["software", "general"]
    steps: List[str] = []
    tips: List[str] = []
    code: List[CodeFile] = []


def _format_final_response(response: FinalResponse) -> str:
    lines = [response.title, ""]

    if response.steps:
        lines.append("Results:")
        for index, step in enumerate(response.steps, start=1):
            lines.append(f"{index}. {step}")

    if response.tips:
        lines.append("")
        lines.append("Notes:")
        for tip in response.tips:
            lines.append(f"- {tip}")

    if response.code:
        lines.append("")
        lines.append("Code:")
        for cf in response.code:
            lines.append(f"\n--- {cf.filename} ---")
            lines.extend(cf.content.splitlines())

    return "\n".join(lines)


def _split_into_steps(text: str) -> list[str]:
    """
    Splits a long LLM result string into meaningful step-sized chunks.
    Tries paragraph splits first; falls back to sentence splitting for
    very long single-paragraph responses.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    
    chunks, group = [], []
    for s in sentences:
        group.append(s)
        if len(group) >= 3:
            chunks.append(" ".join(group))
            group = []
    if group:
        chunks.append(" ".join(group))
    return chunks or [text]


async def finalizer_node(state: AgentState):
    completed_tasks = [t for t in state["tasks"] if t["status"] == "completed"]
    failed_tasks = [t for t in state["tasks"] if t["status"] == "failed"]

    if not completed_tasks:
        error_note = ""
        if failed_tasks:
            error_note = (
                " The following tasks failed and produced no output: "
                + ", ".join(f"'{t['description'][:60]}'" for t in failed_tasks)
            )
        return {
            "final_response": (
                f"No results were produced for: {state['user_query']}.{error_note}"
            )
        }

  
    steps: list[str] = []
    code_files: list[CodeFile] = []

    for task in completed_tasks:
        result = task.get("result") or ""

        if isinstance(result, dict):
            # This is for future coder outputs.
            summary = result.get("summary", "")
            if summary:
                steps.extend(_split_into_steps(summary))
            for f in result.get("files", []):
                code_files.append(CodeFile(
                    filename=f.get("filename", "output.txt"),
                    content=f.get("content", ""),
                ))
        else:
            # Current Used method for all other outputs - just split into steps and add to the response.
            steps.extend(_split_into_steps(str(result)))

    format_type: Literal["software", "general"] = "software" if code_files else "general"
    title = "Implementation" if format_type == "software" else state.get("user_query", "Result")

   
    tips: list[str] = []
    if failed_tasks:
        tips.append(
            f"Note: {len(failed_tasks)} task(s) did not complete: "
            + ", ".join(f"'{t['description'][:50]}'" for t in failed_tasks)
        )
    if code_files:
        tips.append(f"{len(code_files)} code file(s) included above.")
    if state.get("errors"):
        tips.extend(state["errors"])

    response_obj = FinalResponse(
        title=title,
        format_type=format_type,
        steps=steps,
        tips=tips,
        code=code_files,
    )

    return {"final_response": _format_final_response(response_obj)}
