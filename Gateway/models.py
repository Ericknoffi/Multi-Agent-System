from dataclasses import dataclass
from enum import StrEnum

class ModelRole(StrEnum):
    PLANNER = "planner"
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    RETRIEVER = "retriever"
    CODING = "coding"
    FINALIZER = "finalizer"


@dataclass(frozen=True)
class ModelConfig:
    model: str
    temperature: float = 0.0
    timeout: int = 60
    max_retries: int = 3 
    fallback_model: str | None = None
    tool_group: str | None = None   

MODEL_REGISTRY: dict[ModelRole, ModelConfig] = {
    ModelRole.PLANNER: ModelConfig(
        model="gpt-4o", 
        temperature=0.0,
        fallback_model="gpt-4o-2024-08-06"
        ),
    ModelRole.SUPERVISOR: ModelConfig(
        model="gpt-4o", 
        temperature=0.0,
        fallback_model="gpt-4o-2024-08-06"
        ),
    ModelRole.RESEARCHER: ModelConfig(
        model="gpt-4o",
        temperature=0.0,
        fallback_model="gpt-4o-2024-08-06",
        tool_group="research"
        ),
    ModelRole.RETRIEVER: ModelConfig(
        model="gpt-4o", 
        temperature=0.0,
        fallback_model="gpt-4o-2024-08-06",
        ),
    ModelRole.CODING: ModelConfig(
        model="gpt-4o", 
        temperature=0.0,
        fallback_model="gpt-4o-2024-08-06",
        tool_group="coding"
    ),
    ModelRole.FINALIZER: ModelConfig(
        model="gpt-4o", 
        temperature=0.0,
        fallback_model="gpt-4o-2024-08-06"
        )
}