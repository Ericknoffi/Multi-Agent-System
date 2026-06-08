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
        model="openai/gpt-oss-120b:free", 
        temperature=0.0,
        fallback_model="nvidia/nemotron-3-ultra-550b-a55b:free"
        ),
    ModelRole.SUPERVISOR: ModelConfig(
        model="openai/gpt-oss-120b:free", 
        temperature=0.0,
        fallback_model="nvidia/nemotron-3-ultra-550b-a55b:free"
        ),
    ModelRole.RESEARCHER: ModelConfig(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        temperature=0.0,
        fallback_model="moonshotai/kimi-k2.6:free",
        tool_group="research"
        ),
    ModelRole.RETRIEVER: ModelConfig(
        model="openai/gpt-oss-20b:free", 
        temperature=0.0,
        fallback_model="nvidia/nemotron-nano-9b-v2:free",
        ),
    ModelRole.CODING: ModelConfig(
        model="poolside/laguna-m.1:free", 
        temperature=0.0,
        fallback_model="moonshotai/kimi-k2.6:free",
        tool_group="coding"
    ),
    ModelRole.FINALIZER: ModelConfig(
        model="z-ai/glm-4.5-air:free", 
        temperature=0.0,
        fallback_model="openai/gpt-oss-20b:free"
        )
}