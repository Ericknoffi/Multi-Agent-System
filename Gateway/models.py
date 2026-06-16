from dataclasses import dataclass
from enum import StrEnum


class ModelRole(StrEnum):
    PLANNER = "planner"
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    CODING = "coding"
    FINALIZER = "finalizer"   # reserved — finalizer is currently deterministic


@dataclass(frozen=True)
class ModelConfig:
    model: str
    temperature: float = 0.0
    timeout: int = 60
    max_retries: int = 3
    max_tokens: int = 2048
    tool_group: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    provider: str | None = None


MODEL_REGISTRY: dict[ModelRole, ModelConfig] = {

    ModelRole.PLANNER: ModelConfig(
        provider='ollama',
        base_url="http://localhost:11434/v1",
        model="cieloforge/qwen2.5-coder-7b-instruct-spec:latest",
        temperature=0.0,
        timeout=60,
        max_tokens=800,
        max_retries=3,
    ),

    ModelRole.SUPERVISOR: ModelConfig(
        provider='ollama',
        base_url="http://localhost:11434/v1",
        model="cieloforge/qwen2.5-coder-7b-instruct-spec:latest",
        temperature=0.0,
        timeout=60,
        max_tokens=400,
        max_retries=3,
    ),

    ModelRole.RESEARCHER: ModelConfig(
        model="openai/gpt-oss-120b:free",
        temperature=0.0,
        timeout=120,
        max_tokens=4096,
        tool_group="research",
    ),

    ModelRole.CODING: ModelConfig(
        model="nex-agi/nex-n2-pro:free",
        temperature=0.0,
        timeout=120,
        max_tokens=16000,
        tool_group="coding",
    ),
    # Current Finalizer agent is fully deterministic, Finalizer role model is not yet used in this project version.
    ModelRole.FINALIZER: ModelConfig(
        provider='ollama',
        base_url="http://localhost:11434/v1",
        model="gemma4:e4b",
        temperature=0.0,
        timeout=120,
        max_tokens=8192,
    ),
}
