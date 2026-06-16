from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.runnables import Runnable
from langgraph.prebuilt import create_react_agent
from .models import ModelRole, MODEL_REGISTRY
import registry


TOOL_GROUPS = {
    "research": registry.get_research_tools,   # accessor — raises if not initialized
    "coding": registry.get_coding_tools,
}


class LLMGateway:
    def __init__(self, api_key: str):
        self._api_key = SecretStr(api_key)     # never store as plain str
        self._model_cache: dict[ModelRole, ChatOpenAI] = {}
        self._agent_cache: dict[ModelRole, Runnable] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_model(self, role: ModelRole) -> Runnable:
        """
        Returns a (possibly tool-bound) ChatOpenAI instance.
        Result is cached — same object returned on every call for the same role.
        """
        if role not in self._model_cache:
            self._model_cache[role] = self._build_model(role)
        return self._model_cache[role]

    def get_agent(self, role: ModelRole) -> Runnable:
        """
        Returns a full ReAct agent (create_react_agent) for roles with tools.
        For roles without tools, falls back to a plain model invoke.
        Agents are cached per role.
        """
        if role not in self._agent_cache:
            self._agent_cache[role] = self._build_agent(role)
        return self._agent_cache[role]

    def __call__(self, role: ModelRole) -> Runnable:
        return self.get_model(role)

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _get_api_key(self, config) -> SecretStr:
        if config.provider and config.provider != "openai":
            return SecretStr("none")
        return self._api_key

    def _build_model(self, role: ModelRole) -> Runnable:
        config = MODEL_REGISTRY.get(role)
        if config is None:
            raise ValueError(
                f"No model config registered for role: {role!r}. "
                f"Available roles: {list(MODEL_REGISTRY.keys())}"
            )

        llm = ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            timeout=config.timeout,
            max_retries=config.max_retries,
            max_tokens=config.max_tokens,
            api_key=self._get_api_key(config),
            base_url=config.base_url,
        )

        if config.tool_group:
            tools_factory = TOOL_GROUPS.get(config.tool_group)
            if tools_factory:
                tools = tools_factory()
                if tools:
                    llm = llm.bind_tools(tools)

        return llm

    def _build_agent(self, role: ModelRole) -> Runnable:
        config = MODEL_REGISTRY.get(role)
        if config is None:
            raise ValueError(f"No model config registered for role: {role!r}")

        base_llm = ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            timeout=config.timeout,
            max_retries=config.max_retries,
            max_tokens=config.max_tokens,
            api_key=self._get_api_key(config),
            base_url=config.base_url,
        )

        tools: list = []
        if config.tool_group:
            tools_factory = TOOL_GROUPS.get(config.tool_group)
            if tools_factory:
                tools = tools_factory()

        if tools:
            return create_react_agent(base_llm, tools)

        return base_llm
