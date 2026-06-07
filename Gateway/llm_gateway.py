from langchain_openrouter import ChatOpenRouter
from .models import ModelRole, MODEL_REGISTRY
from .. import registry

TOOL_GROUPS = {
    "research": lambda: registry.RESEARCH_TOOLS,
    "coding": lambda: registry.CODING_TOOLS,
}

class LLMGateway:
    def __init__(self,api_key: str):
        self.api_key = api_key
    
    def get_model(self, role: ModelRole) -> ChatOpenRouter:

        config = MODEL_REGISTRY[role]

        llm =  ChatOpenRouter(
            model=config.model,
            temperature=config.temperature,
            timeout=config.timeout,
            max_retries=config.max_retries,
            api_key=self.api_key,
            fallback_model=config.fallback_model
        )
        
        if config.tool_group:

            tools_factory = TOOL_GROUPS.get(
                config.tool_group,
                lambda: []
            )
            tools = tools_factory()

            if tools:
                llm = llm.bind_tools(tools)

        return llm
    
    def __call__(self,role:ModelRole) -> ChatOpenRouter:
        return self.get_model(role=role)