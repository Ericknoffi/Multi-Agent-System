from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
import sys
from Nodes.Coder import coder
from Nodes.Planner import planner
from Nodes.Researcher import researcher
from Nodes.Finalizer import finalizer_node
from Gateway.llm_gateway import LLMGateway
from Nodes.supervisor_node import AgentState, supervisor_node, supervisor_router
from wrapper import pii_middleware
import registry


load_dotenv()


graph = StateGraph(AgentState)
graph.add_node("planner", pii_middleware(planner))
graph.add_node("supervisor", pii_middleware(supervisor_node))
graph.add_node("research", pii_middleware(researcher))
graph.add_node("coding", pii_middleware(coder))
graph.add_node("finalizer", pii_middleware(finalizer_node))

graph.add_edge(START, "planner")
graph.add_edge("planner", "supervisor")
graph.add_edge("research", "supervisor")
graph.add_edge("coding", "supervisor")
graph.add_edge("finalizer", END)

graph.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "research": "research",
        "coding": "coding",
        "finalizer": "finalizer",
    },
)


async def main():
    # Initialize MCP tools before constructing/using models
    await registry.initialize_tools()
    app = graph.compile()

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    user_query = " ".join(sys.argv[1:]).strip()

    llm_gateway = LLMGateway(api_key=api_key)

    initial_state: AgentState = {
        "messages": [],
        "user_query": user_query,
        "llm_gateway": llm_gateway,
        "tasks": [],
        "current_task": None,
        "next_agent": None,
        "final_response": None,
        "iteration_count": 0,
        "errors": [],
        "planner_reasoning": None,
    }

    result = await app.ainvoke(initial_state)
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())