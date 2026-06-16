import asyncio
import os
import sys

# Ensure stdout uses UTF-8 to prevent UnicodeEncodeError on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from Nodes.Coder import coder
from Nodes.Planner import planner
from Nodes.Researcher import researcher
from Nodes.Finalizer import finalizer_node
from Gateway.llm_gateway import LLMGateway
from Nodes.supervisor_node import AgentState, supervisor_node, supervisor_router
from Middlewares.pii import redact_pii
from wrapper import pii_middleware
from utils import log
import registry

load_dotenv()   

# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

graph = StateGraph(AgentState)
graph.add_node("planner",    pii_middleware(planner))
graph.add_node("supervisor", pii_middleware(supervisor_node))
graph.add_node("research",   pii_middleware(researcher))
graph.add_node("coding",     pii_middleware(coder))
graph.add_node("finalizer",  pii_middleware(finalizer_node))

graph.add_edge(START,       "planner")
graph.add_edge("planner",   "supervisor")
graph.add_edge("research",  "supervisor")
graph.add_edge("coding",    "supervisor")
graph.add_edge("finalizer", END)

graph.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {"research": "research", "coding": "coding", "finalizer": "finalizer"},
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    # 1. Initialize MCP tools — must happen before any model call
    log("Initializing MCP tools")
    await registry.initialize_tools()
    log("MCP tools ready")

    # 2. Validate API key
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("Model Calling API_KEY is not set in environment.")

    # 3. Validate user query
    user_query = " ".join(sys.argv[1:]).strip()
    if not user_query:
        print("Usage: python main.py <your query>")
        sys.exit(1)
    if len(user_query) > 2000:
        print("Query too long. Maximum 2000 characters.")
        sys.exit(1)

    # 4. Redact PII once — wrapper.py does NOT redact user_query per-node
    user_query = redact_pii(user_query)

    # 5. Build graph and initial state
    app = graph.compile()
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

    # 6. Run with top-level timeout
    timeout_seconds = int(os.getenv("AGENT_TIMEOUT_SECONDS", "300"))
    log("Starting agent graph")

    try:
        result = await asyncio.wait_for(
            app.ainvoke(initial_state),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Agent timed out after {timeout_seconds}s. "
            "Raise AGENT_TIMEOUT_SECONDS or use a simpler prompt."
        ) from exc
    finally:
        # 7. Always clean up MCP subprocesses on exit
        log("Shutting down MCP tools")
        await registry.shutdown_tools()
        # Give Windows proactor event loop a short moment to clean up closed pipes to prevent closed pipe exceptions on shutdown
        await asyncio.sleep(0.2)

    log("Agent graph finished")
    print(result.get("final_response") or "(no response produced)")


if __name__ == "__main__":
    asyncio.run(main())
