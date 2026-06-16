import asyncio
import shutil
import os
from langchain_mcp_adapters.client import MultiServerMCPClient

from Tools.filesystem import create_filesystem_client
from Tools.github import create_github_client
from Tools.web import create_fetch_client
from Tools.documentation import create_context7_client

RESEARCH_TOOLS: list = []
CODING_TOOLS: list = []
_tools_initialized = False
_active_clients: list[MultiServerMCPClient] = []


def _check_npx():
    if not shutil.which("npx"):
        raise RuntimeError(
            "npx not found on PATH. Install Node.js (https://nodejs.org) to use MCP tools."
        )


async def _get_tools_safe(client: MultiServerMCPClient, name: str, timeout: int = 30) -> list:
    """Get tools from a client with a timeout. Returns [] on any failure."""
    try:
        tools = await asyncio.wait_for(client.get_tools(), timeout=timeout)
        _active_clients.append(client) 
        print(f"[registry] {name}: loaded {len(tools)} tool(s)")
        return tools
    except asyncio.TimeoutError:
        print(f"[registry] Warning: {name} timed out after {timeout}s — skipping")
        return []
    except Exception as e:
        print(f"[registry] Warning: {name} failed to load — {e}")
        return []


async def initialize_tools():
    """
    Initialize all MCP tool groups. Must be awaited before the graph runs.
    - Checks for npx at startup
    - Each tool group has its own client (subprocess)
    - Clients are kept alive in _active_clients for the session lifetime
    - Individual failures don't block the others
    """
    global RESEARCH_TOOLS, CODING_TOOLS, _tools_initialized

    _check_npx()

    workspace = os.path.abspath("./workspace")
    os.makedirs(workspace, exist_ok=True)

    # Instantiate clients via Tools factories
    filesystem_research = create_filesystem_client(workspace)
    filesystem_coding = create_filesystem_client(workspace)

    github_research = create_github_client()
    github_coding = create_github_client()

    fetch_client = create_fetch_client()
    context7_client = create_context7_client()

    # Load all tool groups concurrently where possible
    fs_research_tools, fs_coding_tools, fetch_tools, context7_tools = await asyncio.gather(
        _get_tools_safe(filesystem_research, "filesystem[research]"),
        _get_tools_safe(filesystem_coding,  "filesystem[coding]"),
        _get_tools_safe(fetch_client,        "fetch[research]"),
        _get_tools_safe(context7_client,     "context7[coding]"),
    )

    github_research_tools = []
    github_coding_tools = []
    if github_research and github_coding:
        github_research_tools, github_coding_tools = await asyncio.gather(
            _get_tools_safe(github_research, "github[research]"),
            _get_tools_safe(github_coding,   "github[coding]"),
        )
    else:
        print("[registry] Warning: GITHUB_TOKEN not set — GitHub tools will be unavailable")

    RESEARCH_TOOLS = [*fs_research_tools, *github_research_tools, *fetch_tools]
    CODING_TOOLS   = [*fs_coding_tools,   *github_coding_tools,   *context7_tools]
    _tools_initialized = True

    print(f"[registry] Research tools ready: {len(RESEARCH_TOOLS)}")
    print(f"[registry] Coding tools ready:   {len(CODING_TOOLS)}")


async def shutdown_tools():
    """Call at app shutdown to clean up MCP subprocesses."""
    for client in _active_clients:
        try:
            await client.__aexit__(None, None, None)
        except Exception:
            pass
    _active_clients.clear()


def get_research_tools() -> list:
    if not _tools_initialized:
        raise RuntimeError(
            "Tools not initialized. Call `await registry.initialize_tools()` at startup."
        )
    return RESEARCH_TOOLS


def get_coding_tools() -> list:
    if not _tools_initialized:
        raise RuntimeError(
            "Tools not initialized. Call `await registry.initialize_tools()` at startup."
        )
    return CODING_TOOLS
