"""
=============================================================================
MCP/SHEETS_CLIENT.PY - MCP Tool Loader for Google Sheets (Local Server)
=============================================================================

This module connects to a locally running mcp-google-sheets MCP server and
loads all available tools as LangChain-compatible tools that can be bound
to an LLM for autonomous tool calling.

=============================================================================
TRANSPORT TYPES: STDIO vs HTTP/WebSocket
=============================================================================

This module uses STDIO transport to communicate with the MCP server.
Here's the difference between transport types:

STDIO TRANSPORT (Used Here):
----------------------------
- Launches the MCP server as a local subprocess
- Communicates via standard input/output pipes
- No network latency - everything runs locally
- More reliable for local development and demos
- Easier to debug - you can see the server process
- No external dependencies - works offline
- The MCP server (mcp-google-sheets) runs as a child process

HTTP/WebSocket TRANSPORT (e.g., Smithery):
------------------------------------------
- Connects to a remote MCP server over the network
- Uses SSE (Server-Sent Events) or WebSocket protocol
- May have network latency and connection issues
- Requires external service to be running and available
- Good for production deployments with managed infrastructure
- Server runs independently from your application

For local development and demos, STDIO is preferred because:
1. No external service dependencies
2. No network latency or connection failures
3. Easier to debug and test
4. Works without internet connection (Google API calls still need internet)

=============================================================================
THE OLD APPROACH VS THE NEW APPROACH
=============================================================================

PREVIOUS APPROACH (URL Adapter Pattern):
----------------------------------------
In the previous version, this module was an "adapter" that:
1. Took a Google Sheet URL from the user
2. Called specific, hardcoded MCP tools (read_spreadsheet, list_sheets)
3. Returned the data as a pandas DataFrame
4. The CODE decided which tools to call and in what order

The user had to know WHICH spreadsheet to analyze and provide its URL.
The agent was just a data pipeline - it had no autonomy.

NEW APPROACH (Tool Exposure Pattern):
-------------------------------------
In this version, we expose MCP tools DIRECTLY to the LLM:
1. We connect to the MCP server and load ALL available tools
2. We convert them to LangChain-compatible tool objects
3. We give these tools to the LLM via bind_tools()
4. The LLM decides which tools to call, when, and with what parameters

The user only asks a question. The AGENT decides which spreadsheet to look at.
The agent is autonomous - it makes decisions like a human would.

=============================================================================
WHAT IS "EXPOSING TOOLS TO THE LLM"?
=============================================================================

When we "expose tools to an LLM", we're giving the model the ability to
take actions in the real world. Here's what happens:

1. TOOL DEFINITION: Each tool has a name, description, and parameter schema.
   Example: "search_spreadsheets" - "Search for spreadsheets by name" - {query: str}

2. TOOL BINDING: We attach these tool definitions to the LLM using bind_tools().
   The LLM can now "see" what tools are available and what they do.

3. TOOL CALLING: When the LLM responds, instead of just text, it can output
   a structured tool call: {"tool": "search_spreadsheets", "args": {"query": "sales"}}

4. TOOL EXECUTION: We execute the tool with those arguments and get a result.

5. OBSERVATION: We send the result back to the LLM as an "observation".

6. REASONING: The LLM reads the observation and decides what to do next -
   maybe call another tool, or generate a final answer.

This is the essence of "agentic AI" - the model doesn't just generate text,
it takes actions, observes results, and reasons about what to do next.

=============================================================================
WHAT IS TOOL CALLING?
=============================================================================

Tool calling (also called "function calling") is a capability of modern LLMs
where the model can output structured requests to call external functions.

WITHOUT tool calling:
    User: "What's the weather in Tokyo?"
    LLM: "I don't have real-time data, but Tokyo typically has..."
    (The model can only use its training data)

WITH tool calling:
    User: "What's the weather in Tokyo?"
    LLM: <tool_call>{"name": "get_weather", "args": {"city": "Tokyo"}}</tool_call>
    System: [calls API, gets result] → "72°F, sunny"
    LLM: "The weather in Tokyo is currently 72°F and sunny."
    (The model can access real-time external data)

The LLM doesn't execute tools itself - it outputs a REQUEST to call a tool,
and the system executes it and feeds the result back.

=============================================================================
WHY THIS MAKES THE AGENT MORE AUTONOMOUS AND INTELLIGENT
=============================================================================

1. NO HARDCODING: We don't decide which tools to call - the LLM does.
   If the MCP server adds new tools, the agent automatically has access.

2. CONTEXTUAL DECISIONS: The LLM reads the user's question and decides
   which tools are relevant. "Show me Q4 sales" → search for sales sheets.

3. MULTI-STEP REASONING: The agent can chain tool calls intelligently:
   - First: search for spreadsheets matching "sales"
   - Then: list sheets in the most relevant one
   - Then: read the specific tab with Q4 data

4. ERROR RECOVERY: If one approach fails, the agent can try another.
   "Sheet not found" → try a different search query.

5. NATURAL INTERACTION: Users don't need to know spreadsheet IDs or URLs.
   They just ask questions in natural language.

This is the difference between a "pipeline" (fixed steps) and an "agent"
(dynamic decision-making based on context and observations).

=============================================================================
"""

import os
import asyncio
import threading
import concurrent.futures
from typing import List, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# ASYNC HELPER FOR RUNNING COROUTINES FROM SYNC CONTEXT
# =============================================================================
# The MCP adapter uses anyio/sniffio internally which requires a running
# async event loop. We create a dedicated background thread with its own
# event loop to handle all MCP async operations cleanly.
# =============================================================================

_async_loop = None
_async_thread = None

def _start_async_loop():
    """Start a dedicated event loop in a background thread."""
    global _async_loop
    _async_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_async_loop)
    _async_loop.run_forever()

def _get_async_loop():
    """Get the dedicated async event loop, starting it if needed."""
    global _async_loop, _async_thread
    if _async_loop is None or not _async_loop.is_running():
        _async_thread = threading.Thread(target=_start_async_loop, daemon=True)
        _async_thread.start()
        # Wait for the loop to start
        import time
        while _async_loop is None or not _async_loop.is_running():
            time.sleep(0.01)
    return _async_loop

def run_async(coro):
    """Run an async coroutine from sync code using the dedicated event loop."""
    loop = _get_async_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()

# =============================================================================
# CONFIGURATION
# =============================================================================
# These values configure the local mcp-google-sheets MCP server.
# SERVICE_ACCOUNT_PATH: Path to your Google Service Account JSON key file
# DRIVE_FOLDER_ID: The ID of your Google Drive folder containing spreadsheets
# =============================================================================

SERVICE_ACCOUNT_PATH = os.getenv("SERVICE_ACCOUNT_PATH", "./credentials/service_account.json")
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "")


def get_sheets_tools() -> List[Any]:
    """
    Connect to the local mcp-google-sheets MCP server and load all available tools.
    
    This function:
    1. Launches the mcp-google-sheets server as a local subprocess via uvx
    2. Discovers all available tools exposed by the MCP server
    3. Converts them to LangChain-compatible tool objects
    4. Returns them ready to be bound to an LLM with bind_tools()
    
    Returns:
    --------
    List[Any]
        A list of LangChain-compatible tool objects. Each tool can be:
        - Bound to an LLM with model.bind_tools(tools)
        - Used directly with tool.invoke(args)
        
    Example tools that might be returned (depends on MCP server):
    - search_spreadsheets: Search for spreadsheets by name
    - list_sheets: List all tabs in a spreadsheet  
    - read_spreadsheet: Read data from a specific sheet
    - get_cell_value: Get value of a specific cell
    - etc.
    
    Raises:
    -------
    RuntimeError
        If MCP connection fails or credentials are missing
        
    Example:
    --------
    >>> tools = get_sheets_tools()
    >>> print([t.name for t in tools])
    ['search_spreadsheets', 'list_sheets', 'read_spreadsheet', ...]
    >>> 
    >>> # Bind to LLM
    >>> from langchain_openai import ChatOpenAI
    >>> llm = ChatOpenAI(model="gpt-4")
    >>> llm_with_tools = llm.bind_tools(tools)
    """
    
    # =========================================================================
    # VALIDATE CONFIGURATION
    # =========================================================================
    
    if not SERVICE_ACCOUNT_PATH:
        raise RuntimeError(
            "SERVICE_ACCOUNT_PATH not configured. "
            "Please set it in your .env file. "
            "This should point to your Google Service Account JSON key file."
        )
    
    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        raise RuntimeError(
            f"Service account file not found at: {SERVICE_ACCOUNT_PATH}. "
            "Please download your Service Account JSON key from Google Cloud Console "
            "and place it at the configured path. See credentials/README.md for details."
        )
    
    if not DRIVE_FOLDER_ID:
        raise RuntimeError(
            "DRIVE_FOLDER_ID not configured. "
            "Please set it in your .env file. "
            "This is the ID from your Google Drive folder URL."
        )
    
    # =========================================================================
    # CONNECT TO MCP SERVER AND LOAD TOOLS
    # =========================================================================
    # We use langchain-mcp-adapters to connect to the MCP server via stdio.
    # The MultiServerMCPClient launches mcp-google-sheets as a subprocess
    # and communicates with it via stdin/stdout pipes.
    #
    # This library handles:
    # - Spawning the MCP server process
    # - Stdio communication with the MCP protocol
    # - Tool discovery via the MCP protocol
    # - Conversion to LangChain tool format
    # =========================================================================
    
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        
        # We need to keep the client alive for the duration of tool usage
        # Store it at module level so tools can access it
        global _mcp_client
        
        async def _get_tools():
            global _mcp_client
            # Create MCP client with local server configuration
            # The client spawns mcp-google-sheets via uvx and communicates via stdio
            # Note: As of langchain-mcp-adapters 0.1.0, MultiServerMCPClient
            # cannot be used as a context manager - use client.get_tools() directly
            _mcp_client = MultiServerMCPClient(
                {
                    "google-sheets": {
                        "command": "uvx",
                        "args": ["mcp-google-sheets@latest"],
                        "env": {
                            "SERVICE_ACCOUNT_PATH": SERVICE_ACCOUNT_PATH,
                            "DRIVE_FOLDER_ID": DRIVE_FOLDER_ID
                        },
                        "transport": "stdio"
                    }
                }
            )
            # Get all tools from the MCP server
            # This returns LangChain-compatible tool objects (async-only)
            return await _mcp_client.get_tools()
        
        # Run the async function using our dedicated background event loop
        # This avoids conflicts with Streamlit's event loop and ensures
        # proper async backend detection by sniffio/anyio
        async_tools = run_async(_get_tools())
        
        if not async_tools:
            raise RuntimeError(
                "No tools found on the MCP server. "
                "Check that mcp-google-sheets is properly installed and configured. "
                "Test with: uvx mcp-google-sheets@latest"
            )
        
        # =====================================================================
        # WRAP ASYNC TOOLS FOR SYNC INVOCATION
        # =====================================================================
        # The MCP adapter returns StructuredTool objects that only support
        # async invocation. Since our LangGraph workflow runs synchronously,
        # we need to wrap each tool to support sync calls.
        #
        # We use run_async() to execute the coroutine on our dedicated
        # background event loop, which has proper async backend support.
        #
        # IMPORTANT: We must use StructuredTool (not Tool) to preserve the
        # args_schema, since the LLM sends JSON-structured arguments.
        # =====================================================================
        
        from langchain_core.tools import StructuredTool
        
        def create_sync_wrapper(async_tool):
            """Create a sync-compatible wrapper for an async tool."""
            
            # Get the async invoke function
            async_invoke = async_tool.ainvoke
            
            def sync_invoke(**kwargs):
                """Sync wrapper that calls the async tool on the background loop."""
                # ainvoke expects a dict input for structured tools
                return run_async(async_invoke(kwargs))
            
            # Use StructuredTool to preserve the argument schema
            return StructuredTool(
                name=async_tool.name,
                description=async_tool.description,
                func=sync_invoke,
                args_schema=async_tool.args_schema if hasattr(async_tool, 'args_schema') else None
            )
        
        # Wrap all async tools
        tools = [create_sync_wrapper(t) for t in async_tools]
        
        return tools
        
    except ImportError:
        raise RuntimeError(
            "langchain-mcp-adapters package not installed. "
            "Install it with: pip install langchain-mcp-adapters"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to connect to MCP server: {str(e)}")


async def get_sheets_tools_async() -> List[Any]:
    """
    Async version of get_sheets_tools() for use in async contexts.
    
    This is the preferred method when running in an async environment
    (like inside a LangGraph node that's running async).
    
    Returns:
    --------
    List[Any]
        A list of LangChain-compatible tool objects
    """
    
    if not SERVICE_ACCOUNT_PATH:
        raise RuntimeError(
            "SERVICE_ACCOUNT_PATH not configured. "
            "Please set it in your .env file."
        )
    
    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        raise RuntimeError(
            f"Service account file not found at: {SERVICE_ACCOUNT_PATH}. "
            "Please download your Service Account JSON key from Google Cloud Console."
        )
    
    if not DRIVE_FOLDER_ID:
        raise RuntimeError(
            "DRIVE_FOLDER_ID not configured. "
            "Please set it in your .env file."
        )
    
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        
        # Note: As of langchain-mcp-adapters 0.1.0, MultiServerMCPClient
        # cannot be used as a context manager - use client.get_tools() directly
        client = MultiServerMCPClient(
            {
                "google-sheets": {
                    "command": "uvx",
                    "args": ["mcp-google-sheets@latest"],
                    "env": {
                        "SERVICE_ACCOUNT_PATH": SERVICE_ACCOUNT_PATH,
                        "DRIVE_FOLDER_ID": DRIVE_FOLDER_ID
                    },
                    "transport": "stdio"
                }
            }
        )
        tools = await client.get_tools()
        
        if not tools:
            raise RuntimeError(
                "No tools found on the MCP server. "
                "Check that mcp-google-sheets is properly installed and configured."
            )
        
        return tools
            
    except ImportError:
        raise RuntimeError(
            "langchain-mcp-adapters package not installed. "
            "Install it with: pip install langchain-mcp-adapters"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to connect to MCP server: {str(e)}")
