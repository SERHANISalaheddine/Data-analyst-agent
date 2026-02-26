"""
=============================================================================
AGENTS/MCP_SHEETS_AGENT.PY - Autonomous Google Sheets Data Retrieval Agent
=============================================================================

This agent uses MCP tools exposed directly to an LLM to autonomously find
and fetch data from Google Sheets. Unlike traditional pipelines where code
decides what to do, here the LLM decides which tools to call based on the
user's question.

=============================================================================
WHAT IS THE REACT AGENT PATTERN?
=============================================================================

ReAct stands for "Reasoning and Acting" - a pattern where an LLM alternates
between thinking about what to do and taking actions (calling tools).

The pattern looks like this:

    THOUGHT: "The user wants Q4 sales data. I should search for spreadsheets
              containing 'sales' in the name."
    ACTION:  search_spreadsheets(query="Q4 sales")
    OBSERVATION: [{"id": "abc123", "name": "2024 Q4 Sales Report"}, ...]
    
    THOUGHT: "Found a relevant spreadsheet. Let me see what tabs it has."
    ACTION:  list_sheets(spreadsheet_id="abc123")
    OBSERVATION: ["Summary", "Raw Data", "Charts"]
    
    THOUGHT: "The 'Raw Data' tab likely has the data I need. Let me fetch it."
    ACTION:  read_spreadsheet(spreadsheet_id="abc123", sheet="Raw Data")
    OBSERVATION: [["Date", "Product", "Sales"], ["2024-10-01", "Widget", 500], ...]
    
    THOUGHT: "I have the data. Let me return it as CSV format."
    FINAL ANSWER: "Date,Product,Sales\n2024-10-01,Widget,500\n..."

The key insight is that the LLM DECIDES each step based on observations.
It's not following a script - it's reasoning through the problem.

=============================================================================
WHAT DOES A TOOL-CALLING LOOP LOOK LIKE STEP BY STEP?
=============================================================================

1. PROMPT: We send the user's question + available tools to the LLM

2. LLM RESPONSE: The LLM either:
   a) Returns a tool call request: {"tool": "search_spreadsheets", "args": {...}}
   b) Returns a final answer (no more tools needed)

3. IF TOOL CALL:
   a) We execute the tool with the provided arguments
   b) We get a result (the "observation")
   c) We add the observation to the conversation history
   d) We send everything back to the LLM (go to step 2)

4. IF FINAL ANSWER:
   a) We extract the answer and exit the loop
   b) This is the data we'll save to CSV

This loop continues until the LLM decides it has enough information to
answer the question, or until we hit a maximum iteration limit.

=============================================================================
WHY GIVE THE AGENT AUTONOMY INSTEAD OF EXPLICIT INSTRUCTIONS?
=============================================================================

1. FLEXIBILITY: The agent can handle questions we didn't anticipate.
   "What's in my most recent spreadsheet?" - figures it out on its own.

2. CONTEXT-AWARENESS: The agent reads tool outputs and adapts its strategy.
   If a search returns nothing, it might try a different search term.

3. ERROR RECOVERY: The agent can recover from mistakes autonomously.
   "Spreadsheet not found" -> "Let me try searching differently"

4. NATURAL LANGUAGE: Users ask questions naturally without knowing the
   underlying data structure. The agent bridges that gap.

5. FUTURE-PROOF: If the MCP server adds new tools, the agent can use them
   without code changes - it discovers tools dynamically.

The tradeoff is less predictability - the agent might take unexpected paths.
We mitigate this by logging all tool calls for transparency.

=============================================================================
"""

import os
import pandas as pd
from typing import Dict, Any, List
from io import StringIO
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent

from state import AnalystState
from sheets_mcp.sheets_client import get_sheets_tools

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
# BASE_URL allows using alternative OpenAI-compatible APIs (e.g., Azure, local LLMs)
# OPENAI_API_KEY is your authentication token for the API
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Directory for temporary files
TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# =============================================================================
# SYSTEM PROMPT FOR THE DATA RETRIEVAL AGENT
# =============================================================================
# This prompt tells the LLM what its role is and how to behave.
# Note that we give it AUTONOMY - we tell it to make decisions without
# asking the user for clarification. This is key to the agent pattern.
# =============================================================================

DATA_RETRIEVAL_SYSTEM_PROMPT = """You are a data retrieval agent with access to Google Sheets tools.

The user wants to analyze some data. Your job is to:
1. Find the most relevant spreadsheet based on the user's question
2. Identify the right sheet tab
3. Fetch the data
4. Return it as a clean CSV string

IMPORTANT GUIDELINES:
- Do not ask the user for clarification. Make your best judgment and retrieve the data autonomously.
- If you're unsure which spreadsheet to use, search for keywords from the user's question.
- If a spreadsheet has multiple tabs, choose the one most relevant to the question.
- Always return the final data as a CSV-formatted string with headers in the first row.
- If you cannot find relevant data, explain what you searched for and what was available.

After fetching the data, output ONLY the CSV data with no additional text or explanation.
The CSV should have headers in the first row followed by data rows.
"""


def mcp_sheets_agent(state: AnalystState) -> Dict[str, Any]:
    """
    Autonomous agent that uses MCP tools to find and fetch Google Sheets data.
    
    This agent:
    1. Loads all available MCP tools for Google Sheets
    2. Creates a ReAct agent with those tools bound to the LLM
    3. Gives the agent only the user's question - no URLs or sheet names
    4. The agent autonomously decides which spreadsheets to search/read
    5. Extracts the fetched data, saves to CSV, and updates state
    
    Inputs (from state):
    --------------------
    - user_question: str
        The question the user asked. This is the ONLY input the agent gets.
        Example: "Show me the Q4 sales data" or "What's in my inventory spreadsheet?"
    
    Outputs (written to state):
    ---------------------------
    - csv_path: str
        Path to the temp CSV file containing the fetched data
    - sheet_metadata: str
        Description of what the agent found (which spreadsheet, which tab)
    - mcp_tool_calls: list
        Log of which MCP tools the agent called, in order, for UI display
    
    Parameters:
    -----------
    state : AnalystState
        The current pipeline state
        
    Returns:
    --------
    dict
        State updates containing csv_path, sheet_metadata, and mcp_tool_calls
    """
    
    # =========================================================================
    # STEP 1: Extract user question from state
    # =========================================================================
    
    user_question = state.get("user_question", "")
    
    if not user_question:
        raise RuntimeError(
            "No user question provided. "
            "Please enter a question about your data."
        )
    
    # =========================================================================
    # STEP 2: Load MCP tools from Smithery
    # =========================================================================
    # This connects to the Smithery MCP server and loads all available tools.
    # Each tool becomes callable by the LLM during the agent loop.
    # =========================================================================
    
    try:
        tools = get_sheets_tools()
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load MCP tools: {str(e)}")
    
    # Log available tools for debugging
    tool_names = [t.name for t in tools]
    
    # =========================================================================
    # STEP 3: Create the ReAct agent
    # =========================================================================
    # We use LangGraph's create_react_agent which implements the ReAct pattern:
    # - Binds tools to the LLM
    # - Implements the thought-action-observation loop
    # - Handles tool execution and response parsing
    # =========================================================================
    
    # Initialize the LLM with tool-calling capabilities
    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key=OPENAI_API_KEY,
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),  # Using GPT-4o for best tool-calling performance
        temperature=0    # Low temperature for consistent tool calls
    )
    
    # Create the ReAct agent with MCP tools
    # This binds the tools to the LLM and sets up the agent loop
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=DATA_RETRIEVAL_SYSTEM_PROMPT
    )
    
    # =========================================================================
    # STEP 4: Run the agent with only the user's question
    # =========================================================================
    # We invoke the agent with a single human message containing the question.
    # The agent will autonomously decide which tools to call.
    # =========================================================================
    
    # Prepare the input message
    input_messages = [HumanMessage(content=user_question)]
    
    try:
        # Run the agent
        # This starts the tool-calling loop and runs until the agent
        # decides it has enough information to answer
        result = agent.invoke({"messages": input_messages})
        
    except Exception as e:
        raise RuntimeError(f"Agent execution failed: {str(e)}")
    
    # =========================================================================
    # STEP 5: Extract tool calls for logging
    # =========================================================================
    # We log which tools the agent called so we can display this in the UI.
    # This is crucial for explainability - users can see the agent's reasoning.
    # =========================================================================
    
    mcp_tool_calls = []
    
    for message in result.get("messages", []):
        # Check if this message contains tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                mcp_tool_calls.append({
                    "tool": tool_call.get("name", "unknown"),
                    "args": tool_call.get("args", {})
                })
        
        # Also capture tool response messages
        if hasattr(message, "type") and message.type == "tool":
            # Find the corresponding tool call to update with result
            if mcp_tool_calls and "result" not in mcp_tool_calls[-1]:
                # Truncate long results for display
                content = str(message.content)
                if len(content) > 200:
                    content = content[:200] + "..."
                mcp_tool_calls[-1]["result"] = content
    
    # =========================================================================
    # STEP 6: Extract the final answer (CSV data)
    # =========================================================================
    # The last message from the agent should contain the CSV data.
    # We parse it and save it to a temp file.
    # =========================================================================
    
    # Get the last AI message (the final answer)
    final_message = None
    for message in reversed(result.get("messages", [])):
        if hasattr(message, "type") and message.type == "ai":
            final_message = message.content
            break
        elif hasattr(message, "content") and not hasattr(message, "tool_calls"):
            # AIMessage without tool_calls is a final answer
            if not (hasattr(message, "tool_calls") and message.tool_calls):
                final_message = message.content
                break
    
    if not final_message:
        raise RuntimeError(
            "Agent did not return any data. "
            "It may not have found relevant spreadsheets for your question."
        )
    
    # =========================================================================
    # STEP 7: Parse the CSV data and save to file
    # =========================================================================
    # The agent should return CSV-formatted data. We parse it into a DataFrame
    # and save it so downstream agents can use it exactly as before.
    # =========================================================================
    
    try:
        # Try to parse the response as CSV
        # Clean up the response - remove any markdown code fences if present
        csv_data = final_message.strip()
        if csv_data.startswith("```"):
            # Remove markdown code fences
            lines = csv_data.split("\n")
            csv_data = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        
        # Parse CSV string into DataFrame
        df = pd.read_csv(StringIO(csv_data))
        
    except Exception as e:
        # If parsing fails, the agent might have returned an error message
        # or non-CSV data. We'll create an empty DataFrame and include
        # the message as metadata.
        df = pd.DataFrame()
        sheet_metadata = f"Agent response (could not parse as CSV): {final_message[:500]}"
    
    # =========================================================================
    # STEP 8: Save to temp CSV file
    # =========================================================================
    
    csv_path = os.path.join(TEMP_DIR, "fetched_sheet.csv")
    
    if not df.empty:
        df.to_csv(csv_path, index=False, encoding="utf-8")
        
        # Build metadata string
        sheet_metadata = (
            f"Data retrieved via MCP agent\n"
            f"Tool calls made: {len(mcp_tool_calls)}\n"
            f"Size: {len(df):,} rows × {len(df.columns)} columns\n"
            f"Columns: {', '.join(df.columns[:10])}"
            + (f"... ({len(df.columns) - 10} more)" if len(df.columns) > 10 else "")
        )
    else:
        # Write empty file so downstream agents don't fail
        with open(csv_path, "w") as f:
            f.write("")
        
        if "sheet_metadata" not in locals():
            sheet_metadata = "No data retrieved"
    
    # =========================================================================
    # STEP 9: Return state updates
    # =========================================================================
    
    return {
        "csv_path": csv_path,
        "sheet_metadata": sheet_metadata,
        "mcp_tool_calls": mcp_tool_calls
    }
