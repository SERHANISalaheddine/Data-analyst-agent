"""
=============================================================================
STATE.PY - Shared State Definition for the Multi-Agent System
=============================================================================

This module defines the shared state that flows through all agents in the
data analyst pipeline. Think of this as the "memory" or "context" that gets
passed from agent to agent, with each agent reading from it and writing to it.

WHY USE A TYPED STATE?
----------------------
In a multi-agent system, agents need to communicate with each other. Instead
of passing data directly between agents (which creates tight coupling), we use
a shared state object. This pattern:

1. Decouples agents: Each agent only needs to know about the state, not other agents
2. Provides type safety: TypedDict lets us define exactly what fields exist
3. Enables debugging: We can inspect the state at any point to see what's happening
4. Supports LangGraph: LangGraph uses this state pattern for graph-based workflows

HOW THE STATE FLOWS:
-------------------
User asks question    → mcp_sheets_agent uses tools to find & fetch data
                      → mcp_sheets_agent fills csv_path, sheet_metadata, mcp_tool_calls
                      → schema_agent fills dataframe_summary
                      → intent_agent fills parsed_intent
                      → code_writer_agent fills generated_code
                      → executor_agent fills execution_result & chart_json
                      → narrative_agent fills narrative
                      → critic_agent fills critique & critic_score
=============================================================================
"""

from typing import TypedDict, Optional, List


class AnalystState(TypedDict, total=False):
    """
    The shared state that carries all data between agents in the pipeline.
    
    This is a TypedDict, which means:
    - It's a dictionary at runtime (you access fields like state["csv_path"])
    - But it has type hints that your IDE and type checkers can use
    - The 'total=False' means all fields are optional (not all will be set at once)
    
    Each field below documents:
    - What it contains
    - Which agent WRITES to it (fills it in)
    - Which agent(s) READ from it (use it)
    """
    
    # =========================================================================
    # INPUT DATA - Set by the Streamlit app before the graph runs
    # =========================================================================
    
    user_question: str
    # The raw question typed by the user in the Streamlit interface.
    # This is the ONLY input the user provides - no URLs or sheet selection needed.
    # The MCP agent will autonomously find the relevant data based on this question.
    # Examples: "What are the top 5 products by sales?" or "Show me Q4 trends"
    # WRITTEN BY: Streamlit app (app.py) from the text input
    # READ BY: mcp_sheets_agent (to decide which data to fetch)
    #          intent_agent (to understand what the user wants)
    #          narrative_agent (to reference the original question in the explanation)
    #          critic_agent (to evaluate if the answer matches the question)
    
    # =========================================================================
    # MCP AGENT OUTPUT - Set by mcp_sheets_agent
    # =========================================================================
    
    csv_path: str
    # The filesystem path to the CSV file containing the data to analyze.
    # This is populated by the MCP agent after it autonomously fetches data.
    # WRITTEN BY: mcp_sheets_agent (saves fetched sheet data to temp CSV)
    # READ BY: schema_agent (to load and analyze the CSV)
    #          executor_agent (passed to the generated code for data loading)
    
    sheet_metadata: str
    # Human-readable information about what the MCP agent found and fetched.
    # Includes: which spreadsheet, which tab, number of rows/columns, column names.
    # WRITTEN BY: mcp_sheets_agent (describes what it retrieved)
    # READ BY: Streamlit app (to display info about the fetched data)
    # Example: "Data retrieved via MCP agent\nTool calls: 3\nSize: 1,234 rows × 5 columns"
    
    mcp_tool_calls: List[dict]
    # A log of which MCP tools the agent called, in order, with their arguments.
    # This is used to display the agent's reasoning process in the UI.
    # Each entry is a dict with: {"tool": str, "args": dict, "result": str}
    # WRITTEN BY: mcp_sheets_agent (logs each tool call during execution)
    # READ BY: Streamlit app (to display the "MCP Tool Calls" expander section)
    # Example: [{"tool": "search_spreadsheets", "args": {"query": "sales"}}, ...]
    
    # =========================================================================
    # SCHEMA UNDERSTANDING - Set by schema_agent
    # =========================================================================
    
    dataframe_summary: str
    # A human-readable text summary of the CSV structure including:
    # - Column names and their data types
    # - Number of rows in the dataset
    # - Number of null values per column
    # - A few sample rows to show what the data looks like
    #
    # WRITTEN BY: schema_agent (pure pandas logic, no LLM)
    # READ BY: intent_agent (to understand what columns exist when parsing intent)
    #          code_writer_agent (to know what columns to reference in code)
    #
    # WHY THIS EXISTS: LLMs need to know the data structure to write correct code.
    # Without this, the LLM might hallucinate column names that don't exist.
    
    # =========================================================================
    # INTENT PARSING - Set by intent_agent
    # =========================================================================
    
    parsed_intent: str
    # A cleaned, precise reformulation of the user's question.
    # The raw user question might be vague like "show me sales stuff".
    # This field contains a clear instruction like:
    # "Aggregate the 'sales' column by 'region', compute the sum, display as bar chart"
    #
    # WRITTEN BY: intent_agent (uses LLM to interpret the question)
    # READ BY: code_writer_agent (uses this precise instruction to generate code)
    #
    # WHY THIS EXISTS: This is "intent parsing" - a common NLP task where we
    # convert ambiguous natural language into structured, actionable instructions.
    # It reduces errors in downstream code generation.
    
    # =========================================================================
    # CODE GENERATION - Set by code_writer_agent
    # =========================================================================
    
    generated_code: str
    # The actual Python code that will be executed to answer the question.
    # This code will:
    # - Load the CSV with pandas
    # - Perform data manipulation (filtering, aggregation, etc.)
    # - Create a Plotly figure stored in a variable called 'fig'
    # - Optionally create a text result stored in 'result_text'
    #
    # WRITTEN BY: code_writer_agent (uses LLM to generate Python/Plotly code)
    # READ BY: executor_agent (executes this code using Python's exec())
    #
    # WHY THIS EXISTS: We separate code generation from execution for safety
    # and debugging. We can inspect the code before running it.
    
    # =========================================================================
    # EXECUTION RESULTS - Set by executor_agent
    # =========================================================================
    
    execution_result: str
    # The text output from running the generated code.
    # This might be:
    # - A printed table or summary statistics
    # - A message like "Chart generated successfully"
    # - An error message if execution failed
    #
    # WRITTEN BY: executor_agent (runs the code and captures output)
    # READ BY: narrative_agent (uses this to explain the results)
    #          critic_agent (uses this to evaluate if execution worked)
    
    chart_json: str
    # The Plotly figure serialized as a JSON string.
    # Plotly figures can be converted to JSON with fig.to_json(), which allows:
    # - Safe passing between agents/processes (JSON is serializable)
    # - Storage in state (can't store Python objects in some state backends)
    # - Rendering in Streamlit using plotly.io.from_json()
    #
    # WRITTEN BY: executor_agent (extracts 'fig' from executed code, converts to JSON)
    # READ BY: Streamlit app (converts back to figure and displays with st.plotly_chart)
    #
    # WHY JSON: Python objects like Plotly figures can't be easily serialized.
    # JSON is a universal format that can be stored, transmitted, and reconstructed.
    
    # =========================================================================
    # NARRATIVE OUTPUT - Set by narrative_agent
    # =========================================================================
    
    narrative: str
    # A plain English explanation of the results, written for non-technical users.
    # Example: "Based on your data, the North region had the highest sales at $1.2M,
    # which is 30% higher than the second-place South region."
    #
    # WRITTEN BY: narrative_agent (uses LLM to write human-friendly explanation)
    # READ BY: Streamlit app (displays this to the user)
    #          critic_agent (evaluates if this explanation addresses the question)
    #
    # WHY THIS EXISTS: This is "last mile" communication - even if we have perfect
    # data analysis, users need clear explanations to understand and act on insights.
    
    # =========================================================================
    # CRITIQUE & EVALUATION - Set by critic_agent
    # =========================================================================
    
    critique: str
    # A one-sentence explanation of WHY the answer passed or failed evaluation.
    # Example: "The analysis correctly identified the top 5 products as requested."
    # Or: "The analysis showed a pie chart but the user asked for a time trend."
    #
    # WRITTEN BY: critic_agent (uses LLM to evaluate the response quality)
    # READ BY: Streamlit app (displays this feedback to the user)
    
    critic_score: str
    # Either "PASS" or "FAIL" - a simple binary evaluation.
    # PASS = The final answer adequately addresses the user's original question.
    # FAIL = The answer missed the point or doesn't match what was asked.
    #
    # WRITTEN BY: critic_agent (extracts this from LLM's structured JSON response)
    # READ BY: Streamlit app (shows green checkmark for PASS, red X for FAIL)
    #
    # WHY THIS EXISTS: This is "LLM-as-judge" evaluation - using one LLM call to
    # evaluate the output of another LLM call. It's a common pattern for quality
    # assurance in LLM applications without human review.
