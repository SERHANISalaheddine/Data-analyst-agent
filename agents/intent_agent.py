"""
=============================================================================
INTENT_AGENT.PY - User Intent Parsing Agent
=============================================================================

This module contains the intent_agent, which is the SECOND agent in the
multi-agent pipeline. Its job is to understand what the user is really
asking and translate their natural language question into a precise,
actionable data analysis instruction.

ROLE IN THE PIPELINE:
--------------------
schema_agent → [INTENT_AGENT] → code_writer_agent → executor_agent → ...

WHY INTENT PARSING IS CRUCIAL:
-----------------------------
Users express questions naturally, but natural language is often ambiguous:

User says: "Show me sales"
They might mean:
- Total sales (single number)
- Sales by product (bar chart)
- Sales over time (line chart)
- Sales by region (pie chart)
- The first 10 rows of sales data (table)

Without clarification, a code-writing LLM has to guess. This leads to:
- Wrong chart types
- Wrong aggregations
- Missing filters
- Frustrated users

WHAT THIS AGENT DOES:
--------------------
1. Takes the raw user question
2. Looks at the dataset structure (from schema_agent)
3. Uses an LLM to rewrite the question as a precise instruction
4. Specifies: columns involved, operation needed, output format

EXAMPLE TRANSFORMATION:
----------------------
Input:  "how are sales doing?"
Output: "Create a line chart showing the sum of 'total_sales' column 
        grouped by 'date' (monthly aggregation) to visualize the sales 
        trend. Use the 'date' column for x-axis and sum of 'total_sales' 
        for y-axis. Sort by date ascending."

This precise instruction is much easier for the code_writer_agent to handle.

=============================================================================
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from state import AnalystState
from prompts.prompts import INTENT_PARSER_PROMPT

# Load environment variables from .env file
# This allows us to keep API keys out of the code
load_dotenv()

# Get configuration from environment variables
# BASE_URL allows using alternative OpenAI-compatible APIs (e.g., Azure, local LLMs)
# OPENAI_API_KEY is your authentication token for the API
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def intent_agent(state: AnalystState) -> dict:
    """
    Parse the user's question into a precise data analysis instruction.
    
    This agent uses an LLM to understand what the user is asking and
    translates it into an unambiguous instruction that the code_writer_agent
    can use to generate accurate code.
    
    Parameters:
    -----------
    state : AnalystState
        The shared state dictionary. This agent reads from:
        - state["user_question"]: The raw question from the user
        - state["dataframe_summary"]: The dataset structure from schema_agent
        
    Returns:
    --------
    dict
        A dictionary with updates to apply to the state:
        - "parsed_intent": The precise, actionable interpretation of the question
        
    Why This Two-Step Approach?
    --------------------------
    We could try to go directly from user question to code, but splitting it:
    1. Makes each step simpler and more reliable
    2. Allows us to inspect and validate the parsed intent
    3. Creates a natural "checkpoint" for debugging
    4. Improves overall accuracy (divide and conquer)
    """
    
    # =========================================================================
    # STEP 1: Extract inputs from state
    # =========================================================================
    # We need two pieces of information:
    # - The user's raw question (what they typed)
    # - The dataset structure (so the LLM knows what columns exist)
    #
    # The LLM needs the dataset structure to suggest the right columns and
    # operations. Without it, the LLM might reference columns that don't exist.
    # =========================================================================
    
    user_question = state["user_question"]
    dataframe_summary = state["dataframe_summary"]
    
    # =========================================================================
    # STEP 2: Initialize the LLM client
    # =========================================================================
    # We use LangChain's ChatOpenAI class to interact with OpenAI's API.
    # 
    # KEY PARAMETERS:
    # - model: We use gpt-4o (or whatever is in OPENAI_MODEL env var) for
    #   better understanding of nuanced questions. GPT-3.5 works too but
    #   may produce less precise intent parsing.
    # - temperature: 0 means deterministic output (same input = same output).
    #   For intent parsing, we want consistency, not creativity.
    #
    # The API key is loaded from the environment variable OPENAI_API_KEY.
    # =========================================================================
    
    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key=OPENAI_API_KEY,
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),  # Default to gpt-4o
        temperature=0  # Deterministic output for consistency
    )
    
    # =========================================================================
    # STEP 3: Build the prompt
    # =========================================================================
    # We use a template from prompts.py and fill in the dynamic values.
    # The template includes:
    # - Instructions on how to parse the intent
    # - The dataset structure
    # - The user's question
    # - Examples of good output
    #
    # Using .format() lets us inject the actual values into the template.
    # =========================================================================
    
    prompt = INTENT_PARSER_PROMPT.format(
        dataframe_summary=dataframe_summary,
        user_question=user_question
    )
    
    # =========================================================================
    # STEP 4: Call the LLM
    # =========================================================================
    # LangChain uses a message-based API similar to OpenAI's chat API.
    # We wrap our prompt in a HumanMessage and invoke the LLM.
    #
    # The invoke() method:
    # - Sends the message to the OpenAI API
    # - Waits for the response
    # - Returns an AIMessage with the response content
    #
    # We use .content to extract just the text response.
    # =========================================================================
    
    response = llm.invoke([HumanMessage(content=prompt)])
    parsed_intent = response.content
    
    # =========================================================================
    # STEP 5: Return the parsed intent
    # =========================================================================
    # The parsed_intent now contains a precise instruction like:
    # "Calculate the sum of 'revenue' grouped by 'product_category' and 
    #  display as a horizontal bar chart, sorted descending by revenue."
    #
    # This will be used by code_writer_agent to generate Python code.
    # =========================================================================
    
    return {"parsed_intent": parsed_intent}


# =============================================================================
# UNDERSTANDING INTENT PARSING
# =============================================================================
#
# WHAT IS INTENT PARSING?
#
# Intent parsing (also called intent detection or intent classification) is a
# core NLP task where we determine what a user wants to accomplish from their
# natural language input. It's used in:
# - Voice assistants (Alexa, Siri) - "Play music" → intent: PLAY_MUSIC
# - Chatbots - "I want to return my order" → intent: RETURN_ORDER
# - Search engines - Understanding what users are searching for
#
# In our data analyst, we use intent parsing to:
# 1. Identify the analytical operation (aggregation, filtering, trend, etc.)
# 2. Identify the columns involved
# 3. Identify the desired output format (chart type, table, number)
#
# WHY NOT GO DIRECTLY FROM QUESTION TO CODE?
#
# We could send the question directly to a code-writing LLM, but:
# 1. The task becomes too complex (understand + code in one step)
# 2. Errors are harder to debug (is the understanding wrong or the code?)
# 3. We lose the ability to verify understanding before acting
#
# By separating intent parsing, we can inspect the parsed_intent and verify
# that the system understood correctly before generating code.
#
# =============================================================================
