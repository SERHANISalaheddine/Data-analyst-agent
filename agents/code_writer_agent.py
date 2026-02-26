"""
=============================================================================
CODE_WRITER_AGENT.PY - Python Code Generation Agent
=============================================================================

This module contains the code_writer_agent, which is the THIRD agent in the
multi-agent pipeline. Its job is to generate executable Python code that
performs the data analysis requested by the user.

ROLE IN THE PIPELINE:
--------------------
schema_agent → intent_agent → [CODE_WRITER_AGENT] → executor_agent → ...

WHAT THIS AGENT DOES:
--------------------
1. Takes the parsed intent (precise analysis instruction)
2. Takes the dataset structure (column names, types, etc.)
3. Uses an LLM to generate Python code that performs the analysis
4. Extracts just the code (strips markdown formatting if present)

WHY CODE GENERATION INSTEAD OF DIRECT ANALYSIS?
----------------------------------------------
We could have an LLM analyze data directly, but generating code has advantages:
1. TRANSPARENCY: Users can see exactly what analysis was performed
2. REPRODUCIBILITY: The same code can be run again for verification
3. FLEXIBILITY: Generated code can handle any pandas operation
4. DEBUGGING: When results are wrong, we can inspect the code

PROMPT ENGINEERING FOR CODE GENERATION:
--------------------------------------
Getting LLMs to write clean, runnable code requires careful prompting:
1. Specify exactly what variable names to use (fig, result_text)
2. Tell it what NOT to do (no fig.show(), no markdown)
3. Provide the schema so it uses correct column names
4. Request inline comments for readability
5. Ask for error handling where appropriate

=============================================================================
"""

import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from state import AnalystState
from prompts.prompts import CODE_WRITER_PROMPT

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
# BASE_URL allows using alternative OpenAI-compatible APIs (e.g., Azure, local LLMs)
# OPENAI_API_KEY is your authentication token for the API
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def code_writer_agent(state: AnalystState) -> dict:
    """
    Generate Python code to perform the requested data analysis.
    
    This agent uses an LLM to write Python code that:
    - Loads the CSV data
    - Performs the analysis specified in parsed_intent
    - Creates Plotly visualizations if appropriate
    - Stores results in standard variable names for extraction
    
    Parameters:
    -----------
    state : AnalystState
        The shared state dictionary. This agent reads from:
        - state["parsed_intent"]: The precise analysis instruction
        - state["dataframe_summary"]: The dataset structure
        
    Returns:
    --------
    dict
        A dictionary with updates to apply to the state:
        - "generated_code": The executable Python code
        
    Code Output Conventions:
    -----------------------
    The generated code must follow these conventions for downstream processing:
    - Variable `fig` contains any Plotly figure
    - Variable `result_text` contains any text/table output
    - The code must NOT call fig.show() (we render in Streamlit, not locally)
    """
    
    # =========================================================================
    # STEP 1: Extract inputs from state
    # =========================================================================
    # We need:
    # - parsed_intent: The precise instruction of what to do
    # - dataframe_summary: Schema info so the LLM knows column names/types
    #
    # Without the schema, the LLM might hallucinate column names or use
    # wrong data types in its code.
    # =========================================================================
    
    parsed_intent = state["parsed_intent"]
    dataframe_summary = state["dataframe_summary"]
    
    # =========================================================================
    # STEP 2: Initialize the LLM client
    # =========================================================================
    # For code generation, we use:
    # - temperature=0 for deterministic, consistent output
    # - gpt-4o or better for higher quality code generation
    #
    # Code generation benefits significantly from more capable models.
    # GPT-3.5 can write basic code but makes more errors with complex logic.
    # =========================================================================
    
    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key=OPENAI_API_KEY,
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0  # Deterministic output - we want reliable code
    )
    
    # =========================================================================
    # STEP 3: Build the prompt
    # =========================================================================
    # The CODE_WRITER_PROMPT template includes:
    # - Clear instructions on how to structure the code
    # - The dataset schema (so the LLM knows what columns exist)
    # - The analysis instruction from intent_agent
    # - Rules about variable naming and what NOT to include
    # - Requirements for comments and code quality
    #
    # PROMPT ENGINEERING CHOICES:
    # 
    # 1. We specify variable names (fig, result_text) explicitly because the
    #    executor needs to extract these specific variables from the namespace.
    #
    # 2. We tell it NOT to call fig.show() because:
    #    - We're not running in an interactive environment
    #    - Streamlit will render the chart separately using st.plotly_chart()
    #    - fig.show() would fail or open an unwanted browser window
    #
    # 3. We ask for NO markdown backticks because:
    #    - The code goes directly to exec(), not a code interpreter
    #    - Backticks would cause a SyntaxError
    #
    # 4. We request inline comments for the user's learning benefit.
    # =========================================================================
    
    prompt = CODE_WRITER_PROMPT.format(
        dataframe_summary=dataframe_summary,
        parsed_intent=parsed_intent
    )
    
    # =========================================================================
    # STEP 4: Call the LLM to generate code
    # =========================================================================
    
    response = llm.invoke([HumanMessage(content=prompt)])
    raw_response = response.content
    
    # =========================================================================
    # STEP 5: Clean the generated code
    # =========================================================================
    # Despite our instructions, LLMs sometimes include markdown formatting.
    # We need to strip this out to get clean, executable code.
    #
    # Common issues we handle:
    # 1. Code wrapped in ```python ... ``` blocks
    # 2. Code wrapped in ``` ... ``` blocks (no language specified)
    # 3. Leading/trailing whitespace
    #
    # The regex pattern below handles both cases robustly.
    # =========================================================================
    
    cleaned_code = strip_markdown_code_blocks(raw_response)
    
    # =========================================================================
    # STEP 6: Return the generated code
    # =========================================================================
    # The cleaned code is now ready to be executed by executor_agent.
    # It should contain:
    # - import statements (pandas, plotly)
    # - data loading (pd.read_csv(csv_path))
    # - data manipulation (filtering, grouping, etc.)
    # - visualization (fig = px.bar(...) or similar)
    # - optional text result (result_text = "...")
    # =========================================================================
    
    return {"generated_code": cleaned_code}


def strip_markdown_code_blocks(text: str) -> str:
    """
    Remove markdown code block formatting from LLM output.
    
    LLMs often wrap code in markdown formatting like:
    ```python
    code here
    ```
    
    This function strips that formatting to get clean, executable code.
    
    Parameters:
    -----------
    text : str
        The raw LLM response that might contain markdown formatting
        
    Returns:
    --------
    str
        Clean code without markdown formatting
        
    Examples:
    ---------
    >>> strip_markdown_code_blocks("```python\nprint('hello')\n```")
    "print('hello')"
    
    >>> strip_markdown_code_blocks("print('hello')")
    "print('hello')"  # Already clean, returned as-is
    """
    
    # =========================================================================
    # PATTERN EXPLANATION:
    # =========================================================================
    # The regex pattern handles multiple markdown code block formats:
    #
    # ```python         <- Opening with language specifier
    # code here
    # ```               <- Closing
    #
    # ```               <- Opening without language specifier
    # code here
    # ```               <- Closing
    #
    # The pattern:
    # - ^```(\w+)?    matches opening ``` optionally followed by language name
    # - \n?           matches optional newline after opening
    # - (.*?)         captures the code (non-greedy)
    # - \n?```$       matches closing with optional newline before
    #
    # re.DOTALL makes . match newlines too
    # =========================================================================
    
    # Pattern to match code blocks with or without language specifier
    pattern = r'^```(?:\w+)?\n?(.*?)\n?```$'
    
    # Try to match and extract the code
    match = re.match(pattern, text.strip(), re.DOTALL)
    
    if match:
        # Found a code block - extract just the code
        return match.group(1).strip()
    else:
        # No code block found - return as-is (already clean)
        return text.strip()


# =============================================================================
# PROMPT ENGINEERING DEEP DIVE
# =============================================================================
#
# WHY WE STRUCTURE THE PROMPT THIS WAY:
#
# 1. EXPLICIT VARIABLE NAMES (fig, result_text):
#    The executor_agent needs to extract results from the code's namespace.
#    By specifying exact variable names in the prompt, we ensure the code
#    creates variables we can find and extract.
#
# 2. NO fig.show() INSTRUCTION:
#    This is critical! In a web application:
#    - fig.show() tries to open a browser, which fails in server context
#    - We render charts differently in Streamlit (st.plotly_chart)
#    - By asking for just the figure object, we can serialize it to JSON
#
# 3. PLOTLY NOT MATPLOTLIB:
#    - Matplotlib uses a global state model that's hard to serialize
#    - Plotly figures are JSON-serializable (fig.to_json())
#    - Plotly renders interactively in Streamlit (zoom, hover, etc.)
#    - Plotly Express has a simpler, more declarative API
#
# 4. DATASET SCHEMA IN PROMPT:
#    Without schema info, the LLM might:
#    - Reference columns that don't exist (hallucination)
#    - Use wrong data types (string ops on numbers)
#    - Miss opportunities (not knowing a date column exists)
#
# 5. CLEAN CODE OUTPUT (NO MARKDOWN):
#    Many code-generating LLMs default to markdown formatting because
#    they're often used in chat contexts where formatting helps readability.
#    But for execution, we need raw code. We ask for this AND clean it
#    as a defensive measure.
#
# =============================================================================
