"""
=============================================================================
SCHEMA_AGENT.PY - Dataset Structure Analysis Agent
=============================================================================

This module contains the schema_agent, which is the FIRST agent in the
multi-agent pipeline. Its job is to analyze the uploaded CSV file and create
a human-readable summary of its structure.

ROLE IN THE PIPELINE:
--------------------
User uploads CSV → [SCHEMA_AGENT] → intent_agent → code_writer_agent → ...

WHY THIS AGENT EXISTS:
---------------------
LLMs need context about the data to write correct code. If we just passed
the user's question to a code-writing LLM, it would have to guess column
names, data types, and what the data looks like. This leads to errors like:
- Using column names that don't exist
- Treating text columns as numbers
- Missing the structure of the data entirely

By analyzing the CSV first, we give downstream agents (intent_agent and
code_writer_agent) the context they need to do their jobs accurately.

WHY NO LLM HERE:
---------------
This agent uses pure pandas logic, NOT an LLM. Why?
1. DETERMINISM: Pandas always gives the same result for the same data
2. SPEED: No network call to an LLM provider
3. ACCURACY: Pandas knows exactly what columns exist; LLMs might misread
4. COST: No API token usage for a task that doesn't need intelligence

WHAT THE SUMMARY INCLUDES:
-------------------------
- Dataset dimensions (rows × columns)
- Column names and their data types
- Null value counts per column
- Sample rows to show what the data looks like

This gives downstream agents everything they need to write correct code.

=============================================================================
"""

import pandas as pd
from state import AnalystState


def schema_agent(state: AnalystState) -> dict:
    """
    Analyze the CSV file and create a summary of its structure.
    
    This function reads the CSV file from the path stored in state,
    analyzes its structure using pandas, and creates a human-readable
    summary that will be used by downstream agents.
    
    Parameters:
    -----------
    state : AnalystState
        The shared state dictionary. This agent reads from:
        - state["csv_path"]: Path to the CSV file to analyze
        
    Returns:
    --------
    dict
        A dictionary with updates to apply to the state:
        - "dataframe_summary": A text summary of the dataset structure
        
    How LangGraph Uses This Return Value:
    -------------------------------------
    LangGraph automatically merges the returned dict into the state.
    So returning {"dataframe_summary": "..."} updates state["dataframe_summary"].
    """
    
    # =========================================================================
    # STEP 1: Get the CSV path from state
    # =========================================================================
    # The Streamlit frontend stored the path when the user uploaded a file.
    # We retrieve it here to load and analyze the data.
    # =========================================================================
    
    csv_path = state["csv_path"]
    
    # =========================================================================
    # STEP 2: Load the CSV file into a pandas DataFrame
    # =========================================================================
    # pd.read_csv() is the standard way to load CSV files in Python.
    # It automatically infers column types (numbers, strings, dates, etc.)
    # which we'll report in our summary.
    # =========================================================================
    
    dataframe = pd.read_csv(csv_path)
    
    # =========================================================================
    # STEP 3: Extract basic dataset information
    # =========================================================================
    # We gather several pieces of information that help understand the data:
    #
    # - Shape: How many rows and columns? This tells us the data size.
    # - Column names: What variables are in the dataset?
    # - Data types: Are columns numeric, text, dates? Critical for code writing.
    # - Null counts: Any missing data? Code might need to handle this.
    # =========================================================================
    
    # Get dimensions of the dataset
    num_rows, num_columns = dataframe.shape
    
    # Get column names as a list
    column_names = dataframe.columns.tolist()
    
    # Get data types for each column
    # This returns a Series with column names as index and dtypes as values
    data_types = dataframe.dtypes
    
    # Count null values in each column
    # This returns a Series with column names as index and null counts as values
    null_counts = dataframe.isnull().sum()
    
    # =========================================================================
    # STEP 4: Get sample rows from the dataset
    # =========================================================================
    # Showing a few sample rows helps LLMs understand what values look like.
    # For example, they can see:
    # - Date formats (2024-01-15 vs 01/15/2024)
    # - Text patterns (product codes, categories)
    # - Numeric ranges (millions vs decimals)
    #
    # We limit to 3 rows to keep the summary concise while still informative.
    # =========================================================================
    
    sample_rows = dataframe.head(3).to_string()
    
    # =========================================================================
    # STEP 5: Build the column information section
    # =========================================================================
    # For each column, we show:
    # - Column name
    # - Data type (int64, float64, object, datetime64, etc.)
    # - Number of null/missing values
    #
    # This helps the code_writer_agent know what it's working with.
    # For example, if a column is "object" type (text), you can't calculate
    # its mean without converting it first.
    # =========================================================================
    
    column_info_lines = []
    for column_name in column_names:
        dtype = data_types[column_name]
        nulls = null_counts[column_name]
        column_info_lines.append(f"  - {column_name}: {dtype} ({nulls} nulls)")
    
    column_info_text = "\n".join(column_info_lines)
    
    # =========================================================================
    # STEP 6: Compose the final summary
    # =========================================================================
    # We format all the information into a clear, readable text block.
    # This text will be inserted into prompts for the intent_agent and
    # code_writer_agent, so we make it easy to read and understand.
    #
    # FORMAT CHOICES:
    # - Clear section headers (DATASET OVERVIEW, COLUMNS, SAMPLE DATA)
    # - Bullet points for column details
    # - Actual data samples in a table format
    # =========================================================================
    
    summary = f"""DATASET OVERVIEW:
==================
Total Rows: {num_rows}
Total Columns: {num_columns}

COLUMNS AND DATA TYPES:
=======================
{column_info_text}

SAMPLE DATA (First 3 Rows):
===========================
{sample_rows}
"""
    
    # =========================================================================
    # STEP 7: Return the update to state
    # =========================================================================
    # LangGraph uses a "reducer" pattern where agent return values are merged
    # into the state. By returning {"dataframe_summary": summary}, we're telling
    # LangGraph to update state["dataframe_summary"] with our summary.
    #
    # This summary will be available to all subsequent agents in the pipeline.
    # =========================================================================
    
    return {"dataframe_summary": summary}


# =============================================================================
# DESIGN NOTES
# =============================================================================
#
# WHY RETURN A DICT INSTEAD OF MODIFYING STATE DIRECTLY?
#
# LangGraph is designed around immutable state updates. Instead of modifying
# the state dict in place, we return the changes we want to make. LangGraph
# then handles merging those changes into the state. This pattern:
#
# 1. Makes debugging easier (you can see exactly what each agent changed)
# 2. Enables features like state checkpointing and time-travel debugging
# 3. Follows functional programming principles (no side effects)
#
# WHAT IF THE CSV IS VERY LARGE?
#
# This implementation loads the entire CSV into memory. For very large files
# (millions of rows), you might want to:
# 1. Only load the first N rows for the summary
# 2. Use chunked reading with pd.read_csv(chunksize=...)
# 3. Sample the data randomly instead of taking head()
#
# For this learning project, we assume reasonable file sizes.
#
# =============================================================================
