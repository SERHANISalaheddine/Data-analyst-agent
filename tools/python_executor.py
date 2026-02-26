"""
=============================================================================
PYTHON_EXECUTOR.PY - Safe Code Execution Tool
=============================================================================

This module provides the core functionality for executing dynamically generated
Python code in a controlled environment. It's the "tool" that the executor_agent
uses to run the code written by the code_writer_agent.

WHY USE EXEC() FOR CODE EXECUTION?
----------------------------------
Python's built-in `exec()` function lets us run code stored as a string.
This is powerful but comes with responsibilities:

ADVANTAGES:
- Allows running LLM-generated code dynamically
- We can control what variables are available to the code (namespace)
- We can extract variables from the code after it runs

RISKS & MITIGATIONS:
- SECURITY: exec() can run ANY Python code, including malicious code.
  In production, you'd want sandboxing (Docker, restricted exec, etc.)
  For this learning project, we assume trusted input.
  
- ERROR HANDLING: Generated code might crash. We wrap everything in
  try/except to capture errors gracefully.

THE LOCAL NAMESPACE TRICK:
-------------------------
When you call exec(code, globals, locals), you can provide dictionaries
that act as the global and local namespaces for the executed code.

- We pass csv_path in the namespace so the generated code can use it
- After execution, we can access variables created by the code (like 'fig')
- This isolation prevents the generated code from polluting our main namespace

PLOTLY JSON SERIALIZATION:
-------------------------
Plotly figures are complex Python objects that can't be easily passed between
processes or stored in databases. By converting to JSON:
1. We can store the chart in our state dictionary
2. We can pass it to the Streamlit frontend
3. We can reconstruct it later with plotly.io.from_json()

=============================================================================
"""

import pandas as pd
import plotly.express as px
import plotly.io as pio
from typing import Dict, Any


def execute_code(code: str, csv_path: str) -> Dict[str, Any]:
    """
    Execute generated Python code and extract results.
    
    This function runs code written by the code_writer_agent and extracts
    any Plotly figures and text results that the code produces.
    
    Parameters:
    -----------
    code : str
        The Python code to execute. This code should:
        - Expect a variable `csv_path` to be available
        - Store any Plotly figure in a variable called `fig`
        - Store any text output in a variable called `result_text`
        
    csv_path : str
        The path to the CSV file that the code should analyze.
        This is injected into the execution namespace.
        
    Returns:
    --------
    Dict[str, Any]
        A dictionary containing:
        - "chart_json": The Plotly figure as a JSON string (or None)
        - "execution_result": Text output from the code (or error message)
        - "success": Boolean indicating if execution succeeded
        
    Example:
    --------
    >>> code = '''
    ... import pandas as pd
    ... import plotly.express as px
    ... df = pd.read_csv(csv_path)
    ... fig = px.bar(df, x='category', y='value')
    ... result_text = f"Found {len(df)} rows"
    ... '''
    >>> result = execute_code(code, "data.csv")
    >>> print(result["success"])  # True
    >>> print(result["chart_json"])  # JSON string of the chart
    """
    
    # =========================================================================
    # STEP 1: Set up the execution namespace
    # =========================================================================
    # The namespace is a dictionary that acts as the "environment" where the
    # code will run. We pre-populate it with:
    # - csv_path: So the generated code knows where the data file is
    # - pd: Pandas module, so the code doesn't need to import it
    # - px: Plotly Express module, so the code doesn't need to import it
    #
    # WHY PRE-IMPORT MODULES?
    # Some execution environments restrict imports. By providing these,
    # we ensure the code has what it needs even in restricted settings.
    # =========================================================================
    
    local_namespace = {
        "csv_path": csv_path,  # The path to the CSV file for data loading
        "pd": pd,               # Pandas for data manipulation
        "px": px,               # Plotly Express for charting
    }
    
    # =========================================================================
    # STEP 2: Execute the code in a try/except block
    # =========================================================================
    # We wrap execution in try/except because generated code might:
    # - Have syntax errors
    # - Reference columns that don't exist
    # - Try operations that fail (division by zero, etc.)
    #
    # By catching exceptions, we can return a helpful error message instead
    # of crashing the entire application.
    # =========================================================================
    
    try:
        # exec() runs the code string.
        # - First argument: the code to run
        # - Second argument: global namespace (we use the same as local here)
        # - Third argument: local namespace (where variables are stored)
        #
        # After exec() completes, any variables the code created (like `fig`)
        # will be in local_namespace, and we can access them.
        
        exec(code, local_namespace, local_namespace)
        
        # =====================================================================
        # STEP 3: Extract the Plotly figure (if one was created)
        # =====================================================================
        # The generated code should create a variable called `fig` containing
        # the Plotly figure. We extract it and convert to JSON.
        #
        # WHY CONVERT TO JSON?
        # 1. Plotly figure objects can't be directly serialized by most systems
        # 2. JSON is a universal format that can be stored in databases, caches
        # 3. Streamlit can reconstruct the figure from JSON for display
        # 4. It allows safe state passing in the LangGraph workflow
        # =====================================================================
        
        chart_json = None
        if "fig" in local_namespace:
            fig = local_namespace["fig"]
            # Convert Plotly figure to JSON string
            # This preserves all chart configuration: data, layout, traces, etc.
            chart_json = fig.to_json()
        
        # =====================================================================
        # STEP 4: Extract text results (if any were created)
        # =====================================================================
        # The generated code should create a variable called `result_text`
        # with any text output (tables, summaries, single values, etc.)
        #
        # If no result_text was created, we provide a default message.
        # =====================================================================
        
        result_text = local_namespace.get(
            "result_text", 
            "Code executed successfully. Check the chart for visual results."
        )
        
        # Ensure result_text is a string (in case the code stored something else)
        result_text = str(result_text)
        
        # =====================================================================
        # STEP 5: Return the successful results
        # =====================================================================
        
        return {
            "chart_json": chart_json,        # JSON string of Plotly figure (or None)
            "execution_result": result_text,  # Text output from the code
            "success": True                   # Flag indicating success
        }
        
    except Exception as error:
        # =====================================================================
        # ERROR HANDLING
        # =====================================================================
        # If anything goes wrong during execution, we catch the exception and
        # return a helpful error message. This prevents the entire app from
        # crashing and allows the user to see what went wrong.
        #
        # Common errors include:
        # - SyntaxError: The LLM wrote invalid Python
        # - KeyError: Referenced a column that doesn't exist
        # - TypeError: Wrong data types in operations
        # - ValueError: Invalid values (e.g., empty data for charts)
        # =====================================================================
        
        error_message = f"Error executing code: {type(error).__name__}: {str(error)}"
        
        return {
            "chart_json": None,               # No chart since execution failed
            "execution_result": error_message, # The error message as the result
            "success": False                  # Flag indicating failure
        }


# =============================================================================
# ADDITIONAL NOTES ON SECURITY
# =============================================================================
#
# FOR PRODUCTION USE, CONSIDER:
#
# 1. SANDBOXING: Run code in a Docker container or restricted environment
#    that limits filesystem access, network access, and system calls.
#
# 2. TIMEOUTS: Add a timeout to prevent infinite loops from hanging.
#    You can use the `signal` module or multiprocessing with timeout.
#
# 3. RESOURCE LIMITS: Limit memory and CPU usage to prevent denial-of-service.
#
# 4. CODE VALIDATION: Before execution, scan the code for dangerous patterns
#    like `os.system()`, `subprocess`, `open()` with write mode, etc.
#
# 5. RESTRICTED BUILTINS: You can limit what built-in functions are available
#    by modifying the global namespace passed to exec().
#
# For this learning project, we've kept things simple, but be aware of these
# concerns if you ever deploy something like this in production.
#
# =============================================================================
