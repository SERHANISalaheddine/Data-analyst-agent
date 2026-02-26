"""
=============================================================================
APP.PY - Streamlit Frontend for the Multi-Agent Data Analyst
=============================================================================

This module implements the user interface for the data analyst application
using Streamlit. It handles file uploads, user input, displays results,
and provides a smooth user experience while the multi-agent pipeline runs.

WHAT IS STREAMLIT?
-----------------
Streamlit is a Python framework for building web applications with minimal
frontend code. Key features:

1. PYTHON-NATIVE: Write Python, get a web app. No HTML/CSS/JS required.
2. REACTIVE: When inputs change, the app re-runs automatically.
3. DATA-FRIENDLY: Built-in support for charts, tables, and data viz.
4. RAPID DEVELOPMENT: Changes show instantly with hot reload.

UI STRUCTURE:
-------------
The app has two main areas:

SIDEBAR (left panel):
- File uploader for CSV files
- Preview of the first 5 rows
- Gives users confidence their file was loaded correctly

MAIN AREA (center):
- App title and description
- Text input for questions
- Analyze button
- Results display (chart, narrative, evaluation)

WHY THIS STRUCTURE?
- Sidebar keeps data context visible while working
- Main area focuses on the current task (asking questions)
- Results appear below the input for a natural top-to-bottom flow

EXECUTION FLOW:
--------------
1. User uploads CSV → Saved to temp location, previewed in sidebar
2. User types question → Stored in session state
3. User clicks "Analyze" → Multi-agent pipeline runs with progress updates
4. Results displayed → Chart, narrative, and evaluation

=============================================================================
"""

import streamlit as st
import pandas as pd
import plotly.io as pio
import tempfile
import os
from graph import build_graph

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
# st.set_page_config() MUST be the first Streamlit command.
# It sets global page settings like title, layout, and icon.
#
# - page_title: Appears in the browser tab
# - page_icon: Emoji or image shown in the browser tab
# - layout: "wide" uses the full browser width (better for charts)
# =============================================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# CUSTOM STYLING (Optional)
# =============================================================================
# We inject a small amount of CSS to make the app look cleaner.
# This is optional but improves the user experience.
# =============================================================================

st.markdown("""
    <style>
    /* Make the main content area a bit more spacious */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Style the file uploader */
    .uploadedFile {
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR: FILE UPLOAD AND DATA PREVIEW
# =============================================================================
# The sidebar is perfect for context that should always be visible.
# Users can reference their data while asking questions.
#
# STREAMLIT COMPONENTS USED:
# - st.sidebar: Context manager for sidebar content
# - st.file_uploader: File upload widget
# - st.dataframe: Interactive data table
# - st.info/warning: Styled message boxes
# =============================================================================

with st.sidebar:
    # Sidebar title
    st.title("📁 Data Upload")
    
    # File uploader widget
    # - type=["csv"] restricts to CSV files only
    # - help= provides a tooltip on hover
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="Upload a CSV file to analyze. The file will be used for all questions."
    )
    
    # Show data preview if a file is uploaded
    # WHY PREVIEW HERE?
    # Users need to see their data to ask good questions.
    # Knowing column names helps them reference the right fields.
    
    if uploaded_file is not None:
        # Visual separator
        st.divider()
        
        # Success message confirming upload
        st.success(f"✓ Loaded: {uploaded_file.name}")
        
        # Load and preview the data
        # We read the CSV twice: once here for preview, once later for analysis.
        # For large files, you might want to cache this.
        try:
            # Create a DataFrame for preview
            preview_df = pd.read_csv(uploaded_file)
            
            # Show dataset dimensions
            st.caption(f"📐 {len(preview_df):,} rows × {len(preview_df.columns)} columns")
            
            # Preview header
            st.subheader("Data Preview")
            
            # Show first 5 rows
            # use_container_width=True makes the table fill the sidebar width
            st.dataframe(
                preview_df.head(5),
                use_container_width=True,
                height=200  # Fixed height so it doesn't take too much space
            )
            
            # Show column names for reference
            # This helps users know what fields they can ask about
            with st.expander("📋 Column Names"):
                for column in preview_df.columns:
                    st.code(column, language=None)
            
            # Reset file pointer so it can be read again later
            # IMPORTANT: After reading a file, the pointer is at the end.
            # We must seek(0) to reset it for future reads.
            uploaded_file.seek(0)
            
        except Exception as error:
            st.error(f"Error reading file: {str(error)}")
    else:
        # No file uploaded yet - show instructions
        st.info("👆 Upload a CSV file to get started")

# =============================================================================
# MAIN AREA: TITLE AND DESCRIPTION
# =============================================================================
# The main area starts with a clear title and brief description.
# This sets expectations for what the app does.
# =============================================================================

st.title("📊 AI Data Analyst")

st.markdown("""
Welcome to your AI-powered data analyst! This application uses a team of 
specialized AI agents to analyze your data and answer questions in plain English.

**How it works:**
1. Upload a CSV file in the sidebar
2. Ask a question about your data below
3. Get an interactive chart and explanation

*Powered by LangGraph, OpenAI, and Plotly*
""")

# Visual separator between description and input
st.divider()

# =============================================================================
# MAIN AREA: QUESTION INPUT AND ANALYSIS BUTTON
# =============================================================================
# We use a form to group the input and button together.
# This prevents the app from re-running on every keystroke.
#
# WHY USE A FORM?
# Streamlit re-runs the entire script when any input changes.
# Without a form, the app would try to run analysis on every keystroke.
# Forms batch inputs together and only submit when the button is clicked.
# =============================================================================

# Create a form for the question input
with st.form(key="question_form"):
    # Text input for the user's question
    user_question = st.text_input(
        "What would you like to know about your data?",
        placeholder="e.g., What are the top 5 products by sales? Show me monthly trends.",
        help="Ask any question about your data. The AI will analyze it and create visualizations."
    )
    
    # Submit button
    # type="primary" makes it visually prominent with the theme's primary color
    submit_button = st.form_submit_button(
        "🔍 Analyze",
        type="primary",
        use_container_width=True
    )

# =============================================================================
# ANALYSIS EXECUTION
# =============================================================================
# This section runs when the user clicks "Analyze".
# We validate inputs, run the multi-agent pipeline, and display results.
#
# ERROR HANDLING:
# We check for required inputs before proceeding:
# 1. Is a file uploaded?
# 2. Is a question entered?
#
# If either is missing, we show a friendly warning instead of crashing.
# =============================================================================

if submit_button:
    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================
    # Check that we have everything we need before running the pipeline.
    # =========================================================================
    
    if uploaded_file is None:
        st.warning("⚠️ Please upload a CSV file first.")
        st.stop()  # Stop execution here
    
    if not user_question.strip():
        st.warning("⚠️ Please enter a question about your data.")
        st.stop()  # Stop execution here
    
    # =========================================================================
    # SAVE UPLOADED FILE TO TEMP LOCATION
    # =========================================================================
    # Streamlit's uploaded file is in memory. We need to save it to disk
    # so our agents can access it by path.
    #
    # WHY TEMP FILE?
    # - We don't want to clutter the project directory with uploads
    # - Temp files are automatically cleaned up by the OS
    # - We get a consistent file path to pass to agents
    #
    # IMPORTANT: delete=False keeps the file after closing the handle
    # We'll clean it up after the pipeline completes
    # =========================================================================
    
    # Create a temporary file with .csv extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        # Write the uploaded content to the temp file
        temp_file.write(uploaded_file.getvalue())
        csv_path = temp_file.name  # Get the path for agents to use
    
    try:
        # =====================================================================
        # RUN THE MULTI-AGENT PIPELINE WITH PROGRESS DISPLAY
        # =====================================================================
        # We use st.status() to show users which agent is currently running.
        # This provides transparency into the "thinking" process.
        #
        # st.status() creates an expandable container that shows:
        # - Current status (running, complete, error)
        # - Detailed steps inside (we add these as agents run)
        # =====================================================================
        
        with st.status("🤖 Analyzing your data...", expanded=True) as status:
            # Step 1: Build the graph
            st.write("📋 Initializing analysis pipeline...")
            graph = build_graph()
            
            # Step 2: Prepare initial state
            initial_state = {
                "csv_path": csv_path,
                "user_question": user_question
            }
            
            # Step 3: Run the full pipeline
            # We show which agents are running for transparency
            st.write("🔍 **Schema Agent:** Reading dataset structure...")
            st.write("🎯 **Intent Agent:** Understanding your question...")
            st.write("💻 **Code Writer Agent:** Generating analysis code...")
            st.write("⚙️ **Executor Agent:** Running the analysis...")
            st.write("📝 **Narrative Agent:** Creating explanation...")
            st.write("✅ **Critic Agent:** Evaluating results...")
            
            # Execute the graph
            # graph.invoke() runs all agents in sequence and returns final state
            final_state = graph.invoke(initial_state)
            
            # Update status to complete
            status.update(label="✅ Analysis complete!", state="complete", expanded=False)
        
        # =====================================================================
        # DISPLAY RESULTS
        # =====================================================================
        # After the pipeline completes, we display:
        # 1. The Plotly chart (if one was created)
        # 2. The narrative explanation
        # 3. The critic evaluation (PASS/FAIL)
        # =====================================================================
        
        st.divider()
        st.header("📈 Results")
        
        # -----------------------------------------------------------------
        # DISPLAY CHART (if available)
        # -----------------------------------------------------------------
        # The chart is stored as JSON in state["chart_json"].
        # We convert it back to a Plotly figure using pio.from_json().
        #
        # st.plotly_chart() renders interactive Plotly charts in Streamlit.
        # - use_container_width=True makes the chart fill the available width
        # -----------------------------------------------------------------
        
        chart_json = final_state.get("chart_json")
        
        if chart_json:
            try:
                # Convert JSON back to Plotly figure
                fig = pio.from_json(chart_json)
                
                # Display the interactive chart
                # Users can hover, zoom, pan, and download
                st.plotly_chart(fig, use_container_width=True)
            except Exception as chart_error:
                st.warning(f"Could not display chart: {str(chart_error)}")
        else:
            # No chart was generated - this might be a text-only result
            st.info("No chart was generated for this analysis.")
        
        # -----------------------------------------------------------------
        # DISPLAY NARRATIVE EXPLANATION
        # -----------------------------------------------------------------
        # The narrative is the plain English explanation of results.
        # We display it in an st.info() box for visual distinction.
        #
        # st.info() creates a blue-tinted box that draws attention.
        # -----------------------------------------------------------------
        
        narrative = final_state.get("narrative", "No explanation available.")
        
        st.subheader("💬 Explanation")
        st.info(narrative)
        
        # -----------------------------------------------------------------
        # DISPLAY CRITIC EVALUATION
        # -----------------------------------------------------------------
        # The critic provides a PASS/FAIL score and a reason.
        # We use color-coded boxes to make the result immediately clear:
        # - st.success() = green (for PASS)
        # - st.error() = red (for FAIL)
        # -----------------------------------------------------------------
        
        critic_score = final_state.get("critic_score", "UNKNOWN")
        critique = final_state.get("critique", "No evaluation available.")
        
        st.subheader("🎯 Evaluation")
        
        if critic_score == "PASS":
            # Green success box for passing evaluation
            st.success(f"**PASS** ✅ {critique}")
        elif critic_score == "FAIL":
            # Red error box for failing evaluation
            st.error(f"**FAIL** ❌ {critique}")
        else:
            # Yellow warning for unknown/error state
            st.warning(f"**{critic_score}** - {critique}")
        
        # -----------------------------------------------------------------
        # OPTIONAL: SHOW TECHNICAL DETAILS (EXPANDABLE)
        # -----------------------------------------------------------------
        # For users who want to see what happened "under the hood",
        # we provide an expandable section with technical details.
        # This is great for debugging and learning.
        # -----------------------------------------------------------------
        
        with st.expander("🔧 Technical Details"):
            # Tab layout for organizing details
            tab1, tab2, tab3 = st.tabs(["Parsed Intent", "Generated Code", "Raw Results"])
            
            with tab1:
                st.markdown("**How the AI interpreted your question:**")
                st.code(final_state.get("parsed_intent", "N/A"), language=None)
            
            with tab2:
                st.markdown("**Python code that was generated:**")
                st.code(final_state.get("generated_code", "N/A"), language="python")
            
            with tab3:
                st.markdown("**Raw execution output:**")
                st.code(final_state.get("execution_result", "N/A"), language=None)
    
    except Exception as error:
        # =====================================================================
        # ERROR HANDLING
        # =====================================================================
        # If anything goes wrong during analysis, we display a friendly error.
        # We also show the technical error for debugging.
        # =====================================================================
        
        st.error("❌ An error occurred during analysis")
        st.exception(error)  # Shows the full stack trace
    
    finally:
        # =====================================================================
        # CLEANUP: Delete the temporary file
        # =====================================================================
        # We always clean up the temp file, even if an error occurred.
        # This prevents accumulation of temp files on the server.
        # =====================================================================
        
        if os.path.exists(csv_path):
            os.remove(csv_path)

# =============================================================================
# FOOTER
# =============================================================================
# A simple footer with attribution and helpful links.
# =============================================================================

st.divider()
st.caption("Built with ❤️ using LangGraph, OpenAI, Streamlit, and Plotly")


# =============================================================================
# STREAMLIT EXECUTION MODEL - IMPORTANT CONCEPTS
# =============================================================================
#
# HOW STREAMLIT WORKS (THE "EXECUTION MODEL"):
#
# Unlike traditional web apps (client sends request, server responds once),
# Streamlit re-runs the ENTIRE script from top to bottom whenever:
# - The user interacts with a widget
# - The page is refreshed
# - The code changes (during development)
#
# This might seem inefficient, but it's actually elegant:
# - No complex state management code needed
# - The script IS the application state
# - Easy to reason about: "what runs when"
#
# IMPORTANT IMPLICATIONS:
#
# 1. VARIABLES RESET: Variables defined in the script reset on each run.
#    To persist data across runs, use st.session_state.
#
# 2. WIDGETS HAVE IDENTITY: Streamlit tracks widgets by their key/position.
#    This is why widget states persist (like input text).
#
# 3. FORMS BATCH UPDATES: st.form() prevents re-runs until submitted.
#    Essential for multi-input scenarios like ours.
#
# 4. CACHING SAVES TIME: Use @st.cache_data and @st.cache_resource
#    to avoid re-computing expensive operations.
#
# In this app:
# - The form prevents analysis from running on every keystroke
# - File handling uses a fresh temp file each time (stateless)
# - We don't use session state because each analysis is independent
#
# =============================================================================
