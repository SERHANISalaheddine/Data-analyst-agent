"""
=============================================================================
GRAPH.PY - LangGraph Workflow Definition
=============================================================================

This module defines the LangGraph workflow that orchestrates all agents in
the multi-agent data analyst system. It's the "brain" that connects everything.

WHAT IS LANGGRAPH?
------------------
LangGraph is a library for building stateful, graph-based LLM applications.
It's part of the LangChain ecosystem but focuses specifically on orchestration.

KEY CONCEPTS:

1. GRAPH: A directed graph where nodes are processing steps and edges define
   the flow between them. Unlike simple chains, graphs can have branches,
   loops, and conditional routing.

2. STATE: A shared data structure that flows through the graph. Each node
   can read from and write to the state. This is how nodes communicate.

3. NODES: Individual processing steps (usually functions or agents). Each
   node takes the current state, does some work, and returns state updates.

4. EDGES: Connections between nodes that define the execution order.
   Edges can be unconditional (always follow) or conditional (based on state).

WHAT IS A STATEGRAPH?
--------------------
StateGraph is LangGraph's primary graph type. It:
- Uses TypedDict for the state schema (our AnalystState)
- Automatically merges node outputs into the state
- Supports checkpointing and persistence
- Handles errors and retries

WHY USE A GRAPH INSTEAD OF SEQUENTIAL FUNCTION CALLS?
----------------------------------------------------
You might think: "Why not just call agents in a loop?"

agent_functions = [schema_agent, intent_agent, code_writer_agent, ...]
for agent in agent_functions:
    state = agent(state)

The graph approach provides:

1. VISIBILITY: The graph structure is explicit and inspectable. You can
   visualize the flow, understand dependencies, and debug bottlenecks.

2. FLEXIBILITY: Easy to add conditional branches, parallel execution,
   loops, and error handling paths without spaghetti code.

3. PERSISTENCE: LangGraph can checkpoint state after each node, enabling
   resume-after-failure and human-in-the-loop patterns.

4. OBSERVABILITY: Built-in tracing and monitoring integrations.

5. REUSABILITY: The same graph definition can be used with different
   state implementations or agent configurations.

For this simple linear flow, a graph might seem like overkill, but it sets
the foundation for more complex workflows (like retry loops, user approval
steps, or branching based on question type).

=============================================================================
"""

from langgraph.graph import StateGraph, START, END
from state import AnalystState

# Import all agent functions from their respective modules
from agents.schema_agent import schema_agent
from agents.intent_agent import intent_agent
from agents.code_writer_agent import code_writer_agent
from agents.executor_agent import executor_agent
from agents.narrative_agent import narrative_agent
from agents.critic_agent import critic_agent


def build_graph():
    """
    Build and compile the LangGraph workflow for the data analyst system.
    
    This function creates a StateGraph, adds all agents as nodes, connects
    them with edges, and compiles the graph into an executable workflow.
    
    Returns:
    --------
    CompiledGraph
        A compiled LangGraph that can be invoked with an initial state.
        
    How to Use:
    ----------
    graph = build_graph()
    initial_state = {
        "csv_path": "/path/to/data.csv",
        "user_question": "What are the top products?"
    }
    final_state = graph.invoke(initial_state)
    print(final_state["narrative"])
    
    Graph Structure:
    ---------------
    START → schema_agent → intent_agent → code_writer_agent → 
            executor_agent → narrative_agent → critic_agent → END
    """
    
    # =========================================================================
    # STEP 1: Create the StateGraph with our state schema
    # =========================================================================
    # StateGraph takes a type as its argument - this is the schema for the
    # state that will flow through the graph. We use our AnalystState TypedDict.
    #
    # WHY PASS A TYPE?
    # LangGraph uses the type for:
    # - Validation: ensuring nodes return compatible state updates
    # - Tooling: IDE autocomplete and type checking
    # - Documentation: clear contract for what data flows through
    # =========================================================================
    
    workflow = StateGraph(AnalystState)
    
    # =========================================================================
    # STEP 2: Add nodes (agents) to the graph
    # =========================================================================
    # Each node has a name (string) and a function to execute.
    # The function must:
    # - Take the current state as input (dict-like)
    # - Return a dict of state updates (not the full state)
    #
    # NODE NAMING CONVENTIONS:
    # - Use descriptive names that reflect the agent's purpose
    # - These names will appear in logs, traces, and visualizations
    # - We use snake_case to match Python naming conventions
    #
    # ORDER OF add_node CALLS:
    # The order doesn't affect execution - that's determined by edges.
    # We add them in logical order for readability.
    # =========================================================================
    
    # Node 1: Schema Agent
    # Analyzes the CSV structure and creates a summary
    workflow.add_node("schema_agent", schema_agent)
    
    # Node 2: Intent Agent
    # Parses the user's question into a precise instruction
    workflow.add_node("intent_agent", intent_agent)
    
    # Node 3: Code Writer Agent
    # Generates Python/Plotly code to answer the question
    workflow.add_node("code_writer_agent", code_writer_agent)
    
    # Node 4: Executor Agent
    # Runs the generated code and captures results
    workflow.add_node("executor_agent", executor_agent)
    
    # Node 5: Narrative Agent
    # Creates a plain English explanation of the results
    workflow.add_node("narrative_agent", narrative_agent)
    
    # Node 6: Critic Agent
    # Evaluates whether the answer addressed the question
    workflow.add_node("critic_agent", critic_agent)
    
    # =========================================================================
    # STEP 3: Define edges (connections between nodes)
    # =========================================================================
    # Edges define the execution flow. For this simple pipeline, we use a
    # linear sequence: each agent passes to the next one.
    #
    # add_edge(from_node, to_node) creates a directed connection.
    #
    # SPECIAL NODES:
    # - START: The entry point of the graph (where execution begins)
    # - END: The exit point of the graph (where execution completes)
    #
    # These are imported from langgraph.graph and represent the graph's
    # boundaries.
    #
    # EDGE TYPES (not all used here, but good to know):
    # - Unconditional: Always follows this edge (what we use)
    # - Conditional: Uses a function to decide which edge to follow
    # =========================================================================
    
    # Connect START to the first agent
    # This defines where execution begins when invoke() is called
    workflow.add_edge(START, "schema_agent")
    
    # Connect agents in sequence
    # schema_agent → intent_agent: schema must be extracted before parsing intent
    workflow.add_edge("schema_agent", "intent_agent")
    
    # intent_agent → code_writer_agent: intent must be parsed before writing code
    workflow.add_edge("intent_agent", "code_writer_agent")
    
    # code_writer_agent → executor_agent: code must be written before execution
    workflow.add_edge("code_writer_agent", "executor_agent")
    
    # executor_agent → narrative_agent: results must exist before explaining
    workflow.add_edge("executor_agent", "narrative_agent")
    
    # narrative_agent → critic_agent: narrative must exist before evaluation
    workflow.add_edge("narrative_agent", "critic_agent")
    
    # Connect the last agent to END
    # This defines where execution completes
    workflow.add_edge("critic_agent", END)
    
    # =========================================================================
    # STEP 4: Compile the graph
    # =========================================================================
    # compile() transforms the graph definition into an executable workflow.
    #
    # WHAT COMPILE DOES:
    # 1. Validates the graph structure (no orphan nodes, valid edges, etc.)
    # 2. Optimizes the execution plan
    # 3. Sets up state management
    # 4. Returns a CompiledGraph object with invoke() and stream() methods
    #
    # COMMON COMPILE OPTIONS (not used here, but available):
    # - checkpointer: Add persistence for resumable workflows
    # - interrupt_before/after: Pause for human approval
    # - debug: Enable verbose logging
    # =========================================================================
    
    compiled_graph = workflow.compile()
    
    return compiled_graph


# =============================================================================
# UNDERSTANDING GRAPH EXECUTION
# =============================================================================
#
# WHAT HAPPENS WHEN YOU CALL graph.invoke(state):
#
# 1. The graph starts at START and follows the first edge to schema_agent
# 2. schema_agent runs with the initial state, returns {"dataframe_summary": ...}
# 3. LangGraph merges this into the state: state.update({"dataframe_summary": ...})
# 4. Follows edge to intent_agent
# 5. intent_agent runs with the updated state, returns {"parsed_intent": ...}
# 6. LangGraph merges this into the state
# 7. ...continues through all nodes...
# 8. When critic_agent completes and follows edge to END, invoke() returns
# 9. The return value is the final state with all agent outputs merged in
#
# STATE MERGING:
# By default, LangGraph uses a simple merge strategy: it updates the state
# dict with each node's return value. For more complex cases, you can define
# custom reducers.
#
# ERROR HANDLING:
# If any agent raises an exception:
# - The graph stops execution
# - The error propagates to the caller
# - You can add error handling with try/except in agents
# - Or use LangGraph's built-in retry/fallback mechanisms
#
# =============================================================================

# =============================================================================
# FUTURE ENHANCEMENTS
# =============================================================================
#
# This graph is simple (linear flow), but LangGraph supports much more:
#
# 1. CONDITIONAL ROUTING:
#    def route_by_question_type(state):
#        if "chart" in state["user_question"].lower():
#            return "visualization_agent"
#        else:
#            return "table_agent"
#    
#    workflow.add_conditional_edges("intent_agent", route_by_question_type)
#
# 2. RETRY LOOPS:
#    If critic_agent returns FAIL, loop back to code_writer_agent:
#    
#    def should_retry(state):
#        if state["critic_score"] == "FAIL" and state.get("retry_count", 0) < 3:
#            return "code_writer_agent"  # Try again
#        else:
#            return END
#    
#    workflow.add_conditional_edges("critic_agent", should_retry)
#
# 3. PARALLEL EXECUTION:
#    Run multiple agents simultaneously and merge their outputs:
#    
#    workflow.add_node("sentiment_agent", sentiment_agent)
#    workflow.add_node("key_metrics_agent", key_metrics_agent)
#    # Both run in parallel after executor_agent
#
# 4. HUMAN-IN-THE-LOOP:
#    Pause execution for human approval before certain steps:
#    
#    workflow.compile(interrupt_before=["executor_agent"])
#    # Graph pauses before running code, allowing human review
#
# For this learning project, we keep it simple, but these patterns are
# available when you need them.
#
# =============================================================================
