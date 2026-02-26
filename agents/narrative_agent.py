"""
=============================================================================
NARRATIVE_AGENT.PY - Results Explanation Agent
=============================================================================

This module contains the narrative_agent, which is the FIFTH agent in the
multi-agent pipeline. Its job is to transform technical analysis results
into a clear, plain English explanation that non-technical users can understand.

ROLE IN THE PIPELINE:
--------------------
schema_agent → intent_agent → code_writer_agent → executor_agent → [NARRATIVE_AGENT] → critic_agent

WHAT THIS AGENT DOES:
--------------------
1. Takes the original user question (to reference what was asked)
2. Takes the parsed intent (to know what analysis was performed)
3. Takes the execution result (text output from the code)
4. Uses an LLM to write a human-friendly explanation

THE CONCEPT OF "LAST MILE" COMMUNICATION:
-----------------------------------------
In data products, "last mile" refers to the final step of delivering insights
to end users in a way they can understand and act upon. This concept comes
from logistics (delivering packages to homes) and telecommunications 
(connecting networks to homes).

In data analytics, the "last mile" problem is significant:
- Data scientists can build amazing analyses
- But if business users can't understand the insights...
- ...the analysis provides no value

Many analytics projects fail not because the analysis is wrong, but because
the results weren't communicated effectively. This agent solves that problem
by automatically generating clear, jargon-free explanations.

EXAMPLE TRANSFORMATION:
----------------------
Technical result:
  "groupby: region, agg: sum(sales)
   North: 1234567.89
   South: 987654.32
   East: 876543.21"

Narrative explanation:
  "Looking at your sales data by region, the North region leads with 
   approximately $1.23 million in total sales. The South region follows
   at about $988K, while the East region generated around $877K. The
   North region is outperforming other regions by a significant margin."

The narrative is actionable and understandable without technical background.

=============================================================================
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from state import AnalystState
from prompts.prompts import NARRATIVE_PROMPT

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
# BASE_URL allows using alternative OpenAI-compatible APIs (e.g., Azure, local LLMs)
# OPENAI_API_KEY is your authentication token for the API
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def narrative_agent(state: AnalystState) -> dict:
    """
    Generate a plain English explanation of the analysis results.
    
    This agent takes technical execution results and transforms them
    into a clear, accessible narrative for non-technical stakeholders.
    
    Parameters:
    -----------
    state : AnalystState
        The shared state dictionary. This agent reads from:
        - state["user_question"]: The original question (for context)
        - state["parsed_intent"]: What analysis was performed
        - state["execution_result"]: The technical output to explain
        
    Returns:
    --------
    dict
        A dictionary with updates to apply to the state:
        - "narrative": The plain English explanation
        
    Why Include Original Question?
    -----------------------------
    Including the original question helps the LLM write a narrative that
    directly addresses what the user asked. The narrative should start
    with something like "Based on your question about X..." or "Looking
    at X as you requested..." This makes users feel heard.
    """
    
    # =========================================================================
    # STEP 1: Extract inputs from state
    # =========================================================================
    # We gather three pieces of information:
    # - user_question: So we can reference what was originally asked
    # - parsed_intent: So we know what analysis approach was taken
    # - execution_result: The actual findings to explain
    #
    # All three contribute to a well-rounded narrative.
    # =========================================================================
    
    user_question = state["user_question"]
    parsed_intent = state["parsed_intent"]
    execution_result = state["execution_result"]
    
    # =========================================================================
    # STEP 2: Initialize the LLM client
    # =========================================================================
    # For narrative generation, we use slightly higher temperature (0.3)
    # compared to code generation (0.0). Why?
    #
    # - Narratives benefit from some variation in phrasing
    # - Multiple valid ways to explain the same result exist
    # - A little creativity makes the text feel less robotic
    #
    # Still, we keep it low enough to avoid hallucinating facts.
    # =========================================================================
    
    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key=OPENAI_API_KEY,
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0.3  # Slightly creative for natural-sounding text
    )
    
    # =========================================================================
    # STEP 3: Build the prompt
    # =========================================================================
    # The NARRATIVE_PROMPT template instructs the LLM to:
    # - Write in plain English (no jargon)
    # - Keep it to 3-5 sentences (concise but complete)
    # - Include specific numbers from the results
    # - Avoid bullet points (conversational flow)
    # - Focus on insights, not just numbers
    #
    # We inject all three context pieces into the template.
    # =========================================================================
    
    prompt = NARRATIVE_PROMPT.format(
        user_question=user_question,
        parsed_intent=parsed_intent,
        execution_result=execution_result
    )
    
    # =========================================================================
    # STEP 4: Call the LLM to generate narrative
    # =========================================================================
    
    response = llm.invoke([HumanMessage(content=prompt)])
    narrative = response.content
    
    # =========================================================================
    # STEP 5: Return the narrative
    # =========================================================================
    # The narrative will be:
    # - Displayed to the user in the Streamlit UI
    # - Evaluated by the critic_agent for quality
    #
    # A good narrative should:
    # - Directly address the user's question
    # - Include specific numbers/findings
    # - Be understandable without technical background
    # - Provide actionable insight where possible
    # =========================================================================
    
    return {"narrative": narrative}


# =============================================================================
# WHY NARRATIVE GENERATION MATTERS
# =============================================================================
#
# THE DATA-TO-INSIGHT GAP:
#
# Most data tools stop at charts and numbers. But users often need to:
# 1. Explain results to others (in meetings, reports)
# 2. Understand what numbers MEAN, not just what they ARE
# 3. Make decisions based on analysis
#
# A chart showing "North: 1.2M, South: 900K" is informative.
# A narrative saying "The North region is your top performer, generating
# 33% more revenue than the South" is actionable.
#
# THE LLM ADVANTAGE:
#
# Before LLMs, generating narratives required:
# - Template-based systems (rigid, robotic-sounding)
# - Human writers (doesn't scale)
#
# LLMs can generate natural-sounding narratives that:
# - Adapt to any data pattern
# - Use appropriate language for the audience
# - Highlight what's interesting/unusual
#
# NARRATIVE QUALITY FACTORS:
#
# Good data narratives:
# 1. START with what was asked (shows you understood)
# 2. INCLUDE specific numbers (credibility)
# 3. EXPLAIN what numbers mean (context)
# 4. HIGHLIGHT what's notable (insight)
# 5. USE simple language (accessibility)
#
# Bad data narratives:
# 1. Repeat technical jargon from the analysis
# 2. Just list numbers without context
# 3. Don't connect back to the original question
# 4. Are too long or too short
#
# =============================================================================
