"""
=============================================================================
CRITIC_AGENT.PY - Quality Evaluation Agent
=============================================================================

This module contains the critic_agent, which is the SIXTH and FINAL agent
in the multi-agent pipeline. Its job is to evaluate whether the analysis
actually answered the user's original question.

ROLE IN THE PIPELINE:
--------------------
schema_agent → intent_agent → code_writer_agent → executor_agent → narrative_agent → [CRITIC_AGENT]

WHAT THIS AGENT DOES:
--------------------
1. Takes the original user question
2. Takes the narrative explanation
3. Takes the execution result
4. Uses an LLM to evaluate: "Did we actually answer the question?"
5. Returns a PASS/FAIL score with a reason

THE CONCEPT OF "LLM-AS-JUDGE":
-----------------------------
LLM-as-judge is an emerging pattern in AI evaluation where we use one LLM
to evaluate the output of another LLM (or system). This approach is used
because:

1. SCALABILITY: Human evaluation doesn't scale. An LLM can evaluate
   thousands of outputs per minute.

2. CONSISTENCY: Once tuned, an LLM applies the same criteria every time.
   Human evaluators may have different standards.

3. REASONING: Unlike simple metrics (accuracy, F1), LLMs can reason about
   whether an answer is actually helpful, relevant, and complete.

4. NATURAL LANGUAGE: LLMs can evaluate free-form outputs that don't have
   a single "correct answer."

LIMITATIONS OF LLM-AS-JUDGE:
- Can be overconfident or biased
- Might miss subtle issues a human would catch
- Quality depends on clear evaluation criteria in the prompt
- Should be complemented with human review for high-stakes applications

WHY ENFORCE JSON OUTPUT:
-----------------------
We ask the LLM to respond in strict JSON format for several reasons:

1. RELIABLE PARSING: We can use json.loads() to extract the score and reason
2. STRUCTURE ENFORCEMENT: The LLM must think in terms of our required fields
3. AUTOMATION: The Streamlit UI can programmatically display PASS vs FAIL

This is a common pattern called "structured output" or "function calling"
that makes LLM outputs machine-readable.

=============================================================================
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from state import AnalystState
from prompts.prompts import CRITIC_PROMPT

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
# BASE_URL allows using alternative OpenAI-compatible APIs (e.g., Azure, local LLMs)
# OPENAI_API_KEY is your authentication token for the API
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def critic_agent(state: AnalystState) -> dict:
    """
    Evaluate whether the analysis adequately addressed the user's question.
    
    This agent acts as quality assurance, using an LLM to determine if
    the final answer is relevant, specific, and complete.
    
    Parameters:
    -----------
    state : AnalystState
        The shared state dictionary. This agent reads from:
        - state["user_question"]: What the user originally asked
        - state["narrative"]: The explanation we generated
        - state["execution_result"]: The raw results for additional context
        
    Returns:
    --------
    dict
        A dictionary with updates to apply to the state:
        - "critic_score": Either "PASS" or "FAIL"
        - "critique": A one-sentence explanation of the score
        
    Evaluation Criteria:
    -------------------
    The LLM evaluates based on:
    - Relevance: Does the answer relate to what was asked?
    - Specificity: Are there concrete findings, not generic fluff?
    - Completeness: If the user asked for a chart, was one provided?
    - Accuracy: Any obvious errors or misunderstandings?
    """
    
    # =========================================================================
    # STEP 1: Extract inputs from state
    # =========================================================================
    # For evaluation, we need:
    # - user_question: What was asked (the "grading rubric")
    # - narrative: What we're evaluating (the final explanation)
    # - execution_result: Additional context (did execution succeed?)
    #
    # The user_question is crucial - it's what we compare against.
    # =========================================================================
    
    user_question = state["user_question"]
    narrative = state["narrative"]
    execution_result = state["execution_result"]
    
    # =========================================================================
    # STEP 2: Initialize the LLM client
    # =========================================================================
    # For evaluation, we use temperature=0 for determinism.
    # We want consistent scoring - the same input should get the same score.
    #
    # We also prefer a capable model (gpt-4o) because evaluation requires
    # nuanced judgment about relevance and completeness.
    # =========================================================================
    
    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key=OPENAI_API_KEY,
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0  # Deterministic for consistent evaluation
    )
    
    # =========================================================================
    # STEP 3: Build the evaluation prompt
    # =========================================================================
    # The CRITIC_PROMPT template:
    # - Presents the original question, narrative, and results
    # - Asks the LLM to evaluate relevance, specificity, completeness
    # - ENFORCES a strict JSON output format
    #
    # JSON ENFORCEMENT:
    # We explicitly tell the LLM to respond in JSON format because:
    # 1. We need to programmatically extract the score
    # 2. The Streamlit UI needs to show different colors for PASS/FAIL
    # 3. We store them in separate state fields
    # =========================================================================
    
    prompt = CRITIC_PROMPT.format(
        user_question=user_question,
        narrative=narrative,
        execution_result=execution_result
    )
    
    # =========================================================================
    # STEP 4: Call the LLM for evaluation
    # =========================================================================
    
    response = llm.invoke([HumanMessage(content=prompt)])
    raw_response = response.content
    
    # =========================================================================
    # STEP 5: Parse the JSON response
    # =========================================================================
    # The LLM should return something like:
    # {"score": "PASS", "reason": "The analysis correctly identified top products."}
    #
    # We parse this JSON to extract the score and reason separately.
    #
    # DEFENSIVE PARSING:
    # LLMs sometimes wrap JSON in markdown or include extra text.
    # We use a helper function to robustly extract the JSON.
    # If parsing fails completely, we default to FAIL with an explanation.
    # =========================================================================
    
    critique_result = parse_critic_response(raw_response)
    
    # =========================================================================
    # STEP 6: Return the evaluation results
    # =========================================================================
    # We return both:
    # - critic_score: "PASS" or "FAIL" for programmatic use
    # - critique: Human-readable reason for the score
    #
    # The Streamlit app will use these to show:
    # - Green "PASS" banner with reason, or
    # - Red "FAIL" banner with reason
    # =========================================================================
    
    return {
        "critic_score": critique_result["score"],
        "critique": critique_result["reason"]
    }


def parse_critic_response(response: str) -> dict:
    """
    Parse the critic LLM's JSON response into a structured dict.
    
    This function handles various edge cases that can occur when
    LLMs generate JSON output:
    - Clean JSON (ideal case)
    - JSON wrapped in markdown code blocks
    - JSON with extra text before/after
    - Malformed JSON (fallback to defaults)
    
    Parameters:
    -----------
    response : str
        The raw LLM response that should contain JSON
        
    Returns:
    --------
    dict
        A dictionary with keys:
        - "score": Either "PASS" or "FAIL"
        - "reason": A string explanation
        
    Examples:
    ---------
    >>> parse_critic_response('{"score": "PASS", "reason": "Good answer"}')
    {"score": "PASS", "reason": "Good answer"}
    
    >>> parse_critic_response('Sure! Here is my evaluation: {"score": "FAIL", "reason": "..."}')
    {"score": "FAIL", "reason": "..."}  # Extracts JSON from text
    """
    
    # =========================================================================
    # ATTEMPT 1: Try parsing the raw response as JSON
    # =========================================================================
    # Sometimes the LLM follows instructions perfectly and returns clean JSON.
    # =========================================================================
    
    try:
        result = json.loads(response.strip())
        return validate_critic_result(result)
    except json.JSONDecodeError:
        pass  # Try other extraction methods
    
    # =========================================================================
    # ATTEMPT 2: Try extracting JSON from markdown code block
    # =========================================================================
    # LLMs often wrap JSON in ```json ... ``` blocks despite instructions.
    # =========================================================================
    
    import re
    
    # Look for JSON in code blocks
    code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(code_block_pattern, response, re.DOTALL)
    
    if match:
        try:
            result = json.loads(match.group(1))
            return validate_critic_result(result)
        except json.JSONDecodeError:
            pass
    
    # =========================================================================
    # ATTEMPT 3: Try finding JSON object anywhere in the text
    # =========================================================================
    # Sometimes LLMs add explanatory text before/after the JSON.
    # We try to find the JSON object within the text.
    # =========================================================================
    
    json_pattern = r'\{[^{}]*"score"[^{}]*\}'
    match = re.search(json_pattern, response, re.DOTALL)
    
    if match:
        try:
            result = json.loads(match.group(0))
            return validate_critic_result(result)
        except json.JSONDecodeError:
            pass
    
    # =========================================================================
    # FALLBACK: Return default FAIL response
    # =========================================================================
    # If we can't parse the response at all, something went wrong.
    # We default to FAIL to be conservative (safer to flag issues than miss them).
    # =========================================================================
    
    return {
        "score": "FAIL",
        "reason": "Could not parse evaluation response. Please try again."
    }


def validate_critic_result(result: dict) -> dict:
    """
    Validate and normalize the parsed critic result.
    
    Ensures the result has the expected structure and values.
    
    Parameters:
    -----------
    result : dict
        The parsed JSON result
        
    Returns:
    --------
    dict
        A validated result with guaranteed "score" and "reason" keys
    """
    
    # Ensure score is present and valid
    score = result.get("score", "FAIL")
    if score not in ["PASS", "FAIL"]:
        score = "FAIL"  # Default to FAIL for safety
    
    # Ensure reason is present
    reason = result.get("reason", "No reason provided")
    
    return {
        "score": score,
        "reason": str(reason)
    }


# =============================================================================
# LLM-AS-JUDGE: DEEPER UNDERSTANDING
# =============================================================================
#
# WHEN TO USE LLM-AS-JUDGE:
#
# 1. OPEN-ENDED TASKS: When there's no single "correct" answer, LLMs can
#    evaluate whether the output is reasonable, helpful, and relevant.
#
# 2. QUALITY ASSURANCE: As a first pass filter before human review. The LLM
#    can flag obvious issues for human attention.
#
# 3. A/B TESTING: Comparing which of two outputs is better. LLMs can make
#    judgments like "Output A is more helpful because..."
#
# 4. AUTOMATED PIPELINES: When you need evaluation at scale without humans
#    in the loop.
#
# MAKING LLM-AS-JUDGE MORE RELIABLE:
#
# 1. CLEAR CRITERIA: Define exactly what PASS and FAIL mean. Don't leave
#    it to interpretation.
#
# 2. EXAMPLES: Provide examples of passing and failing outputs in the prompt.
#    Few-shot examples dramatically improve consistency.
#
# 3. STRUCTURED OUTPUT: Enforce JSON to get machine-readable results.
#    Don't ask for free-form evaluation text.
#
# 4. CALIBRATION: Periodically check LLM judgments against human judgments.
#    Adjust prompts if they diverge.
#
# 5. MULTIPLE JUDGES: For high-stakes decisions, use multiple LLMs or
#    multiple prompts and aggregate their votes.
#
# IN THIS APPLICATION:
#
# We use a simple PASS/FAIL schema for clarity. The critic checks:
# - Did we answer what was asked? (not just any question)
# - Did we provide specific information? (not vague generalizations)
# - Were there any obvious errors? (execution failures, wrong chart types)
#
# This helps users trust the results and flags when manual review is needed.
#
# =============================================================================
