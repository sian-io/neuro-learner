from neuro_learner.models import AgentState, RecallEvaluation
from neuro_learner.llm import default_llm_client, LLMClient

ACTIVE_RECALL_SYSTEM_PROMPT = """You are an expert cognitive learning designer applying Bloom's Taxonomy (Analysis, Evaluation, or Synthesis).
When the student demonstrates foundational understanding of the concept:
1. Design a challenging, realistic situational problem, scenario debugging task, or architectural trade-off justification.
2. NEVER produce multiple-choice questions or trivial definition checks.
3. Require the student to apply, critique, or synthesize the principles in an unfamiliar context."""

RECALL_EVAL_SYSTEM_PROMPT = """You are an objective evaluator grading active recall performance based on cognitive mastery criteria:
- 'Again': Severe retrieval failure, major hallucinations, or incorrect fundamentals.
- 'Hard': Correct core ideas but requires significant effort, imprecise rationale, or missed nuances.
- 'Good': Solid, accurate, and concise response within expectations.
- 'Easy': Complete mastery, immediate synthesis, intuitive handling of edge cases and trade-offs.

Output structured JSON matching the RecallEvaluation schema."""

def active_recall_node(state: AgentState, llm: LLMClient | None = None) -> dict:
    client = llm or default_llm_client
    topic = state.get("topic", "")
    explanation = state.get("user_explanation", "")

    prompt = f"""Topic: {topic}
Verified Understanding from Student:
\"\"\"{explanation}\"\"\"

Generate an advanced Bloom's Taxonomy challenge (Analysis, Evaluation, or Synthesis level) to test active retrieval practice."""

    challenge = client.generate_text(
        prompt=prompt,
        system_instruction=ACTIVE_RECALL_SYSTEM_PROMPT,
    )

    history = list(state.get("dialogue_history", []))
    history.append({
        "role": "assistant",
        "type": "active_recall_challenge",
        "content": challenge,
    })

    return {
        "current_challenge": challenge,
        "dialogue_history": history,
        "next_step": "active_recall",
    }

def evaluate_recall_perf(state: AgentState, llm: LLMClient | None = None) -> dict:
    client = llm or default_llm_client
    topic = state.get("topic", "")
    challenge = state.get("current_challenge", "")
    user_response = state.get("user_response", "")

    prompt = f"""Topic: {topic}

Challenge Presented:
\"\"\"{challenge}\"\"\"

Student's Response:
\"\"\"{user_response}\"\"\"

Evaluate the response accuracy, depth, and mastery. Assign a grade strictly among: 'Again', 'Hard', 'Good', 'Easy' with feedback and rationale."""

    eval_result = client.generate_structured(
        prompt=prompt,
        schema=RecallEvaluation,
        system_instruction=RECALL_EVAL_SYSTEM_PROMPT,
    )

    history = list(state.get("dialogue_history", []))
    history.append({
        "role": "assistant",
        "type": "evaluation_feedback",
        "content": f"[{eval_result.grade}] {eval_result.feedback}",
    })

    next_step = "socratic_loop" if eval_result.grade == "Again" else "schedule"

    return {
        "recall_grade": eval_result.grade,
        "dialogue_history": history,
        "next_step": next_step,
    }
