from neuro_learner.models import AgentState, GapAnalysis
from neuro_learner.llm import default_llm_client, LLMClient

FEYNMAN_SYSTEM_PROMPT = """You are a rigorous metacognitive evaluator. Your task is to deconstruct the explanation provided by the student on the defined topic.
Identify:
1. Technical jargon used as a crutch without elementary explanation.
2. Cause-and-effect leaps.
3. Incorrect premises or half-truths.
Strictly return a structured JSON following the GapAnalysis schema."""

def feynman_evaluator(state: AgentState, llm: LLMClient | None = None) -> dict:
    client = llm or default_llm_client
    topic = state.get("topic", "")
    explanation = state.get("user_explanation", "")
    history = state.get("dialogue_history", [])

    history_context = ""
    if history:
        history_context = "\nDialogue Context:\n" + "\n".join(
            f"- {turn.get('role', 'user')}: {turn.get('content', '')}" for turn in history
        )

    prompt = f"""Topic: {topic}

Student's Explanation:
\"\"\"{explanation}\"\"\"
{history_context}

Analyze whether this explanation demonstrates genuine foundational understanding without relying on undefined jargon, logical leaps, or factual errors."""

    gap_analysis = client.generate_structured(
        prompt=prompt,
        schema=GapAnalysis,
        system_instruction=FEYNMAN_SYSTEM_PROMPT,
    )

    next_step = "socratic_loop" if gap_analysis.has_gaps else "active_recall"
    return {
        "gap_analysis": gap_analysis,
        "next_step": next_step,
    }
