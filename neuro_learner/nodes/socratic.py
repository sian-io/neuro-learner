from neuro_learner.models import AgentState
from neuro_learner.llm import default_llm_client, LLMClient

SOCRATIC_SYSTEM_PROMPT = """You are a Socratic tutor guiding a student to uncover gaps in their own understanding.
Based on the identified gaps, technical jargon, or logical leaps:
1. Generate a sharp counter-example (Reductio ad Absurdum) or a practical dilemma.
2. DO NOT provide the solution, correction, or direct answer.
3. Formulate an open, challenging question that compels the student to realize the contradiction or ambiguity in their premise."""

def socratic_inquisitor(state: AgentState, llm: LLMClient | None = None) -> dict:
    client = llm or default_llm_client
    topic = state.get("topic", "")
    explanation = state.get("user_explanation", "")
    gap_analysis = state.get("gap_analysis")
    history = list(state.get("dialogue_history", []))

    missing = gap_analysis.missing_elements if gap_analysis else []
    jargon = gap_analysis.detected_jargon if gap_analysis else []
    leaps = gap_analysis.logical_leaps if gap_analysis else []

    prompt = f"""Topic: {topic}

Student's Explanation:
\"\"\"{explanation}\"\"\"

Identified Issues:
- Missing Elements: {', '.join(missing) if missing else 'None'}
- Jargon without explanation: {', '.join(jargon) if jargon else 'None'}
- Logical Leaps / Flaws: {', '.join(leaps) if leaps else 'None'}

Formulate a Socratic question or counter-scenario to guide the student to discover their error."""

    question = client.generate_text(
        prompt=prompt,
        system_instruction=SOCRATIC_SYSTEM_PROMPT,
    )

    history.append({
        "role": "assistant",
        "type": "socratic_question",
        "content": question,
    })

    return {
        "dialogue_history": history,
        "next_step": "socratic_loop",
    }
