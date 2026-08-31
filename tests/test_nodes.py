from unittest.mock import MagicMock
from neuro_learner.models import AgentState, GapAnalysis, RecallEvaluation, ConceptMetadata
from neuro_learner.nodes.feynman import feynman_evaluator
from neuro_learner.nodes.socratic import socratic_inquisitor
from neuro_learner.nodes.active_recall import active_recall_node, evaluate_recall_perf
from neuro_learner.llm import LLMClient

def test_feynman_evaluator_with_gaps():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_structured.return_value = GapAnalysis(
        has_gaps=True,
        missing_elements=["chain rule"],
        detected_jargon=["gradient descent"],
        logical_leaps=["weights update automatically"],
        confidence_score=0.9,
    )

    state: AgentState = {
        "topic": "Backpropagation",
        "user_explanation": "Gradient descent makes weights better automatically.",
        "dialogue_history": [],
        "concept_meta": ConceptMetadata(
            concept_id="backprop",
            topic="Backpropagation",
            last_review="",
            next_review="",
        ),
        "gap_analysis": None,
        "current_challenge": None,
        "user_response": None,
        "recall_grade": None,
        "next_step": "socratic_loop",
    }

    result = feynman_evaluator(state, llm=mock_llm)
    assert result["gap_analysis"].has_gaps is True
    assert result["next_step"] == "socratic_loop"

def test_feynman_evaluator_without_gaps():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_structured.return_value = GapAnalysis(
        has_gaps=False,
        missing_elements=[],
        detected_jargon=[],
        logical_leaps=[],
        confidence_score=0.98,
    )

    state: AgentState = {
        "topic": "Backpropagation",
        "user_explanation": "Backpropagation applies the calculus chain rule backwards from the loss function to compute partial derivatives of each weight, which gradient descent then uses to update parameters.",
        "dialogue_history": [],
        "concept_meta": ConceptMetadata(
            concept_id="backprop",
            topic="Backpropagation",
            last_review="",
            next_review="",
        ),
        "gap_analysis": None,
        "current_challenge": None,
        "user_response": None,
        "recall_grade": None,
        "next_step": "socratic_loop",
    }

    result = feynman_evaluator(state, llm=mock_llm)
    assert result["gap_analysis"].has_gaps is False
    assert result["next_step"] == "active_recall"

def test_socratic_inquisitor_node():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_text.return_value = "If weights update automatically, what role does the error gradient play when the network makes no mistakes?"

    state: AgentState = {
        "topic": "Backpropagation",
        "user_explanation": "Gradient descent updates weights automatically.",
        "dialogue_history": [],
        "concept_meta": ConceptMetadata(
            concept_id="backprop",
            topic="Backpropagation",
            last_review="",
            next_review="",
        ),
        "gap_analysis": GapAnalysis(
            has_gaps=True,
            missing_elements=["loss error relationship"],
            detected_jargon=[],
            logical_leaps=[],
            confidence_score=0.85,
        ),
        "current_challenge": None,
        "user_response": None,
        "recall_grade": None,
        "next_step": "socratic_loop",
    }

    result = socratic_inquisitor(state, llm=mock_llm)
    assert len(result["dialogue_history"]) == 1
    assert result["dialogue_history"][0]["type"] == "socratic_question"
    assert "If weights update automatically" in result["dialogue_history"][0]["content"]

def test_active_recall_node():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_text.return_value = "Imagine you have a deep network with vanishing gradients in the first layer. Explain how the chain rule formulation reveals why this happens."

    state: AgentState = {
        "topic": "Backpropagation",
        "user_explanation": "Valid explanation.",
        "dialogue_history": [],
        "concept_meta": ConceptMetadata(
            concept_id="backprop",
            topic="Backpropagation",
            last_review="",
            next_review="",
        ),
        "gap_analysis": GapAnalysis(has_gaps=False),
        "current_challenge": None,
        "user_response": None,
        "recall_grade": None,
        "next_step": "active_recall",
    }

    result = active_recall_node(state, llm=mock_llm)
    assert result["current_challenge"] is not None
    assert "vanishing gradients" in result["current_challenge"]

def test_evaluate_recall_perf():
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_structured.return_value = RecallEvaluation(
        grade="Good",
        feedback="Clear explanation of matrix multiplication and compounding derivatives.",
        rationale="Accurately linked derivative products to vanishing values.",
    )

    state: AgentState = {
        "topic": "Backpropagation",
        "user_explanation": "Valid explanation.",
        "dialogue_history": [],
        "concept_meta": ConceptMetadata(
            concept_id="backprop",
            topic="Backpropagation",
            last_review="",
            next_review="",
        ),
        "gap_analysis": GapAnalysis(has_gaps=False),
        "current_challenge": "Explain vanishing gradients via chain rule.",
        "user_response": "Since the chain rule multiplies derivatives across layers, multiplying many terms < 1 causes the gradient to shrink exponentially.",
        "recall_grade": None,
        "next_step": "active_recall",
    }

    result = evaluate_recall_perf(state, llm=mock_llm)
    assert result["recall_grade"] == "Good"
    assert result["next_step"] == "schedule"
