import sqlite3
from unittest.mock import patch, MagicMock
from langgraph.checkpoint.sqlite import SqliteSaver
from neuro_learner.models import AgentState, ConceptMetadata, GapAnalysis, RecallEvaluation
from neuro_learner.graph import create_learner_graph

def test_graph_flow_with_gaps():
    with patch("neuro_learner.nodes.feynman.default_llm_client") as mock_feynman_llm, \
         patch("neuro_learner.nodes.socratic.default_llm_client") as mock_socratic_llm:

        mock_feynman_llm.generate_structured.return_value = GapAnalysis(
            has_gaps=True,
            missing_elements=["gradient formula"],
            detected_jargon=["backprop"],
            logical_leaps=["it optimizes"],
            confidence_score=0.9,
        )
        mock_socratic_llm.generate_text.return_value = "What is the mathematical definition of backpropagation error?"

        graph = create_learner_graph()
        initial_state: AgentState = {
            "topic": "Backpropagation",
            "user_explanation": "It optimizes neural nets with backprop.",
            "dialogue_history": [],
            "concept_meta": ConceptMetadata(
                concept_id="backprop",
                topic="Backpropagation",
                last_review="2026-08-30T12:00:00Z",
                next_review="2026-08-30T12:00:00Z",
            ),
            "gap_analysis": None,
            "current_challenge": None,
            "user_response": None,
            "recall_grade": None,
            "next_step": "socratic_loop",
        }

        output = graph.invoke(initial_state)
        assert output["gap_analysis"].has_gaps is True
        assert any(turn["type"] == "socratic_question" for turn in output["dialogue_history"])

def test_graph_flow_without_gaps():
    with patch("neuro_learner.nodes.feynman.default_llm_client") as mock_feynman_llm, \
         patch("neuro_learner.nodes.active_recall.default_llm_client") as mock_recall_llm:

        mock_feynman_llm.generate_structured.return_value = GapAnalysis(
            has_gaps=False,
            missing_elements=[],
            detected_jargon=[],
            logical_leaps=[],
            confidence_score=0.99,
        )
        mock_recall_llm.generate_text.return_value = "Design a dynamic learning rate schedule based on gradient norms."

        graph = create_learner_graph()
        initial_state: AgentState = {
            "topic": "Gradient Descent",
            "user_explanation": "Gradient descent computes the loss gradient with respect to parameters and steps in the negative gradient direction scaled by a learning rate.",
            "dialogue_history": [],
            "concept_meta": ConceptMetadata(
                concept_id="grad_desc",
                topic="Gradient Descent",
                last_review="2026-08-30T12:00:00Z",
                next_review="2026-08-30T12:00:00Z",
            ),
            "gap_analysis": None,
            "current_challenge": None,
            "user_response": None,
            "recall_grade": None,
            "next_step": "socratic_loop",
        }

        output = graph.invoke(initial_state)
        assert output["gap_analysis"].has_gaps is False
        assert output["current_challenge"] == "Design a dynamic learning rate schedule based on gradient norms."

def test_graph_sqlite_persistence():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = create_learner_graph(checkpointer=checkpointer)

    with patch("neuro_learner.nodes.feynman.default_llm_client") as mock_feynman_llm, \
         patch("neuro_learner.nodes.active_recall.default_llm_client") as mock_recall_llm:

        mock_feynman_llm.generate_structured.return_value = GapAnalysis(has_gaps=False)
        mock_recall_llm.generate_text.return_value = "Analyze vanishing gradients."

        config = {"configurable": {"thread_id": "thread_persistence_test"}}
        initial_state: AgentState = {
            "topic": "Backpropagation",
            "user_explanation": "Solid explanation.",
            "dialogue_history": [],
            "concept_meta": ConceptMetadata(
                concept_id="backprop",
                topic="Backpropagation",
                last_review="2026-08-30T12:00:00Z",
                next_review="2026-08-30T12:00:00Z",
            ),
            "gap_analysis": None,
            "current_challenge": None,
            "user_response": None,
            "recall_grade": None,
            "next_step": "socratic_loop",
        }

        output = graph.invoke(initial_state, config)
        assert output["current_challenge"] == "Analyze vanishing gradients."

        # Fetch state from checkpointer
        saved_state = graph.get_state(config)
        assert saved_state.values["current_challenge"] == "Analyze vanishing gradients."
