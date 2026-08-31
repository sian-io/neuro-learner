from neuro_learner.models import ConceptMetadata, GapAnalysis, RecallEvaluation, AgentState

def test_concept_metadata_defaults():
    meta = ConceptMetadata(
        concept_id="backprop",
        topic="Backpropagation",
        last_review="2026-08-30T12:00:00Z",
        next_review="2026-08-31T12:00:00Z",
    )
    assert meta.stability == 1.0
    assert meta.difficulty == 5.0
    assert meta.reps == 0
    assert meta.lapses == 0

def test_gap_analysis_serialization():
    gap = GapAnalysis(
        has_gaps=True,
        missing_elements=["chain rule application", "loss gradient"],
        detected_jargon=["gradient descent"],
        logical_leaps=["weights update magically"],
        confidence_score=0.95,
    )
    data = gap.model_dump()
    assert data["has_gaps"] is True
    assert len(data["missing_elements"]) == 2
    assert "gradient descent" in data["detected_jargon"]

    restored = GapAnalysis.model_validate(data)
    assert restored.confidence_score == 0.95
    assert restored.has_gaps is True

def test_recall_evaluation_valid_grades():
    eval_good = RecallEvaluation(
        grade="Good",
        feedback="Solid understanding.",
        rationale="Explained gradient computation clearly.",
    )
    assert eval_good.grade == "Good"

def test_agent_state_structure():
    meta = ConceptMetadata(
        concept_id="test",
        topic="Test Topic",
        last_review="2026-08-30T12:00:00Z",
        next_review="2026-08-30T12:00:00Z",
    )
    state: AgentState = {
        "topic": "Test Topic",
        "user_explanation": "Test explanation",
        "dialogue_history": [],
        "concept_meta": meta,
        "gap_analysis": None,
        "current_challenge": None,
        "user_response": None,
        "recall_grade": None,
        "next_step": "socratic_loop",
    }
    assert state["topic"] == "Test Topic"
    assert state["next_step"] == "socratic_loop"
