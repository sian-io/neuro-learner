from datetime import datetime, timezone, timedelta
from neuro_learner.models import ConceptMetadata, AgentState
from neuro_learner.nodes.scheduler import calculate_fsrs_update, fsrs_scheduler_node

def test_fsrs_again_grade():
    now = datetime(2026, 8, 30, 12, 0, 0, tzinfo=timezone.utc)
    meta = ConceptMetadata(
        concept_id="test",
        topic="Testing",
        stability=2.0,
        difficulty=5.0,
        reps=3,
        lapses=0,
        last_review="2026-08-20T12:00:00Z",
        next_review="2026-08-30T12:00:00Z",
    )
    updated, interval = calculate_fsrs_update(meta, "Again", now=now)
    assert updated.stability == 0.4  # 2.0 * 0.2
    assert updated.lapses == 1
    assert updated.reps == 3
    assert interval == 0.0
    assert updated.next_review == now.isoformat()

def test_fsrs_hard_grade():
    now = datetime(2026, 8, 30, 12, 0, 0, tzinfo=timezone.utc)
    meta = ConceptMetadata(
        concept_id="test",
        topic="Testing",
        stability=2.0,
        difficulty=5.0,
        reps=1,
        lapses=0,
        last_review="2026-08-20T12:00:00Z",
        next_review="2026-08-30T12:00:00Z",
    )
    updated, interval = calculate_fsrs_update(meta, "Hard", now=now)
    assert updated.stability == 2.4  # 2.0 * 1.2
    assert updated.difficulty == 5.5  # 5.0 + 0.5
    assert updated.reps == 2
    assert interval == 2.4

def test_fsrs_good_grade():
    now = datetime(2026, 8, 30, 12, 0, 0, tzinfo=timezone.utc)
    meta = ConceptMetadata(
        concept_id="test",
        topic="Testing",
        stability=2.0,
        difficulty=5.0,
        reps=1,
        lapses=0,
        last_review="2026-08-20T12:00:00Z",
        next_review="2026-08-30T12:00:00Z",
    )
    updated, interval = calculate_fsrs_update(meta, "Good", now=now)
    assert updated.stability == 5.0  # 2.0 * 2.5
    assert updated.difficulty == 5.0
    assert updated.reps == 2
    assert interval == 5.0
    expected_next = (now + timedelta(days=5.0)).isoformat()
    assert updated.next_review == expected_next

def test_fsrs_easy_grade_and_difficulty_bounds():
    now = datetime(2026, 8, 30, 12, 0, 0, tzinfo=timezone.utc)
    meta = ConceptMetadata(
        concept_id="test",
        topic="Testing",
        stability=1.0,
        difficulty=1.2,
        reps=0,
        lapses=0,
        last_review="2026-08-20T12:00:00Z",
        next_review="2026-08-30T12:00:00Z",
    )
    updated, interval = calculate_fsrs_update(meta, "Easy", now=now)
    assert updated.stability == 4.0  # 1.0 * 4.0
    assert updated.difficulty == 1.0  # max(1.0, 1.2 - 0.5) floored at 1.0
    assert updated.reps == 1
    assert interval == 4.0

def test_fsrs_scheduler_node_execution():
    state: AgentState = {
        "topic": "Neural Networks",
        "user_explanation": "A network of layers with weights.",
        "dialogue_history": [],
        "concept_meta": None,
        "gap_analysis": None,
        "current_challenge": None,
        "user_response": None,
        "recall_grade": "Good",
        "next_step": "schedule",
    }
    result = fsrs_scheduler_node(state)
    assert result["next_step"] == "complete"
    assert result["concept_meta"].stability == 2.5
    assert result["concept_meta"].reps == 1
    assert len(result["dialogue_history"]) == 1
