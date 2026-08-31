from datetime import datetime, timedelta, timezone
from neuro_learner.models import AgentState, ConceptMetadata

def calculate_fsrs_update(
    meta: ConceptMetadata,
    grade: str,
    now: datetime | None = None,
) -> tuple[ConceptMetadata, float]:
    current_time = now or datetime.now(timezone.utc)
    stability = meta.stability
    difficulty = meta.difficulty
    reps = meta.reps
    lapses = meta.lapses

    if grade == "Again":
        stability = max(0.1, stability * 0.2)
        lapses += 1
        interval_days = 0.0
    elif grade == "Hard":
        stability = stability * 1.2
        difficulty = min(10.0, difficulty + 0.5)
        reps += 1
        interval_days = max(1.0, stability)
    elif grade == "Good":
        stability = stability * 2.5
        reps += 1
        interval_days = max(1.0, stability)
    elif grade == "Easy":
        stability = stability * 4.0
        difficulty = max(1.0, difficulty - 0.5)
        reps += 1
        interval_days = max(1.0, stability)
    else:
        interval_days = max(1.0, stability)

    last_review_str = current_time.isoformat()
    next_review_dt = current_time + timedelta(days=interval_days)
    next_review_str = next_review_dt.isoformat()

    updated_meta = ConceptMetadata(
        concept_id=meta.concept_id,
        topic=meta.topic,
        stability=round(stability, 4),
        difficulty=round(difficulty, 4),
        reps=reps,
        lapses=lapses,
        last_review=last_review_str,
        next_review=next_review_str,
    )
    return updated_meta, interval_days

def fsrs_scheduler_node(state: AgentState) -> dict:
    meta = state.get("concept_meta")
    grade = state.get("recall_grade") or "Good"
    topic = state.get("topic", "")

    if meta is None:
        now_str = datetime.now(timezone.utc).isoformat()
        meta = ConceptMetadata(
            concept_id=topic.lower().replace(" ", "_"),
            topic=topic,
            stability=1.0,
            difficulty=5.0,
            reps=0,
            lapses=0,
            last_review=now_str,
            next_review=now_str,
        )

    updated_meta, _ = calculate_fsrs_update(meta, grade)

    history = list(state.get("dialogue_history", []))
    history.append({
        "role": "system",
        "type": "schedule_update",
        "content": (
            f"FSRS Review Scheduled. Next Review: {updated_meta.next_review} | "
            f"Stability: {updated_meta.stability}d | Difficulty: {updated_meta.difficulty}/10 | "
            f"Reps: {updated_meta.reps} | Lapses: {updated_meta.lapses}"
        ),
    })

    return {
        "concept_meta": updated_meta,
        "dialogue_history": history,
        "next_step": "complete",
    }
