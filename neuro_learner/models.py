from typing import Literal, TypedDict
from pydantic import BaseModel, Field

class ConceptMetadata(BaseModel):
    concept_id: str
    topic: str
    stability: float = Field(default=1.0, description="FSRS Stability factor (in days)")
    difficulty: float = Field(default=5.0, description="FSRS Difficulty (1.0 to 10.0)")
    reps: int = Field(default=0, description="Total successful reviews")
    lapses: int = Field(default=0, description="Total failures/re-explanations needed")
    last_review: str
    next_review: str

class GapAnalysis(BaseModel):
    has_gaps: bool
    missing_elements: list[str] = Field(default_factory=list)
    detected_jargon: list[str] = Field(default_factory=list)
    logical_leaps: list[str] = Field(default_factory=list)
    confidence_score: float = Field(default=1.0)

class RecallEvaluation(BaseModel):
    grade: Literal["Again", "Hard", "Good", "Easy"]
    feedback: str
    rationale: str

class AgentState(TypedDict):
    topic: str
    user_explanation: str
    dialogue_history: list[dict[str, str]]
    concept_meta: ConceptMetadata
    gap_analysis: GapAnalysis | None
    current_challenge: str | None
    user_response: str | None
    recall_grade: Literal["Again", "Hard", "Good", "Easy"] | None
    next_step: Literal["socratic_loop", "active_recall", "schedule", "complete"]
