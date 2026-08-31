import sqlite3
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from neuro_learner.models import AgentState
from neuro_learner.config import settings
from neuro_learner.nodes import (
    feynman_evaluator,
    socratic_inquisitor,
    active_recall_node,
    evaluate_recall_perf,
    fsrs_scheduler_node,
)

def has_knowledge_gap(state: AgentState) -> Literal["socratic_inquisitor", "active_recall_node"]:
    gap = state.get("gap_analysis")
    if gap is not None and gap.has_gaps:
        return "socratic_inquisitor"
    return "active_recall_node"

def route_recall_eval(state: AgentState) -> Literal["socratic_inquisitor", "fsrs_scheduler_node"]:
    grade = state.get("recall_grade")
    if grade == "Again":
        return "socratic_inquisitor"
    return "fsrs_scheduler_node"

def create_learner_graph(checkpointer: SqliteSaver | None = None):
    builder = StateGraph(AgentState)

    builder.add_node("feynman_evaluator", feynman_evaluator)
    builder.add_node("socratic_inquisitor", socratic_inquisitor)
    builder.add_node("active_recall_node", active_recall_node)
    builder.add_node("evaluate_recall_perf", evaluate_recall_perf)
    builder.add_node("fsrs_scheduler_node", fsrs_scheduler_node)

    # Initial flow
    builder.add_edge(START, "feynman_evaluator")
    builder.add_conditional_edges(
        "feynman_evaluator",
        has_knowledge_gap,
        {
            "socratic_inquisitor": "socratic_inquisitor",
            "active_recall_node": "active_recall_node",
        },
    )
    builder.add_edge("socratic_inquisitor", END)
    builder.add_edge("active_recall_node", END)
    builder.add_conditional_edges(
        "evaluate_recall_perf",
        route_recall_eval,
        {
            "socratic_inquisitor": "socratic_inquisitor",
            "fsrs_scheduler_node": "fsrs_scheduler_node",
        },
    )
    builder.add_edge("fsrs_scheduler_node", END)

    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()

def get_sqlite_checkpointer(db_path: str | None = None) -> SqliteSaver:
    target_path = db_path or settings.persistence_db_path
    settings.ensure_db_dir()
    conn = sqlite3.connect(target_path, check_same_thread=False)
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("neuro_learner.models", "ConceptMetadata"),
            ("neuro_learner.models", "GapAnalysis"),
            ("neuro_learner.models", "RecallEvaluation"),
        ]
    )
    return SqliteSaver(conn, serde=serde)
