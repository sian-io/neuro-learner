import argparse
import sys
from datetime import datetime, timezone
from neuro_learner.config import settings
from neuro_learner.models import AgentState, ConceptMetadata
from neuro_learner.graph import create_learner_graph, get_sqlite_checkpointer
from neuro_learner.nodes.active_recall import evaluate_recall_perf
from neuro_learner.nodes.scheduler import fsrs_scheduler_node
from neuro_learner.storage import (
    save_concept_index,
    list_saved_concepts,
    get_concept_by_id,
)

def format_topic_slug(topic: str) -> str:
    return topic.strip().lower().replace(" ", "_")

def display_saved_concepts(concepts: list[dict]) -> None:
    print("\nSaved Topics in Persistence DB:")
    print("-" * 70)
    for idx, c in enumerate(concepts, start=1):
        print(f"[{idx}] {c['topic']}")
        print(f"    Stability: {c['stability']}d | Difficulty: {c['difficulty']}/10.0 | Reps: {c['reps']} | Lapses: {c['lapses']}")
        print(f"    Last Review: {c['last_review']} | Next Review: {c['next_review']}")
    print("-" * 70)

def run_study_session(
    topic: str,
    db_path: str | None = None,
    resume: bool = False,
) -> None:
    checkpointer = get_sqlite_checkpointer(db_path)
    graph = create_learner_graph(checkpointer=checkpointer)

    concept_id = format_topic_slug(topic)
    thread_id = f"concept_{concept_id}"
    thread_config = {"configurable": {"thread_id": thread_id}}

    state: AgentState | None = None

    if resume:
        saved_snapshot = graph.get_state(thread_config)
        if saved_snapshot and saved_snapshot.values:
            state = dict(saved_snapshot.values)  # type: ignore

    now_iso = datetime.now(timezone.utc).isoformat()

    if state is None:
        # Check if concept metadata exists in storage index
        stored_concept = get_concept_by_id(concept_id, db_path)
        if stored_concept:
            meta = ConceptMetadata(
                concept_id=stored_concept["concept_id"],
                topic=stored_concept["topic"],
                stability=stored_concept["stability"],
                difficulty=stored_concept["difficulty"],
                reps=stored_concept["reps"],
                lapses=stored_concept["lapses"],
                last_review=stored_concept["last_review"],
                next_review=stored_concept["next_review"],
            )
        else:
            meta = ConceptMetadata(
                concept_id=concept_id,
                topic=topic,
                stability=1.0,
                difficulty=5.0,
                reps=0,
                lapses=0,
                last_review=now_iso,
                next_review=now_iso,
            )

        print("=" * 60)
        print(f"NeuroLearner — Metacognitive Session: {topic}")
        print("=" * 60)
        print("Step 1: Feynman Technique (Explain the concept in your own words)")
        print("-" * 60)

        try:
            user_input = input("\nYour explanation:\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession aborted.")
            return

        if not user_input:
            print("Explanation cannot be empty. Session aborted.")
            return

        state = {
            "topic": topic,
            "user_explanation": user_input,
            "dialogue_history": [{"role": "user", "type": "explanation", "content": user_input}],
            "concept_meta": meta,
            "gap_analysis": None,
            "current_challenge": None,
            "user_response": None,
            "recall_grade": None,
            "next_step": "socratic_loop",
        }
    else:
        print("=" * 60)
        print(f"Resuming Previous Session: {state.get('topic', topic)}")
        print("=" * 60)
        meta = state.get("concept_meta")
        if meta:
            print(f"Current Stats: Stability {meta.stability}d | Reps {meta.reps} | Lapses {meta.lapses}")
        
        # Show last dialogue entries
        history = state.get("dialogue_history", [])
        if history:
            print("\n--- Recent Dialogue Context ---")
            for item in history[-3:]:
                role = item.get("role", "user").capitalize()
                content = item.get("content", "")
                print(f"[{role}]: {content}")
            print("-" * 30)

        # If previous session was completed, ask for a new explanation to initiate review
        if state.get("next_step") == "complete":
            print("\nThis topic was previously completed. Starting a new review cycle.")
            print("Explain the concept in your own words (Feynman Technique):")
            try:
                user_input = input("\nYour explanation:\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nSession aborted.")
                return

            if not user_input:
                print("Explanation cannot be empty. Session aborted.")
                return

            state["user_explanation"] = user_input
            history.append({"role": "user", "type": "explanation", "content": user_input})
            state["dialogue_history"] = history
            state["next_step"] = "socratic_loop"

    # Socratic dialogue loop
    while True:
        result = graph.invoke(state, thread_config)
        state.update(result)

        gap_analysis = state.get("gap_analysis")
        if gap_analysis and gap_analysis.has_gaps:
            print("\n[Knowledge Gap Detected]")
            if gap_analysis.detected_jargon:
                print(f" - Unanchored Jargon: {', '.join(gap_analysis.detected_jargon)}")
            if gap_analysis.logical_leaps:
                print(f" - Logical Leaps: {', '.join(gap_analysis.logical_leaps)}")
            if gap_analysis.missing_elements:
                print(f" - Missing Elements: {', '.join(gap_analysis.missing_elements)}")

            latest_question = ""
            for item in reversed(state.get("dialogue_history", [])):
                if item.get("type") == "socratic_question":
                    latest_question = item.get("content", "")
                    break

            print(f"\nSocratic Inquisitor:\n{latest_question}\n")

            try:
                clarification = input("Your clarification:\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nSession aborted.")
                return

            state["user_explanation"] = f"{state['user_explanation']}\nClarification: {clarification}"
            history = list(state.get("dialogue_history", []))
            history.append({"role": "user", "type": "clarification", "content": clarification})
            state["dialogue_history"] = history
        else:
            print("\n[Feynman Check Passed: Foundational understanding verified]")
            break

    # Active Recall Stage
    challenge = state.get("current_challenge", "")
    print("\n" + "=" * 60)
    print("Step 2: Active Recall Challenge (Bloom's Taxonomy)")
    print("=" * 60)
    print(f"{challenge}\n")

    try:
        recall_ans = input("Your solution / response:\n> ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nSession aborted.")
        return

    state["user_response"] = recall_ans
    history = list(state.get("dialogue_history", []))
    history.append({"role": "user", "type": "recall_response", "content": recall_ans})
    state["dialogue_history"] = history

    # Evaluate recall performance and schedule review
    eval_output = evaluate_recall_perf(state)
    state.update(eval_output)

    schedule_output = fsrs_scheduler_node(state)
    state.update(schedule_output)

    # Save to SQLite index
    updated_meta = state["concept_meta"]
    save_concept_index(updated_meta, thread_id=thread_id, db_path=db_path)

    # Persist final state in graph checkpointer
    graph.update_state(thread_config, state)

    print("\n" + "=" * 60)
    print("Step 3: Performance Evaluation & Spaced Repetition Scheduling")
    print("=" * 60)
    print(f"Grade Assigned : {state.get('recall_grade')}")
    for item in reversed(state.get("dialogue_history", [])):
        if item.get("type") == "evaluation_feedback":
            print(f"Feedback       : {item.get('content')}")
            break

    print("\n--- FSRS Memory Metrics ---")
    print(f"Stability (S)  : {updated_meta.stability} days")
    print(f"Difficulty (D) : {updated_meta.difficulty} / 10.0")
    print(f"Total Reps     : {updated_meta.reps}")
    print(f"Total Lapses   : {updated_meta.lapses}")
    print(f"Last Review    : {updated_meta.last_review}")
    print(f"Next Review    : {updated_meta.next_review}")
    print("=" * 60)

def interactive_menu(db_path: str | None = None) -> None:
    while True:
        print("\n" + "=" * 60)
        print("NeuroLearner — Metacognitive Study System")
        print("=" * 60)
        print("1. Start a new study session (define a new topic)")
        print("2. Load a previous conversation / topic")
        print("3. List all saved topics")
        print("4. Exit")
        print("-" * 60)

        try:
            choice = input("Select an option [1-4]: > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if choice == "1":
            try:
                topic = input("\nEnter the concept/topic to study:\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if topic:
                run_study_session(topic=topic, db_path=db_path, resume=False)
            else:
                print("Topic cannot be empty.")
        elif choice == "2":
            saved = list_saved_concepts(db_path)
            if not saved:
                print("\nNo saved topics found. Please start a new study session.")
                continue

            display_saved_concepts(saved)
            try:
                selected = input(f"\nSelect a topic number to load [1-{len(saved)}], or 0 to cancel:\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if selected.isdigit():
                idx = int(selected)
                if 1 <= idx <= len(saved):
                    chosen_topic = saved[idx - 1]["topic"]
                    run_study_session(topic=chosen_topic, db_path=db_path, resume=True)
                elif idx == 0:
                    continue
                else:
                    print("Invalid selection.")
            else:
                print("Invalid input.")
        elif choice == "3":
            saved = list_saved_concepts(db_path)
            if not saved:
                print("\nNo saved topics found.")
            else:
                display_saved_concepts(saved)
        elif choice == "4" or choice.lower() in ["exit", "quit", "q"]:
            print("\nGoodbye.")
            break
        else:
            print("Invalid option. Please choose 1, 2, 3, or 4.")

def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroLearner Metacognitive Agent CLI")
    parser.add_argument("--topic", type=str, default=None, help="Start or review a specific topic")
    parser.add_argument("--load", type=str, default=None, help="Load and resume a previously saved topic")
    parser.add_argument("--list", action="store_true", help="List all saved topics in persistence DB")
    parser.add_argument("--db-path", type=str, default=settings.persistence_db_path, help="Path to SQLite persistence DB")
    args = parser.parse_args()

    if args.list:
        saved = list_saved_concepts(args.db_path)
        if not saved:
            print("No saved topics found.")
        else:
            display_saved_concepts(saved)
        return

    if args.load:
        run_study_session(topic=args.load, db_path=args.db_path, resume=True)
        return

    if args.topic:
        run_study_session(topic=args.topic, db_path=args.db_path, resume=False)
        return

    # Default to interactive menu
    interactive_menu(db_path=args.db_path)

if __name__ == "__main__":
    main()
