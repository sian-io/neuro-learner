import tempfile
import os
from neuro_learner.models import ConceptMetadata
from neuro_learner.storage import (
    save_concept_index,
    list_saved_concepts,
    get_concept_by_id,
)

def test_storage_save_and_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_neuro.db")

        meta1 = ConceptMetadata(
            concept_id="gradient_descent",
            topic="Gradient Descent",
            stability=2.5,
            difficulty=5.0,
            reps=1,
            lapses=0,
            last_review="2026-08-30T12:00:00Z",
            next_review="2026-09-02T12:00:00Z",
        )
        save_concept_index(meta1, thread_id="concept_gradient_descent", db_path=db_path)

        concepts = list_saved_concepts(db_path=db_path)
        assert len(concepts) == 1
        assert concepts[0]["topic"] == "Gradient Descent"
        assert concepts[0]["stability"] == 2.5

        fetched = get_concept_by_id("gradient_descent", db_path=db_path)
        assert fetched is not None
        assert fetched["reps"] == 1

        # Test update
        meta1.reps = 2
        meta1.stability = 6.25
        save_concept_index(meta1, thread_id="concept_gradient_descent", db_path=db_path)

        updated_concepts = list_saved_concepts(db_path=db_path)
        assert len(updated_concepts) == 1
        assert updated_concepts[0]["reps"] == 2
        assert updated_concepts[0]["stability"] == 6.25
