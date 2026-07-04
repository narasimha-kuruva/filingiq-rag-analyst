"""
Comprehensive Retrieval and DB Test for FilingIQ.

Processes sample files, indexes them into a test collection,
performs similarity searches, and verifies that the dual-index
retrieval and similarity scoring logic operates correctly.

Run:
    python test_retrieval.py
"""

import os
import sys
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from config import SIMILARITY_THRESHOLD
from ingestion.router import ingest_file
from retrieval.embedder import get_embedding_function
from retrieval.vector_store import DualVectorStore
from retrieval.retriever import retrieve

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_CHROMA_DIR = os.path.join(os.path.dirname(__file__), "test_chroma_db")


def setup_test_db(vector_store: DualVectorStore):
    """Load sample files and insert them into the DB."""
    files_to_load = [
        ("apple_q4_2024_earnings_notes.txt", "narrative"),
        ("tech_company_ratios.csv", "structured"),
    ]

    print("\nIngesting and Indexing test files...")
    for filename, target in files_to_load:
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"❌ Test file missing: {filepath}. Run generate_sample_data.py first.")
            return False

        print(f"  Loading {filename} -> {target}...")
        docs, collection_target = ingest_file(filepath, filename)
        vector_store.add_documents(docs, collection_target)

    return True


def run_tests():
    # Clean old test database
    if os.path.exists(TEST_CHROMA_DIR):
        shutil.rmtree(TEST_CHROMA_DIR)

    # Temporarily override configuration persist dir for testing
    import config
    original_persist_dir = config.CHROMA_PERSIST_DIR
    config.CHROMA_PERSIST_DIR = TEST_CHROMA_DIR

    try:
        print("Initializing Embeddings and DualVectorStore...")
        embedding_fn = get_embedding_function()
        vector_store = DualVectorStore(embedding_fn)

        # Clear just in case
        vector_store.clear_all()

        if not setup_test_db(vector_store):
            return

        # ── Test counts ────────────────────────────────────────────────
        narrative_count = vector_store.get_document_count("narrative")
        structured_count = vector_store.get_document_count("structured")
        print(f"\nIndexed counts:")
        print(f"  Narrative collection: {narrative_count} docs")
        print(f"  Structured collection: {structured_count} docs")

        assert narrative_count > 0, "Narrative store should not be empty!"
        assert structured_count > 0, "Structured store should not be empty!"

        # ── Test Idempotency (Delete-before-insert & deterministic IDs) ──
        print("\nTesting Idempotency (uploading the same files again)...")
        # Ingest and index again
        docs, collection_target = ingest_file(os.path.join(DATA_DIR, "tech_company_ratios.csv"), "tech_company_ratios.csv")
        
        # Test count before
        count_before = vector_store.get_document_count(collection_target)
        
        # Delete old
        deleted = vector_store.delete_file("tech_company_ratios.csv", collection_target)
        print(f"  Deleted count: {deleted} (expected: {structured_count})")
        assert deleted == structured_count, f"Deleted count {deleted} does not match structured count {structured_count}!"
        
        # Insert again
        vector_store.add_documents(docs, collection_target)
        
        # Test count after
        count_after = vector_store.get_document_count(collection_target)
        print(f"  Count before: {count_before} | Count after: {count_after}")
        assert count_before == count_after, f"Collection count changed on duplicate insert! {count_before} != {count_after}"

        # ── Test 1: Query that matches Narrative ───────────────────────
        query_1 = "What was Services revenue in Q4 2024?"
        print(f"\nRunning Query 1 (Narrative match): '{query_1}'")
        res_1 = retrieve(query_1, vector_store)
        print(f"  Is relevant: {res_1.is_relevant}")
        print(f"  Is greeting: {res_1.is_greeting}")
        print(f"  Sources found: {len(res_1.sources)}")
        for src in res_1.sources:
            print(f"    - {src['source_filename']} (score: {src['similarity_score']}) | Origin: {src['origin']}")

        assert res_1.is_relevant, "Query 1 should return relevant results!"
        assert any(src['origin'] == 'NARRATIVE' for src in res_1.sources), "Query 1 should find narrative sources!"

        # ── Test 2: Query that matches Structured ──────────────────────
        query_2 = "What was Tesla's PE Ratio in 2024?"
        print(f"\nRunning Query 2 (Structured match): '{query_2}'")
        res_2 = retrieve(query_2, vector_store)
        print(f"  Is relevant: {res_2.is_relevant}")
        print(f"  Sources found: {len(res_2.sources)}")
        for src in res_2.sources:
            print(f"    - {src['source_filename']} (score: {src['similarity_score']}) | Origin: {src['origin']}")

        assert res_2.is_relevant, "Query 2 should return relevant results!"
        assert any(src['origin'] == 'STRUCTURED' for src in res_2.sources), "Query 2 should find structured sources!"

        # ── Test 3: Greeting query ─────────────────────────────────────
        query_3 = "hello"
        print(f"\nRunning Query 3 (Greeting): '{query_3}'")
        res_3 = retrieve(query_3, vector_store)
        print(f"  Is relevant: {res_3.is_relevant}")
        print(f"  Is greeting: {res_3.is_greeting}")
        assert res_3.is_greeting, "Query 3 should be classified as a greeting!"

        # ── Test 4: Irrelevant query (threshold check) ─────────────────
        query_4 = "How to grow banana plants indoors?"
        print(f"\nRunning Query 4 (Irrelevant): '{query_4}'")
        res_4 = retrieve(query_4, vector_store)
        print(f"  Is relevant: {res_4.is_relevant}")
        assert not res_4.is_relevant, "Query 4 should be marked irrelevant (below threshold)!"

        print("\n✅ All retrieval and scoring database tests passed successfully!")

    finally:
        # Restore configuration persist dir
        config.CHROMA_PERSIST_DIR = original_persist_dir
        # Clean up test DB
        if os.path.exists(TEST_CHROMA_DIR):
            shutil.rmtree(TEST_CHROMA_DIR)


if __name__ == "__main__":
    run_tests()
