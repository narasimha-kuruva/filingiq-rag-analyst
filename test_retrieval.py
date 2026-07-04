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
from utils.file_hash import compute_file_hash
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
        file_hash = compute_file_hash(filepath)
        docs, collection_target = ingest_file(filepath, filename)
        vector_store.sync_document(
            documents=docs,
            file_hash=file_hash,
            source_filename=filename,
            collection_target=collection_target
        )

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
        filepath = os.path.join(DATA_DIR, "tech_company_ratios.csv")
        
        file_hash = compute_file_hash(filepath)
        docs, collection_target = ingest_file(filepath, "tech_company_ratios.csv")
        
        # Call high-level index_file logic for the exact same file
        stats = vector_store.sync_document(
            documents=docs,
            file_hash=file_hash,
            source_filename="tech_company_ratios.csv",
            collection_target=collection_target
        )
        print(f"  Action: {stats['action']}")
        print(f"  Already Exists: {stats['already_indexed']}")
        print(f"  File Changed: {stats['file_changed']}")
        print(f"  Deleted count: {stats['deleted_count']} (expected: 0)")
        print(f"  Inserted count: {stats['inserted_count']} (expected: 0)")
        
        assert stats['action'] == "Skipped", "Action should be Skipped for unchanged file!"
        assert stats['already_indexed'] is True, "Should detect that the file already exists!"
        assert stats['file_changed'] is False, "Should report that the file has not changed!"
        assert stats['deleted_count'] == 0, "No vectors should be deleted!"
        assert stats['inserted_count'] == 0, "No vectors should be inserted!"
        assert stats['count_before'] == stats['count_after'], f"Collection count changed on duplicate insert! {stats['count_before']} != {stats['count_after']}"

        # ── Test Versioning/Replacement (Modifying file) ──────────────────
        print("\nTesting Versioning (uploading a modified version of the file)...")
        modified_filepath = os.path.join(DATA_DIR, "tech_company_ratios_mod.csv")
        with open(filepath, "r") as f:
            lines = f.readlines()
        lines[-1] = lines[-1].replace("112.8", "120.0") # Change PE Ratio of Tesla in 2021
        with open(modified_filepath, "w") as f:
            f.writelines(lines)
            
        try:
            file_hash_mod = compute_file_hash(modified_filepath)
            docs_mod, collection_target_mod = ingest_file(modified_filepath, "tech_company_ratios.csv")
            
            stats_mod = vector_store.sync_document(
                documents=docs_mod,
                file_hash=file_hash_mod,
                source_filename="tech_company_ratios.csv",
                collection_target=collection_target_mod
            )
            print(f"  Action: {stats_mod['action']}")
            print(f"  Already Exists: {stats_mod['already_indexed']}")
            print(f"  File Changed: {stats_mod['file_changed']}")
            print(f"  Deleted count: {stats_mod['deleted_count']} (expected: {structured_count})")
            print(f"  Inserted count: {stats_mod['inserted_count']} (expected: {structured_count})")
            
            assert stats_mod['action'] == "Replaced", "Action should be Replaced for modified file!"
            assert stats_mod['already_indexed'] is True, "Should detect that the file already exists!"
            assert stats_mod['file_changed'] is True, "Should report that the file has changed!"
            assert stats_mod['deleted_count'] == structured_count, "Old vectors should be deleted!"
            assert stats_mod['inserted_count'] == structured_count, "New vectors should be inserted!"
            assert stats_mod['successful'] is True, "Replacement should report success!"
        finally:
            if os.path.exists(modified_filepath):
                os.remove(modified_filepath)

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

        # ── Test Validation Constraints ────────────────────────────────
        print("\nTesting Validation Constraints...")
        from langchain_core.documents import Document
        
        # Test Case A: Document without required metadata (missing file_type)
        invalid_doc_1 = Document(
            page_content="Test text",
            metadata={"source_filename": "test.txt", "chunk_index": 0}
        )
        res_val_1 = vector_store.replace_file([invalid_doc_1], "narrative")
        print(f"  Missing file_type validation test: success={res_val_1['successful']}")
        assert res_val_1['successful'] is False, "Should abort replacement if file_type metadata is missing!"
        
        # Test Case B: Multiple filenames in batch
        invalid_doc_2a = Document(
            page_content="Text A",
            metadata={"source_filename": "file_a.txt", "file_type": "txt", "chunk_index": 0}
        )
        invalid_doc_2b = Document(
            page_content="Text B",
            metadata={"source_filename": "file_b.txt", "file_type": "txt", "chunk_index": 0}
        )
        res_val_2 = vector_store.replace_file([invalid_doc_2a, invalid_doc_2b], "narrative")
        print(f"  Multiple filenames validation test: success={res_val_2['successful']}")
        assert res_val_2['successful'] is False, "Should abort replacement if multiple source filenames are in the batch!"

        print("\n✅ All retrieval and scoring database tests passed successfully!")

    finally:
        # Restore configuration persist dir
        config.CHROMA_PERSIST_DIR = original_persist_dir
        # Clean up test DB
        if os.path.exists(TEST_CHROMA_DIR):
            shutil.rmtree(TEST_CHROMA_DIR)


if __name__ == "__main__":
    run_tests()
