"""
Ingestion Test — verifies all 5 loaders produce correct Document objects.

Loads one sample file of each type from data/ and prints the first 3
Documents with metadata for visual inspection.

Run from project root:
    python test_ingestion.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from ingestion.router import ingest_file


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Map of test files — (filename, expected_collection_target)
TEST_FILES = [
    ("apple_10k_excerpt_2024.pdf", "narrative"),
    ("apple_analyst_memo_2024.docx", "narrative"),
    ("apple_q4_2024_earnings_notes.txt", "narrative"),
    ("apple_financials_2024.xlsx", "structured"),
    ("tech_company_ratios.csv", "structured"),
]


def test_file(filename: str, expected_target: str) -> bool:
    """Test a single file and print results."""
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        print(f"  ❌ File not found: {filepath}")
        print(f"     Run 'python generate_sample_data.py' first.\n")
        return False

    try:
        documents, collection_target = ingest_file(filepath, filename)

        print(f"  Collection target: {collection_target} (expected: {expected_target})")
        print(f"  Documents created: {len(documents)}")

        assert collection_target == expected_target, (
            f"Wrong collection target: {collection_target} != {expected_target}"
        )

        # Print first 3 documents
        for i, doc in enumerate(documents[:3]):
            print(f"\n  --- Document {i + 1} ---")
            print(f"  Metadata: {doc.metadata}")
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"  Content:  {preview}...")

        print(f"\n  ✅ PASSED\n")
        return True

    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("FilingIQ — Ingestion Layer Test")
    print("=" * 70)

    results = {}
    for filename, expected_target in TEST_FILES:
        ext = os.path.splitext(filename)[1].upper()
        print(f"\n{'─' * 50}")
        print(f"Testing {ext}: {filename}")
        print(f"{'─' * 50}")
        results[ext] = test_file(filename, expected_target)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = True
    for ext, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {ext:8s} {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed! Ingestion layer is ready.")
    else:
        print("Some tests failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
