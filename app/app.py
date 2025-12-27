import argparse
from pdf_loader import load_pdf
from splitter import split_docs
from embeddings_store import build_vector_store
from stage1_coverage import run_stage1
from stage2_pseudocode import run_stage2


def main():
    parser = argparse.ArgumentParser(description="Control Narrative → Pseudocode RAG System")

    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the PDF file"
    )

    parser.add_argument(
        "--task",
        required=True,
        help="Task instruction for pseudocode generator (ex: 'Generate system pseudocode')"
    )

    parser.add_argument(
        "--query",
        default=(
            "extract complete control / system logic requirements including startup, "
            "operating control logic, alarms, interlocks, shutdown, abort, transitions, "
            "and subsystem behavior"
        ),
        help="(Optional) Custom retrieval query for Stage-1 logic extraction"
    )

    args = parser.parse_args()

    pdf_path = args.pdf
    task_prompt = args.task
    retriever_query = args.query

    print(f"Loading PDF: {pdf_path}")
    docs = load_pdf(pdf_path)

    print("Splitting text...")
    chunks = split_docs(docs)

    print("Building vector store...")
    retriever = build_vector_store(chunks)

    # ---------- Stage 1 (Internal Only — No Print) ----------
    coverage_logic = run_stage1(retriever, retriever_query)

    # ---------- Stage 2 (Final Output Only) ----------
    print("\nRunning Stage 2: Pseudocode Generation...\n")
    pseudocode = run_stage2(coverage_logic, task_prompt)

    print("\n===== FINAL PSEUDOCODE OUTPUT =====\n")
    print(pseudocode)


if __name__ == "__main__":
    main()
