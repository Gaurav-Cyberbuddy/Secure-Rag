from pathlib import Path

from backend.rag.ingestion import ingest_document
from backend.rag.chunking import split_documents
from backend.retrieval.vector_store import (
    add_documents_to_vector_store,
    create_vector_store,
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

VECTOR_DB = PROJECT_DIR / "vector_db"


# =========================================================
# TRUSTED SOURCES
# =========================================================

TRUSTED_SOURCES = {
    "official_aws_docs",
    "research_paper",
    "internal_verified_docs",
}


# =========================================================
# ADD DOCUMENT
# =========================================================

def add_document(
    file_path: str,
    source: str = "internal_verified_docs",
):
    """
    Ingest, chunk, and add a document to the
    existing Secure AskRAG vector database.
    """

    path = Path(file_path)

    print("\n========== DOCUMENT ADDITION ==========")
    print(f"Document: {path.name}")

    # -----------------------------------------------------
    # 1. Ingest document
    # -----------------------------------------------------

    print("\n[1/3] Ingesting document...")

    documents = ingest_document(
        file_path=str(path),
        source=source,
        trusted_sources=TRUSTED_SOURCES,
    )

    print(
        f"Loaded pages/documents: {len(documents)}"
    )

    # -----------------------------------------------------
    # 2. Create chunks
    # -----------------------------------------------------

    print("\n[2/3] Creating chunks...")

    chunks = split_documents(
        documents
    )

    print(
        f"Created chunks: {len(chunks)}"
    )

    # -----------------------------------------------------
    # 3. Add to vector database
    # -----------------------------------------------------

    print("\n[3/3] Updating vector database...")

    if VECTOR_DB.exists():

        _, added = add_documents_to_vector_store(
            documents=chunks,
            persist_directory=str(VECTOR_DB),
        )

        if not added:

            print(
                "\nDocument already exists."
            )

            print(
                "No duplicate chunks were added."
            )

            return False

        print(
            "\nNew document added successfully."
        )

    else:

        create_vector_store(
            documents=chunks,
            persist_directory=str(VECTOR_DB),
        )

        print(
            "\nVector database created."
        )

    print("\n=======================================")

    return True


# =========================================================
# COMMAND-LINE TEST
# =========================================================

if __name__ == "__main__":

    print("\n======================================")
    print("       SECURE ASKRAG DOCUMENT ADDER")
    print("======================================")

    file_path = input(
        "\nEnter document path: "
    ).strip()

    if not file_path:

        print(
            "\nNo document path provided."
        )

        raise SystemExit

    try:

        add_document(
            file_path=file_path,
            source="internal_verified_docs",
        )

    except Exception as error:

        print(
            "\n========== DOCUMENT ADDITION FAILED =========="
        )

        print(error)

        print(
            "==============================================\n"
        )