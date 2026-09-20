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

PROJECT_DIR = Path(
    __file__
).resolve().parents[2]

VECTOR_DB = (
    PROJECT_DIR
    / "vector_db"
)


# =========================================================
# TRUSTED SOURCES
# =========================================================

TRUSTED_SOURCES = {
    "official_aws_docs",
    "research_paper",
    "internal_verified_docs",
}


# =========================================================
# UPLOAD / INGEST DOCUMENT
# =========================================================

def upload_document(
    file_path: str,
    source: str = "internal_verified_docs",
):
    """
    Validate, ingest, chunk, and store a document
    in the Secure AskRAG vector database.
    """

    path = Path(file_path)

    # -----------------------------------------------------
    # Check file exists
    # -----------------------------------------------------

    if not path.exists():

        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    print(
        "\n========== DOCUMENT UPLOAD =========="
    )

    print(
        f"File: {path.name}"
    )

    # -----------------------------------------------------
    # 1. INGEST
    # -----------------------------------------------------

    print(
        "\n[1/3] Validating and ingesting..."
    )

    documents = ingest_document(
        file_path=str(path),
        source=source,
        trusted_sources=TRUSTED_SOURCES,
    )

    print(
        f"Documents/pages loaded: "
        f"{len(documents)}"
    )

    # -----------------------------------------------------
    # 2. CHUNK
    # -----------------------------------------------------

    print(
        "\n[2/3] Creating chunks..."
    )

    chunks = split_documents(
        documents
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    # -----------------------------------------------------
    # 3. VECTOR DATABASE
    # -----------------------------------------------------

    print(
        "\n[3/3] Adding to vector database..."
    )

    if VECTOR_DB.exists():

        _, added = add_documents_to_vector_store(
            documents=chunks,
            persist_directory=str(
                VECTOR_DB
            ),
        )

        if not added:

            print(
                "\nDocument already exists."
            )

            print(
                "Duplicate document was not added."
            )

            return False

    else:

        create_vector_store(
            documents=chunks,
            persist_directory=str(
                VECTOR_DB
            ),
        )

    print(
        "\n========== UPLOAD SUCCESS =========="
    )

    print(
        f"Document: {path.name}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        f"Vector DB: {VECTOR_DB}"
    )

    print(
        "===================================="
    )

    return True


# =========================================================
# COMMAND-LINE TEST
# =========================================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "      SECURE ASKRAG DOCUMENT UPLOAD"
    )

    print(
        "======================================"
    )

    file_path = input(
        "\nEnter document path: "
    ).strip()

    if not file_path:

        print(
            "\nNo document selected."
        )

        raise SystemExit

    try:

        upload_document(
            file_path=file_path,
            source="internal_verified_docs",
        )

    except Exception as error:

        print(
            "\n========== UPLOAD FAILED =========="
        )

        print(error)

        print(
            "===================================\n"
        )