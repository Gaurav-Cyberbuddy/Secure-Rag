from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma

from backend.rag.embeddings import create_embedding_model


COLLECTION_NAME = "secure_askrag_documents"


# =========================================================
# CREATE VECTOR STORE
# =========================================================

def create_vector_store(
    documents: List[Document],
    persist_directory: str,
) -> Chroma:
    """
    Create a new Chroma vector store from document chunks.
    """

    if not documents:
        raise ValueError("No documents were provided.")

    embedding_model = create_embedding_model()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
    )

    return vector_store


# =========================================================
# LOAD VECTOR STORE
# =========================================================

def load_vector_store(
    persist_directory: str,
) -> Chroma:
    """
    Load an existing Chroma vector store.
    """

    embedding_model = create_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )


# =========================================================
# GET AVAILABLE DOCUMENTS
# =========================================================

def get_available_documents(
    persist_directory: str,
) -> List[str]:
    """
    Get the unique document filenames currently stored
    in the vector database.
    """

    vector_store = load_vector_store(
        persist_directory=persist_directory
    )

    data = vector_store.get(
        include=["metadatas"]
    )

    filenames = set()

    for metadata in data.get(
        "metadatas",
        []
    ):

        if metadata:

            filename = metadata.get(
                "filename"
            )

            if filename:
                filenames.add(filename)

    return sorted(filenames)


# =========================================================
# CHECK DOCUMENT DUPLICATE
# =========================================================

def document_already_exists(
    vector_store: Chroma,
    sha256: str,
) -> bool:
    """
    Check whether a document with the same SHA-256
    hash already exists in the vector store.
    """

    if not sha256:
        return False

    result = vector_store.get(
        where={
            "sha256": sha256
        },
        limit=1,
    )

    ids = result.get("ids", [])

    return len(ids) > 0


# =========================================================
# ADD DOCUMENTS TO EXISTING VECTOR STORE
# =========================================================

def add_documents_to_vector_store(
    documents: List[Document],
    persist_directory: str,
) -> tuple[Chroma, bool]:
    """
    Add a new document to an existing Chroma vector store.

    Returns:

        vector_store
        document_added

    document_added is False when the same document
    already exists based on its SHA-256 hash.
    """

    if not documents:
        raise ValueError(
            "No documents were provided."
        )

    vector_store = load_vector_store(
        persist_directory=persist_directory
    )

    # -----------------------------------------------------
    # Get SHA-256 from document metadata
    # -----------------------------------------------------

    sha256 = documents[0].metadata.get(
        "sha256"
    )

    if not sha256:
        raise ValueError(
            "Document chunks do not contain "
            "SHA-256 metadata."
        )

    # -----------------------------------------------------
    # Duplicate check
    # -----------------------------------------------------

    if document_already_exists(
        vector_store=vector_store,
        sha256=sha256,
    ):

        return vector_store, False

    # -----------------------------------------------------
    # Add new document chunks
    # -----------------------------------------------------

    vector_store.add_documents(
        documents
    )

    return vector_store, True


# =========================================================
# VECTOR STORE TEST
# =========================================================

if __name__ == "__main__":

    from backend.rag.ingestion import (
        ingest_document
    )

    from backend.rag.chunking import (
        split_documents
    )

    # -----------------------------------------------------
    # Project paths
    # -----------------------------------------------------

    PROJECT_DIR = Path(
        __file__
    ).resolve().parents[2]

    DOCUMENT = (
        PROJECT_DIR
        / "data"
        / "documents"
        / "Internship report.pdf.pdf"
    )

    VECTOR_DB = (
        PROJECT_DIR
        / "vector_db"
    )

    TRUSTED_SOURCES = {
        "official_aws_docs",
        "research_paper",
        "internal_verified_docs",
    }

    # -----------------------------------------------------
    # Load document
    # -----------------------------------------------------

    print(
        "\n========== DOCUMENT INGESTION =========="
    )

    documents = ingest_document(
        file_path=str(DOCUMENT),
        source="official_aws_docs",
        trusted_sources=TRUSTED_SOURCES,
    )

    print(
        f"Pages loaded: {len(documents)}"
    )

    # -----------------------------------------------------
    # Chunk document
    # -----------------------------------------------------

    print(
        "\n========== CHUNKING =========="
    )

    chunks = split_documents(
        documents
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    # -----------------------------------------------------
    # Create or update vector store
    # -----------------------------------------------------

    print(
        "\n========== VECTOR STORE =========="
    )

    if VECTOR_DB.exists():

        print(
            "Existing vector database found."
        )

        vector_store, added = (
            add_documents_to_vector_store(
                documents=chunks,
                persist_directory=str(
                    VECTOR_DB
                ),
            )
        )

        if added:

            print(
                "New document added successfully."
            )

        else:

            print(
                "Document already exists."
            )

            print(
                "Duplicate document was NOT added."
            )

    else:

        print(
            "No vector database found."
        )

        vector_store = create_vector_store(
            documents=chunks,
            persist_directory=str(
                VECTOR_DB
            ),
        )

        print(
            "New vector database created."
        )

    # -----------------------------------------------------
    # Show available documents
    # -----------------------------------------------------

    print(
        "\n========== AVAILABLE DOCUMENTS =========="
    )

    available_documents = get_available_documents(
        persist_directory=str(
            VECTOR_DB
        )
    )

    if available_documents:

        for index, filename in enumerate(
            available_documents,
            start=1,
        ):

            print(
                f"{index}. {filename}"
            )

    else:

        print(
            "No documents found."
        )

    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    print(
        "\n========== VECTOR STORE SUCCESS =========="
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Processed chunks: {len(chunks)}"
    )

    print(
        f"Available documents: "
        f"{len(available_documents)}"
    )

    print(
        f"Database: {VECTOR_DB}"
    )

    print(
        "==========================================\n"
    )