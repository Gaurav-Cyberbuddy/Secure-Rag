from typing import List

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =========================================================
# MODEL CACHE
# =========================================================

_embedding_model = None


def create_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create the embedding model once and reuse it.
    """

    global _embedding_model

    if _embedding_model is None:

        print(
            "\n[MODEL] Loading embedding model..."
        )

        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        print(
            "[MODEL] Embedding model loaded."
        )

    return _embedding_model


# =========================================================
# EMBED DOCUMENTS
# =========================================================

def embed_documents(
    documents: List[Document],
) -> HuggingFaceEmbeddings:
    """
    Return the cached embedding model.

    The vector store uses this model to generate
    embeddings for document chunks and queries.
    """

    return create_embedding_model()


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print(
        "Loading embedding model..."
    )

    embedding_model = create_embedding_model()

    test_text = (
        "Amazon S3 provides object storage."
    )

    vector = embedding_model.embed_query(
        test_text
    )

    print(
        "\n========== EMBEDDING SUCCESS =========="
    )

    print(
        f"Model: {EMBEDDING_MODEL}"
    )

    print(
        f"Vector dimensions: {len(vector)}"
    )

    print(
        f"Vector preview: {vector[:5]}"
    )

    print(
        "=======================================\n"
    )