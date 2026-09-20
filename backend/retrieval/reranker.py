from typing import List, Tuple
import re
import math

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Number of candidates expected from retrieval
DEFAULT_TOP_K = 5


# =========================================================
# MODEL
# =========================================================

_reranker = None


def create_reranker() -> CrossEncoder:
    """
    Load the cross-encoder reranking model once
    and reuse it for subsequent requests.
    """

    global _reranker

    if _reranker is None:
        print(
            "\n[MODEL] Loading reranker model..."
        )

        _reranker = CrossEncoder(
            RERANKER_MODEL
        )

        print(
            "[MODEL] Reranker model loaded."
        )

    return _reranker

# =========================================================
# TEXT NORMALIZATION
# =========================================================

def tokenize(text: str) -> set[str]:
    """
    Convert text into normalized word tokens.

    Very common stop words are removed so that words such as
    'the', 'was', 'used', etc. do not dominate relevance.
    """

    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "from",
        "by",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
        "used",
        "use",
        "using",
        "used",
        "do",
        "did",
        "does",
        "were",
        "their",
        "they",
        "it",
    }

    words = re.findall(
        r"\b[a-zA-Z0-9][a-zA-Z0-9.+#-]*\b",
        text.lower(),
    )

    return {
        word
        for word in words
        if word not in stop_words
        and len(word) > 1
    }


# =========================================================
# LEXICAL RELEVANCE
# =========================================================

def lexical_relevance(
    query: str,
    document: str,
) -> float:
    """
    Calculate direct vocabulary overlap between the query
    and document.

    This is NOT used instead of semantic reranking.

    It is an additional signal that helps ensure that a
    document containing important words from the actual
    question receives an appropriate ranking.
    """

    query_words = tokenize(query)
    document_words = tokenize(document)

    if not query_words or not document_words:
        return 0.0

    overlap = query_words.intersection(
        document_words
    )

    return len(overlap) / len(query_words)


# =========================================================
# NORMALIZE CROSS-ENCODER SCORE
# =========================================================

def normalize_cross_encoder_score(
    score: float,
) -> float:
    """
    Convert the MS-MARCO cross-encoder score into a
    stable 0-1 range.

    MS-MARCO scores are logits rather than probabilities.
    """

    try:
        return 1.0 / (
            1.0 + math.exp(-float(score))
        )
    except OverflowError:

        return (
            1.0
            if score > 0
            else 0.0
        )


# =========================================================
# RERANK
# =========================================================

def rerank_documents(
    query: str,
    results: List[Tuple[Document, float]],
    top_k: int = DEFAULT_TOP_K,
) -> List[Tuple[Document, float]]:
    """
    Rerank retrieved document chunks.

    Ranking combines:

        1. Cross-encoder semantic relevance
        2. Direct query/document vocabulary relevance

    The semantic model remains the primary signal.

    The lexical signal is used only to improve ranking when
    the query contains terms that are explicitly present in
    a relevant document chunk.

    No question category or document-specific rule is used.
    """

    if not results:
        return []

    reranker = create_reranker()

    # -----------------------------------------------------
    # Prepare query/document pairs
    # -----------------------------------------------------

    pairs = [
        (
            query,
            document.page_content,
        )
        for document, _
        in results
    ]

    # -----------------------------------------------------
    # Cross-encoder scores
    # -----------------------------------------------------

    semantic_scores = reranker.predict(
    pairs,
    batch_size=16,
    show_progress_bar=False,
)

    reranked = []

    # -----------------------------------------------------
    # Combine relevance signals
    # -----------------------------------------------------

    for (
        (document, original_score),
        semantic_score,
    ) in zip(
        results,
        semantic_scores,
    ):

        semantic_score = float(
            semantic_score
        )

        semantic_probability = (
            normalize_cross_encoder_score(
                semantic_score
            )
        )

        lexical_score = lexical_relevance(
            query=query,
            document=document.page_content,
        )

        # -------------------------------------------------
        # FINAL RELEVANCE SCORE
        # -------------------------------------------------
        #
        # Semantic relevance remains dominant.
        #
        # 80% semantic
        # 20% lexical
        #
        # This prevents a chunk from winning merely because
        # it contains a few matching words.
        # -------------------------------------------------

        final_score = (
            0.80 * semantic_probability
            + 0.20 * lexical_score
        )

        reranked.append(
            (
                document,
                final_score,
            )
        )

    # -----------------------------------------------------
    # Sort highest relevance first
    # -----------------------------------------------------

    reranked.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return reranked[:top_k]


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    from pathlib import Path

    from backend.retrieval.retriever import (
        retrieve_documents,
    )

    PROJECT_DIR = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    VECTOR_DB = (
        PROJECT_DIR
        / "vector_db"
    )

    query = input(
        "\nAsk a question: "
    ).strip()

    print(
        "\nRetrieving candidate chunks..."
    )

    candidates = retrieve_documents(
        query=query,
        persist_directory=str(
            VECTOR_DB
        ),
        top_k=20,
    )

    print(
        f"Initial candidates: "
        f"{len(candidates)}"
    )

    if not candidates:

        print(
            "\nNo candidates found."
        )

        raise SystemExit

    print(
        "\nReranking candidates..."
    )

    results = rerank_documents(
        query=query,
        results=candidates,
        top_k=5,
    )

    print(
        "\n========== RERANKED RESULTS =========="
    )

    for rank, (
        document,
        score,
    ) in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n--- Result {rank} ---"
        )

        print(
            f"Final relevance score: "
            f"{score:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{document.metadata.get('chunk_id')}"
        )

        print(
            f"Source: "
            f"{document.metadata.get('filename')}"
        )

        lexical_score = lexical_relevance(
            query=query,
            document=document.page_content,
        )

        print(
            f"Lexical relevance: "
            f"{lexical_score:.4f}"
        )

        print(
            "\nContent:"
        )

        print(
            document.page_content[:1000]
        )

    print(
        "\n=======================================\n"
    )