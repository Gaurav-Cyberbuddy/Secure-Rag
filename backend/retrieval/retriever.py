from typing import List, Tuple, Optional

from langchain_core.documents import Document

from backend.retrieval.vector_store import load_vector_store


DEFAULT_TOP_K = 5

# Number of chunks before and after the matched chunk
# that should be combined into the same evidence block.
NEIGHBOR_CHUNKS = 1


def retrieve_documents(
    query: str,
    persist_directory: str,
    top_k: int = DEFAULT_TOP_K,
    filename: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    """
    Retrieve relevant document chunks and combine neighboring
    chunks from the same document.

    Uses the original query plus a lightweight future-scope
    query expansion when the question contains future/expansion
    terminology.

    The expanded retrieval only improves candidate recall.
    Existing reranking and security verification remain unchanged.

    If filename is provided, only that document is searched.
    """

    vector_store = load_vector_store(
        persist_directory=persist_directory
    )

    # =====================================================
    # 1. ORIGINAL SEMANTIC RETRIEVAL
    # =====================================================

    if filename:

        results = vector_store.similarity_search_with_score(
            query,
            k=top_k,
            filter={
                "filename": filename
            },
        )

    else:

        results = vector_store.similarity_search_with_score(
            query,
            k=top_k,
        )

    # =====================================================
    # 2. FUTURE-SCOPE QUERY EXPANSION
    # =====================================================

    expanded_query = query

    future_terms = (
        "future",
        "expand",
        "expansion",
        "features",
    )

    if any(
        term in query.lower()
        for term in future_terms
    ):
        expanded_query = (
            f"{query} "
            "future scope "
            "future improvements "
            "future recommendations "
            "future requirements "
            "framework expansion"
        )

    elif any(
        term in query.lower()
        for term in ("schedule", "timeline", "project plan")
    ):
        expanded_query = (
            f"{query} "
            "project schedule "
            "project plan "
            "timeline "
            "Phase 1 "
            "Phase 2 "
            "Phase 3 "
            "Requirements gathering "
            "Design and prototyping "
            "Development "
            "Testing release"
        )

    # =====================================================
    # 3. EXPANDED SEMANTIC RETRIEVAL
    # =====================================================

    if expanded_query != query:

        if filename:

            expanded_results = (
                vector_store.similarity_search_with_score(
                    expanded_query,
                    k=top_k,
                    filter={
                        "filename": filename
                    },
                )
            )

        else:

            expanded_results = (
                vector_store.similarity_search_with_score(
                    expanded_query,
                    k=top_k,
                )
            )

    else:

        expanded_results = []

    # =====================================================
    # 4. MERGE ORIGINAL + EXPANDED RESULTS
    # =====================================================

    merged_results = []

    seen = set()

    for document, score in (
        results + expanded_results
    ):

        source = document.metadata.get(
            "filename"
        )

        chunk_id = document.metadata.get(
            "chunk_id"
        )

        key = (
            source,
            str(chunk_id),
        )

        if key in seen:
            continue

        seen.add(key)

        merged_results.append(
            (
                document,
                score,
            )
        )

    results = merged_results

    if not results:
        return []

    # =====================================================
    # 5. FIND DOCUMENTS CONTAINING MATCHING CHUNKS
    # =====================================================

    document_names = set()

    for document, _ in results:

        source = document.metadata.get(
            "filename"
        )

        if source:
            document_names.add(source)

    # =====================================================
    # 6. LOAD ALL CHUNKS FROM RELEVANT DOCUMENTS
    # =====================================================

    chunk_lookup = {}

    for source in document_names:

        source_data = vector_store.get(
            where={
                "filename": source
            }
        )

        documents = source_data.get(
            "documents",
            []
        )

        metadatas = source_data.get(
            "metadatas",
            []
        )

        for content, metadata in zip(
            documents,
            metadatas,
        ):

            if not content:
                continue

            metadata = metadata or {}

            chunk_id = metadata.get(
                "chunk_id"
            )

            if chunk_id is None:
                continue

            try:

                chunk_id = int(
                    chunk_id
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            document = Document(
                page_content=content,
                metadata=metadata,
            )

            chunk_lookup[
                (
                    source,
                    chunk_id,
                )
            ] = document

    # =====================================================
    # 7. COMBINE NEIGHBORING CHUNKS
    # =====================================================

    expanded_results = []

    added_groups = set()

    for document, score in results:

        source = document.metadata.get(
            "filename"
        )

        chunk_id = document.metadata.get(
            "chunk_id"
        )

        if (
            source is None
            or chunk_id is None
        ):

            expanded_results.append(
                (
                    document,
                    score,
                )
            )

            continue

        try:

            chunk_id = int(
                chunk_id
            )

        except (
            TypeError,
            ValueError,
        ):

            expanded_results.append(
                (
                    document,
                    score,
                )
            )

            continue

        # -------------------------------------------------
        # Determine neighboring range
        # -------------------------------------------------

        start_id = max(
            0,
            chunk_id - NEIGHBOR_CHUNKS
        )

        end_id = (
            chunk_id
            + NEIGHBOR_CHUNKS
        )

        group_key = (
            source,
            start_id,
            end_id,
        )

        if group_key in added_groups:
            continue

        added_groups.add(
            group_key
        )

        # -------------------------------------------------
        # Collect neighboring chunks
        # -------------------------------------------------

        combined_documents = []

        combined_ids = []

        for current_id in range(
            start_id,
            end_id + 1,
        ):

            neighbor = chunk_lookup.get(
                (
                    source,
                    current_id,
                )
            )

            if neighbor is None:
                continue

            combined_documents.append(
                neighbor.page_content
            )

            combined_ids.append(
                current_id
            )

        if not combined_documents:
            continue

        # -------------------------------------------------
        # Combine content
        # -------------------------------------------------

        combined_content = (
            "\n\n".join(
                combined_documents
            )
        )

        # -------------------------------------------------
        # Preserve useful metadata
        # -------------------------------------------------

        combined_metadata = dict(
            document.metadata
        )

        combined_metadata[
            "chunk_id"
        ] = chunk_id

        combined_metadata[
            "neighbor_chunk_ids"
        ] = ",".join(
            str(value)
            for value in combined_ids
        )

        combined_metadata[
            "filename"
        ] = source

        combined_metadata[
            "expanded_context"
        ] = True

        combined_document = Document(
            page_content=combined_content,
            metadata=combined_metadata,
        )

        expanded_results.append(
            (
                combined_document,
                score,
            )
        )

    return expanded_results


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    from pathlib import Path

    PROJECT_DIR = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    VECTOR_DB = (
        PROJECT_DIR
        / "vector_db"
    )

    # -----------------------------------------------------
    # Ask question
    # -----------------------------------------------------

    query = input(
        "\nAsk a question: "
    ).strip()

    if not query:

        print(
            "\nPlease enter a question."
        )

        raise SystemExit

    # -----------------------------------------------------
    # Optional document selection
    # -----------------------------------------------------

    filename = input(
        "\nEnter document name "
        "(leave empty to search all documents): "
    ).strip()

    if not filename:

        filename = None

    # -----------------------------------------------------
    # Retrieve
    # -----------------------------------------------------

    results = retrieve_documents(
        query=query,
        persist_directory=str(
            VECTOR_DB
        ),
        top_k=5,
        filename=filename,
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print(
        "\n========== RETRIEVAL RESULTS =========="
    )

    print(
        f"Query: {query}"
    )

    if filename:

        print(
            f"Document filter: {filename}"
        )

    else:

        print(
            "Document filter: ALL DOCUMENTS"
        )

    print(
        f"Retrieved evidence groups: "
        f"{len(results)}"
    )

    # -----------------------------------------------------
    # Show results
    # -----------------------------------------------------

    for rank, (document, score) in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n--- Result {rank} ---"
        )

        print(
            f"Score: {score}"
        )

        print(
            f"Main Chunk ID: "
            f"{document.metadata.get('chunk_id')}"
        )

        print(
            f"Neighbor Chunks: "
            f"{document.metadata.get('neighbor_chunk_ids')}"
        )

        print(
            f"Source: "
            f"{document.metadata.get('filename')}"
        )

        print(
            "\nContent:"
        )

        print(
            document.page_content[:2000]
        )

    print(
        "\n=======================================\n"
    )