from pathlib import Path

from backend.retrieval.retriever import retrieve_documents
from backend.retrieval.reranker import rerank_documents
from backend.retrieval.vector_store import get_available_documents
from backend.rag.generator import generate_answer


TOP_K_RETRIEVAL = 20
TOP_K_RERANKED = 5


def build_context(results) -> str:
    """
    Combine reranked document chunks into the context
    provided to the LLM.
    """

    return "\n\n--- EVIDENCE ---\n\n".join(
        document.page_content
        for document, _ in results
    )


def get_sources(results) -> list[str]:
    """
    Get unique source documents used for the answer.
    """

    sources = []

    for document, _ in results:

        filename = document.metadata.get(
            "filename"
        )

        if filename and filename not in sources:
            sources.append(filename)

    return sources


def ask_rag(
    query: str,
    vector_db: str,
    filename: str | None = None,
):
    """
    Complete RAG pipeline.

    If filename is provided, only that document
    will be searched.

    If filename is None, all documents are searched.
    """

    # -----------------------------------------------------
    # 1. Retrieve candidate chunks
    # -----------------------------------------------------

    candidates = retrieve_documents(
        query=query,
        persist_directory=vector_db,
        top_k=TOP_K_RETRIEVAL,
        filename=filename,
    )

    if not candidates:

        return (
            "I could not find the answer "
            "in the selected document.",
            [],
        )

    # -----------------------------------------------------
    # 2. Rerank candidates
    # -----------------------------------------------------

    reranked = rerank_documents(
        query=query,
        results=candidates,
        top_k=TOP_K_RERANKED,
    )

    if not reranked:

        return (
            "I could not find the answer "
            "in the selected document.",
            [],
        )

    # -----------------------------------------------------
    # 3. Build context
    # -----------------------------------------------------

    context = build_context(
        reranked
    )

    # -----------------------------------------------------
    # 4. Generate answer
    # -----------------------------------------------------

    answer = generate_answer(
        query=query,
        context=context,
    )

    # -----------------------------------------------------
    # 5. Identify sources
    # -----------------------------------------------------

    sources = get_sources(
        reranked
    )

    return answer, sources


# =========================================================
# COMMAND-LINE INTERFACE
# =========================================================

if __name__ == "__main__":

    PROJECT_DIR = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    VECTOR_DB = (
        PROJECT_DIR
        / "vector_db"
    )

    print(
        "\n======================================"
    )

    print(
        "          SECURE ASKRAG"
    )

    print(
        "======================================"
    )

    # -----------------------------------------------------
    # Get documents from vector database
    # -----------------------------------------------------

    try:

        available_documents = (
            get_available_documents(
                persist_directory=str(
                    VECTOR_DB
                )
            )
        )

    except Exception as error:

        print(
            "\nCould not load available documents:"
        )

        print(error)

        raise SystemExit

    if not available_documents:

        print(
            "\nNo documents are currently available."
        )

        print(
            "Please ingest a document first."
        )

        raise SystemExit

    # -----------------------------------------------------
    # Document selection
    # -----------------------------------------------------

    while True:

        print(
            "\n========== AVAILABLE DOCUMENTS =========="
        )

        for index, filename in enumerate(
            available_documents,
            start=1,
        ):

            print(
                f"{index}. {filename}"
            )

        print(
            "0. Search all documents"
        )

        print(
            "Type 'exit' to stop."
        )

        print(
            "========================================="
        )

        selection = input(
            "\nSelect document: "
        ).strip()

        if selection.lower() == "exit":

            print(
                "\nExiting..."
            )

            break

        # -------------------------------------------------
        # Search all documents
        # -------------------------------------------------

        if selection == "0":

            selected_filename = None

        # -------------------------------------------------
        # Specific document
        # -------------------------------------------------

        else:

            try:

                selected_index = int(
                    selection
                )

            except ValueError:

                print(
                    "\nInvalid selection."
                )

                print(
                    "Please enter a document number."
                )

                continue

            if (
                selected_index < 1
                or selected_index
                > len(available_documents)
            ):

                print(
                    "\nInvalid document number."
                )

                continue

            selected_filename = (
                available_documents[
                    selected_index - 1
                ]
            )

        # -------------------------------------------------
        # Show selected document
        # -------------------------------------------------

        if selected_filename:

            print(
                f"\nSelected document:"
                f" {selected_filename}"
            )

        else:

            print(
                "\nSelected:"
                " ALL DOCUMENTS"
            )

        # -------------------------------------------------
        # Ask questions
        # -------------------------------------------------

        while True:

            query = input(
                "\nAsk Question "
                "(type 'back' to change document): "
            ).strip()

            if query.lower() == "back":

                break

            if query.lower() == "exit":

                print(
                    "\nExiting..."
                )

                raise SystemExit

            if not query:

                print(
                    "Please enter a question."
                )

                continue

            try:

                answer, sources = ask_rag(
                    query=query,
                    vector_db=str(
                        VECTOR_DB
                    ),
                    filename=selected_filename,
                )

                print(
                    "\n========== ANSWER =========="
                )

                print(answer)

                if sources:

                    print(
                        "\n========== SOURCE =========="
                    )

                    for source in sources:

                        print(
                            f"- {source}"
                        )

                print(
                    "\n============================"
                )

            except Exception as error:

                print(
                    "\nPipeline error:"
                )

                print(error)