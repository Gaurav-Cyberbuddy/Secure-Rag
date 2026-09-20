from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split loaded documents into smaller overlapping chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    # Give every chunk a unique identifier.
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


if __name__ == "__main__":

    from backend.rag.ingestion import ingest_document
    from pathlib import Path

    PROJECT_DIR = Path(__file__).resolve().parents[2]

    DOCUMENT = PROJECT_DIR / "data" / "documents" / "Internship report.pdf.pdf"

    TRUSTED_SOURCES = {
        "official_aws_docs",
        "research_paper",
        "internal_verified_docs",
    }

    documents = ingest_document(
        file_path=str(DOCUMENT),
        source="official_aws_docs",
        trusted_sources=TRUSTED_SOURCES,
    )

    print("\n========== DOCUMENT LOADED ==========")
    print(f"Pages loaded: {len(documents)}")

    chunks = split_documents(documents)

    print("\n========== CHUNKING SUCCESS ==========")
    print(f"Total chunks: {len(chunks)}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")

    if chunks:
        print("\n========== FIRST CHUNK ==========")
        print(f"Chunk ID: {chunks[0].metadata['chunk_id']}")
        print(f"Source: {chunks[0].metadata['filename']}")
        print("\nContent preview:")
        print(chunks[0].page_content[:500])
        print("================================")