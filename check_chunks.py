from pathlib import Path

from backend.rag.ingestion import ingest_document
from backend.rag.chunking import split_documents


PROJECT_DIR = Path(__file__).resolve().parent

DOCUMENT = (
    PROJECT_DIR
    / "data"
    / "documents"
    / "Internship report.pdf.pdf"
)

documents = ingest_document(
    file_path=str(DOCUMENT),
    source="official_aws_docs",
    trusted_sources={"official_aws_docs"},
)

chunks = split_documents(documents)

keywords = [
    "future",
    "recommendation",
    "recommendations",
    "SIEM",
    "RBAC",
    "cloud",
    "expansion",
    "improvement",
]

print(f"\nTotal chunks: {len(chunks)}")

for chunk in chunks:
    text = chunk.page_content.lower()

    matched = [
        keyword
        for keyword in keywords
        if keyword.lower() in text
    ]

    if matched:
        print("\n" + "=" * 80)
        print(f"CHUNK ID: {chunk.metadata.get('chunk_id')}")
        print(f"MATCHED: {matched}")
        print("=" * 80)
        print(chunk.page_content)