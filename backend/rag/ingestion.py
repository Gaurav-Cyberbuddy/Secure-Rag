from pathlib import Path
import hashlib
from typing import Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)


# =========================================================
# SUPPORTED DOCUMENT TYPES
# =========================================================

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".csv",
}


# =========================================================
# SHA-256
# =========================================================

def calculate_sha256(file_path: Path) -> str:
    """
    Calculate the SHA-256 hash of a document.
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:

        while chunk := file.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()


# =========================================================
# FILE TYPE VALIDATION
# =========================================================

def validate_file_type(file_path: Path) -> None:
    """
    Verify that the document has an allowed file extension.
    """

    extension = file_path.suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Allowed types: {sorted(ALLOWED_EXTENSIONS)}"
        )


# =========================================================
# INTEGRITY VERIFICATION
# =========================================================

def verify_integrity(
    file_path: Path,
    expected_sha256: Optional[str] = None,
) -> str:
    """
    Calculate the document SHA-256 hash.

    If an expected hash is provided,
    compare it with the calculated hash.
    """

    actual_sha256 = calculate_sha256(file_path)

    if expected_sha256 is not None:

        if actual_sha256.lower() != expected_sha256.lower():

            raise ValueError(
                "Document integrity verification failed.\n"
                f"Expected: {expected_sha256}\n"
                f"Actual:   {actual_sha256}"
            )

    return actual_sha256


# =========================================================
# SOURCE VALIDATION
# =========================================================

def validate_source(
    source: str,
    trusted_sources: set[str],
) -> None:
    """
    Verify that the document comes from an approved source.
    """

    if source not in trusted_sources:

        raise ValueError(
            f"Untrusted document source: {source}"
        )


# =========================================================
# DOCUMENT LOADING
# =========================================================

def load_document(file_path: Path):
    """
    Load a supported document into LangChain
    Document objects.
    """

    extension = file_path.suffix.lower()

    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------

    if extension == ".pdf":

        loader = PyPDFLoader(
            str(file_path)
        )

        return loader.load()

    # -----------------------------------------------------
    # DOCX
    # -----------------------------------------------------

    if extension == ".docx":

        loader = Docx2txtLoader(
            str(file_path)
        )

        return loader.load()

    # -----------------------------------------------------
    # TXT
    # -----------------------------------------------------

    if extension == ".txt":

        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
        )

        return loader.load()

    # -----------------------------------------------------
    # Markdown
    # -----------------------------------------------------

    if extension == ".md":

        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
        )

        return loader.load()

    # -----------------------------------------------------
    # CSV
    # -----------------------------------------------------

    if extension == ".csv":

        loader = CSVLoader(
            str(file_path)
        )

        return loader.load()

    raise ValueError(
        f"Unsupported document type: {extension}"
    )


# =========================================================
# COMPLETE INGESTION PIPELINE
# =========================================================

def ingest_document(
    file_path: str,
    source: str,
    trusted_sources: set[str],
    expected_sha256: Optional[str] = None,
):
    """
    Complete document-ingestion pipeline.

    Flow:

        File
         ↓
        Type Validation
         ↓
        SHA-256 Integrity Check
         ↓
        Source Validation
         ↓
        Document Loading
         ↓
        Security Metadata
    """

    path = Path(file_path)

    # -----------------------------------------------------
    # 1. Check file exists
    # -----------------------------------------------------

    if not path.exists():

        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    # -----------------------------------------------------
    # 2. Validate file type
    # -----------------------------------------------------

    validate_file_type(path)

    # -----------------------------------------------------
    # 3. Verify integrity
    # -----------------------------------------------------

    actual_sha256 = verify_integrity(
        path,
        expected_sha256,
    )

    # -----------------------------------------------------
    # 4. Verify trusted source
    # -----------------------------------------------------

    validate_source(
        source,
        trusted_sources,
    )

    # -----------------------------------------------------
    # 5. Load document
    # -----------------------------------------------------

    documents = load_document(path)

    # -----------------------------------------------------
    # 6. Attach security metadata
    # -----------------------------------------------------

    for document in documents:

        document.metadata.update(
            {
                "filename": path.name,
                "source": source,
                "sha256": actual_sha256,
            }
        )

    return documents


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # DOCUMENT PATH
    # -----------------------------------------------------
    #
    # Your terminal is currently inside:
    #
    # 07_Project
    #
    # Therefore we use:
    #
    # data/documents/Internship report.pdf.pdf
    #
    # -----------------------------------------------------

    DOCUMENT = (
        "data/documents/"
        "Internship report.pdf.pdf"
    )

    # -----------------------------------------------------
    # TRUSTED SOURCES
    # -----------------------------------------------------

    TRUSTED_SOURCES = {
        "official_aws_docs",
        "research_paper",
        "internal_verified_docs",
    }

    # -----------------------------------------------------
    # RUN INGESTION
    # -----------------------------------------------------

    try:

        documents = ingest_document(
            file_path=DOCUMENT,
            source="official_aws_docs",
            trusted_sources=TRUSTED_SOURCES,
        )

        print(
            "\n========== INGESTION SUCCESS =========="
        )

        print(
            f"Pages/Documents loaded: "
            f"{len(documents)}"
        )

        print(
            f"Filename: "
            f"{documents[0].metadata['filename']}"
        )

        print(
            f"Source: "
            f"{documents[0].metadata['source']}"
        )

        print(
            f"SHA-256: "
            f"{documents[0].metadata['sha256']}"
        )

        print(
            "=======================================\n"
        )

    except Exception as error:

        print(
            "\n========== INGESTION FAILED =========="
        )

        print(error)

        print(
            "======================================\n"
        )