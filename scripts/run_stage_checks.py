"""
Stage-by-stage Secure AskRAG checks.

Usage (from 07_Project):

    set PYTHONPATH=.
    set PYTHONIOENCODING=utf-8
    python scripts/run_stage_checks.py
"""

from __future__ import annotations

from backend.auth.authentication import authenticate
from backend.rag.generator import (
    focus_context_for_question,
    generate_answer,
)
from backend.rag.secure_pipeline import (
    build_context,
    secure_ask,
    verify_answer_claims,
)
from backend.retrieval.reranker import rerank_documents
from backend.retrieval.retriever import retrieve_documents
from backend.security.semantic_evidence_check import (
    SemanticEvidenceChecker,
)
from backend.security.verification_evidence import (
    select_verification_evidence,
)


QUERY = "What software and tools were used in the research?"
FILENAME = "paper (1).pdf"
VECTOR_DB = "vector_db"


def main() -> None:
    print("\n=== 1) RETRIEVAL ===")
    candidates = retrieve_documents(
        query=QUERY,
        persist_directory=VECTOR_DB,
        top_k=20,
        filename=FILENAME,
    )
    print(f"Candidates: {len(candidates)}")

    print("\n=== 2) RERANK ===")
    reranked = rerank_documents(
        query=QUERY,
        results=candidates,
        top_k=3,
    )
    evidence = build_context(reranked)
    print(f"Reranked: {len(reranked)}")
    print("Has software list:", "Scikit-learn" in evidence)
    print("Has Visual Studio:", "Visual Studio" in evidence)

    print("\n=== 3) GENERATION FOCUS ===")
    focused = focus_context_for_question(QUERY, evidence)
    print(focused[:1200])
    print("...")

    print("\n=== 4) GENERATION ===")
    answer = generate_answer(QUERY, evidence)
    print(answer)

    print("\n=== 5) VERIFIER EVIDENCE ===")
    targeted = select_verification_evidence(answer, evidence)
    print(targeted[:1200])

    print("\n=== 6) NLI VERIFICATION ===")
    checker = SemanticEvidenceChecker()
    verification = verify_answer_claims(
        answer=answer,
        evidence=evidence,
        checker=checker,
    )
    print("Supported:", verification["supported"])
    print("Reason:", verification["reason"])

    print("\n=== 7) END-TO-END SECURE ASK ===")
    import os
    user = authenticate("admin_001", os.getenv("ADMIN_PASSWORD"))
    
    result = secure_ask(
        user=user,
        query=QUERY,
        vector_db=VECTOR_DB,
        filename=FILENAME,
    )
    print("Decision:", result["decision"])
    print("Answer:", result["answer"])


if __name__ == "__main__":
    main()
