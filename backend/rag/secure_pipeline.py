from pathlib import Path
import re
import sys
import time
_DUMP_RE = re.compile(
    r"(?i)\b(verbatim|whole document|entire document|full document|"
    r"dump|print everything|output everything|all contents?|raw text)\b"
)


# =========================================================
# SAFE WINDOWS PRINT
# =========================================================

def _safe_print(*args, **kwargs) -> None:
    """
    Print text safely on Windows even when PDF text contains
    Unicode characters.
    """

    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    text = sep.join(str(arg) for arg in args) + end
    stream = kwargs.get("file", sys.stdout)

    try:
        stream.write(text)

        if hasattr(stream, "flush"):
            stream.flush()

    except UnicodeEncodeError:

        encoding = getattr(
            stream,
            "encoding",
            None,
        ) or "utf-8"

        if hasattr(stream, "buffer"):
            stream.buffer.write(
                text.encode(
                    encoding,
                    errors="replace",
                )
            )
        else:
            stream.write(
                text.encode(
                    encoding,
                    errors="replace",
                ).decode(
                    encoding,
                    errors="replace",
                )
            )

        if hasattr(stream, "flush"):
            stream.flush()


print = _safe_print


# =========================================================
# IMPORTS
# =========================================================

from backend.auth.authentication import authenticate
from backend.auth.models import User

from backend.retrieval.retriever import (
    retrieve_documents,
)

from backend.retrieval.reranker import (
    rerank_documents,
    rerank_with_floor,
)

from backend.retrieval.vector_store import (
    get_available_documents,
)

from backend.disclosure.controller import (
    control_disclosure,
)

from backend.rag.generator import (
    generate_answer,
    FALLBACK_ANSWER,
)

from backend.security.semantic_evidence_check import (
    SemanticEvidenceChecker,
)

from backend.security.evidence_relevance import (
    EvidenceRelevanceChecker,
)

from backend.security.verification_evidence import (
    build_nli_candidate_groups,
)


# =========================================================
# CONFIGURATION
# =========================================================

TOP_K_RETRIEVAL = 10
TOP_K_RERANKED = 5
RERANK_FLOOR = 0

EVIDENCE_THRESHOLD = 0.70

# Set True if you want to see all retrieved chunks.
# False is recommended when using the frontend.
DEBUG_EVIDENCE = True


# =========================================================
# LAZY SECURITY MODELS
# =========================================================

_relevance_checker: EvidenceRelevanceChecker | None = None
_evidence_checker: SemanticEvidenceChecker | None = None


def get_relevance_checker() -> EvidenceRelevanceChecker:

    global _relevance_checker

    if _relevance_checker is None:

        print(
            "Loading evidence relevance model..."
        )

        _relevance_checker = (
            EvidenceRelevanceChecker()
        )

    return _relevance_checker


def get_evidence_checker() -> SemanticEvidenceChecker:

    global _evidence_checker

    if _evidence_checker is None:

        print(
            "Loading semantic evidence model..."
        )

        _evidence_checker = (
            SemanticEvidenceChecker()
        )

    return _evidence_checker


# =========================================================
# CONTEXT
# =========================================================

def build_context(results) -> str:
    """
    Combine reranked document chunks into one evidence context.
    """

    return (
        "\n\n--- EVIDENCE ---\n\n"
        .join(
            document.page_content
            for document, _ in results
        )
    )


# =========================================================
# SOURCES
# =========================================================

def get_sources(results) -> list[str]:
    """
    Get unique source documents used for the answer.
    """

    sources: list[str] = []

    for document, _ in results:

        filename = document.metadata.get(
            "filename"
        )

        if (
            filename
            and filename not in sources
        ):

            sources.append(filename)

    return sources


# =========================================================
# ANSWER CLAIM SPLITTER
# =========================================================

# Splits a sentence into sub-clauses on internal connectors,
# so compound multi-fact sentences are verified fact-by-fact
# instead of as one unverifiable blob.
_CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:,\s+(?:while|whereas|and|but)\s+|;\s+|\s+while\s+)\s*"
)

# Matches enumeration claims like "... include A, B, and C"
# so each listed item can be verified independently.
_LIST_TRIGGER_RE = re.compile(
    r"(?i)^(?P<subject>.*?)\b(?P<trigger>includ(?:e[sd]?|ing)|such as|consist(?:s)? of|comprised of)\b\s*:?\s*(?P<items>.+?)\.?$"
)


MAX_ITEM_WORDS = 10


def split_list_claim(claim: str) -> list[str]:
    match = _LIST_TRIGGER_RE.match(claim.strip())

    if not match:
        return [claim]

    subject = match.group("subject").strip()

    if "," in subject:
        return [claim]

    trigger = match.group("trigger")

    items: list[str] = []

    for part in re.split(
        r",\s*",
        match.group("items").strip(),
    ):
        part = re.sub(
            r"(?i)^and\s+",
            "",
            part,
        ).strip().rstrip(".")

        if not part:
            continue

        if len(part.split()) <= 4:
            items.extend(
                p.strip()
                for p in re.split(
                    r"\s+and\s+",
                    part,
                )
                if p.strip()
            )
        else:
            items.append(part)

    if len(items) < 2:
        return [claim]

    subject_l = subject.lower()

    for item in items:

        if (
            not re.search(
                r"[A-Za-z0-9]",
                item,
            )
            or len(item.split()) > MAX_ITEM_WORDS
            or item.lower() in subject_l
        ):
            return [claim]

    return [
        f"{subject} {trigger} {item}."
        for item in items
    ]
def _split_clauses(sentence: str) -> list[str]:         
    parts = [p.strip().rstrip(",;") for p in _CLAUSE_SPLIT_RE.split(sentence) if p and p.strip()]
    if len(parts) > 1 and all(len(p.split()) >= 4 for p in parts):
        return parts
    return [sentence.strip().rstrip(",;")]  



def split_into_claims(
    answer: str,
) -> list[str]:
    """
    Split generated answer into meaningful claims.

    Each sentence is first isolated, then further decomposed
    into sub-clauses on connectors like "while", "whereas",
    ";", ", and", ", but" — so a single compound sentence
    describing two separate facts produces two claims, each
    independently checkable against evidence.
    """

    if not answer or not answer.strip():
        return []

    text = answer.replace(
        "\n",
        " ",
    ).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    claims: list[str] = []

    for sentence in sentences:

        sentence = sentence.strip()

        # Remove markdown/list numbering.
        sentence = re.sub(
            r"^\s*(?:[-*]|\d+[.)])\s*",
            "",
            sentence,
        ).strip()

        if not sentence:
            continue

        # -----------------------------------------------------
                # Lists first, on the WHOLE sentence, so ", and <last item>" is never cut off.
        list_claims = split_list_claim(sentence)
        if len(list_claims) > 1:
            claims.extend(list_claims)
            continue

        for clause in _split_clauses(sentence):
            if len(clause.split()) < 2:
                continue
            claims.extend(split_list_claim(clause))

    return claims


# =========================================================
# VERIFY ANSWER CLAIMS
# =========================================================

def _nli(checker, claim: str, premise: str):
    r = checker.check_claim(
        claim=claim,
        evidence=premise,
        threshold=EVIDENCE_THRESHOLD,
    )
    print(
        f"  NLI {r.label:13} score={r.score:.4f} "
        f"E={r.entailment_score:.4f} C={r.contradiction_score:.4f} "
        f"N={r.neutral_score:.4f}"
    )
    return r


def _is_entailed(r) -> bool:
    return (
        r.label == "ENTAILMENT"
        and r.score >= EVIDENCE_THRESHOLD
        and r.supported
    )


def verify_claim(claim: str, focused: list[str], broad: list[str], checker):
    """Returns (supported, result, premise). Assumes at least one premise."""

    best = None

    # Focused premises may approve and may reject.
    for premise in focused:
        r = _nli(checker, claim, premise)

        if best is None or r.score > best[0].score:
            best = (r, premise)

        # Strong entailment wins immediately.
        if _is_entailed(r):
            return True, r, premise

           # Strong contradiction rejects.
        if (
            r.label == "CONTRADICTION"
            and r.score >= EVIDENCE_THRESHOLD
        ):
            return False, r, premise

    # Broad premises may approve, but must never reject.
    for premise in broad:
        r = _nli(checker, claim, premise)

        if best is None or r.score > best[0].score:
            best = (r, premise)

        if _is_entailed(r):
            return True, r, premise

    return False, best[0], best[1]


def decompose_claim(claim: str) -> list[str]:
    """Split a multi-fact claim into self-contained parts, or return []."""
    text = claim.strip().rstrip(".")

    parts = [
        p.strip()
        for p in re.split(
            r";\s+|,\s+(?=(?:the ability|the development|and the development|and they achieved|ensuring that))",
            text,
            flags=re.IGNORECASE,
        )
        if p.strip()
    ]

    if not (2 <= len(parts) <= 8):
        return []

    if any(len(p.split()) < 3 for p in parts):
        return []

    return [p + "." for p in parts]


def verify_claim_with_fallback(claim: str, evidence: str, checker):
    """Returns None if no premise can be built, else (supported, result, premise)."""
    focused, broad = build_nli_candidate_groups(
        claim=claim,
        evidence=evidence,
    )

    if not focused and not broad:
        return None

    supported, result, premise = verify_claim(
        claim,
        focused,
        broad,
        checker,
    )

    if supported:
        return True, result, premise

    parts = decompose_claim(claim)

    if parts:
        print(f"  Whole claim failed; checking {len(parts)} sub-claims")
        outcomes = []

        for part in parts:
            f2, b2 = build_nli_candidate_groups(
                claim=part,
                evidence=evidence,
            )

            if not f2 and not b2:
                break

            ok, r2, p2 = verify_claim(
                part,
                f2,
                b2,
                checker,
            )

            if not ok:
                break

            outcomes.append((r2, p2))
        else:
            weakest = min(outcomes, key=lambda o: o[0].score)
            return True, weakest[0], "\n".join(
                p for _, p in outcomes
            )

    return False, result, premise


def verify_answer_claims(
    answer: str,
    evidence: str,
    checker: SemanticEvidenceChecker,
) -> dict:
    """
    Verify every generated claim against permitted evidence.

    Focused premises may approve or reject. Broad premises may
    approve but cannot reject solely because of contradiction.
    Failed multi-fact claims are decomposed and every part must pass.
    """

    claims = split_into_claims(answer)

    if not claims:
        return {
            "supported": False,
            "reason": "Generated answer contains no meaningful claims.",
            "claims": [],
        }

    if not evidence.strip():
        return {
            "supported": False,
            "reason": "No permitted evidence is available.",
            "claims": [],
        }

    verified_claims: list[dict] = []

    for claim in claims:
        print("\n----------------------------------------")
        print(f"Verifying claim: {claim}")

        outcome = verify_claim_with_fallback(
            claim,
            evidence,
            checker,
        )

        if outcome is None:
            verified_claims.append({
                "claim": claim,
                "supported": False,
                "score": 0.0,
                "label": "NO_EVIDENCE",
                "semantic_score": 0.0,
                "reason": "No verification evidence could be constructed for the claim.",
                "evidence": "",
            })
            continue

        supported, r, premise = outcome

        if DEBUG_EVIDENCE:
            print("\nDeciding evidence:")
            print(premise[:1500])

        reason = (
            r.reason
            if (supported or r.label == "CONTRADICTION")
            else "The claim was not semantically entailed by the permitted evidence."
        )

        verified_claims.append({
            "claim": claim,
            "supported": supported,
            "score": r.score,
            "label": r.label,
            "semantic_score": r.score,
            "reason": reason,
            "evidence": premise,
        })

        print(
            f"Final: supported={supported} "
            f"label={r.label} score={r.score:.4f}"
        )

    unsupported = [
        c for c in verified_claims
        if not c["supported"]
    ]

    if unsupported:
        failed = "; ".join(
            c["claim"]
            for c in unsupported
        )
        return {
            "supported": False,
            "reason": f"Unsupported claim(s): {failed}",
            "claims": verified_claims,
        }

    return {
        "supported": True,
        "reason": "All generated claims were semantically entailed by permitted evidence.",
        "claims": verified_claims,
    }


# =========================================================
# SECURE ASK
# =========================================================

def secure_ask(
    user: User,
    query: str,
    vector_db: str,
    filename: str | None = None,
) -> dict:
    """
    End-to-end Secure AskRAG pipeline.

    Flow:

        Query
          ↓
        Retrieval
          ↓
        Reranking
          ↓
        Evidence Relevance
          ↓
        Role-aware Disclosure
          ↓
        Security Gate
          ↓
        LLM Generation
          ↓
        Claim-level NLI Verification
          ↓
        ALLOW / DENY
    """

    total_start = time.perf_counter()
    if _DUMP_RE.search(query) and user.role.value.upper() != "ADMIN":
        return {
            "decision": "DENY",
            "answer": "I cannot provide that information under the current security policy.",
            "reason": "Bulk document extraction is restricted for this role.",
        }

    # =====================================================
    # 1. RETRIEVAL
    # =====================================================

    retrieval_start = time.perf_counter()

    candidates = retrieve_documents(
        query=query,
        persist_directory=vector_db,
        top_k=TOP_K_RETRIEVAL,
        filename=filename,
    )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    print(
        f"[TIMING] Retrieval: "
        f"{retrieval_time:.2f}s"
    )

    if not candidates:

        total_time = (
            time.perf_counter()
            - total_start
        )

        print(
            f"[TIMING] TOTAL: "
            f"{total_time:.2f}s"
        )

        return {
            "decision": "DENY",
            "answer": (
                "I could not find the answer "
                "in the selected document."
            ),
            "reason": "No evidence retrieved.",
        }

    # =====================================================
    # 2. RERANKING
    # =====================================================

    rerank_start = time.perf_counter()

    reranked = rerank_with_floor(
        query=query,
        results=candidates,
        top_k=TOP_K_RERANKED,
        floor=RERANK_FLOOR,
    )

    rerank_time = (
        time.perf_counter()
        - rerank_start
    )

    print(
        f"[TIMING] Reranking: "
        f"{rerank_time:.2f}s"
    )

    if not reranked:

        total_time = (
            time.perf_counter()
            - total_start
        )

        print(
            f"[TIMING] TOTAL: "
            f"{total_time:.2f}s"
        )

        return {
            "decision": "DENY",
            "answer": (
                "I could not find the answer "
                "in the selected document."
            ),
            "reason": "No relevant evidence found.",
        }

    evidence = build_context(
        reranked
    )

    # =====================================================
    # DEBUG RETRIEVED EVIDENCE
    # =====================================================

    if DEBUG_EVIDENCE:

        print(
            "\n========== RETRIEVED EVIDENCE =========="
        )

        print(
            "Selected document: "
            f"{filename if filename else 'ALL DOCUMENTS'}"
        )

        print(
            f"Reranked chunks: "
            f"{len(reranked)}"
        )

        for index, (
            document,
            score,
        ) in enumerate(
            reranked,
            start=1,
        ):

            print(
                f"\n--- Evidence {index} ---"
            )

            print(
                "Source: "
                f"{document.metadata.get('filename')}"
            )

            print(
                "Chunk ID: "
                f"{document.metadata.get('chunk_id')}"
            )

            print(
                "Reranker Score: "
                f"{score:.4f}"
            )

            print(
                "\nContent:"
            )

            print(
                document.page_content[:1500]
            )

        print(
            "\n========================================"
        )

    # =====================================================
    # 3. EVIDENCE RELEVANCE
    # =====================================================

    relevance_start = time.perf_counter()

    evidence_chunks = [
        document.page_content
        for document, _ in reranked
    ]

    relevance = (
        get_relevance_checker().check(
            query=query,
            evidence_chunks=evidence_chunks,
        )
    )

    relevance_time = (
        time.perf_counter()
        - relevance_start
    )

    print(
        f"[TIMING] Evidence relevance: "
        f"{relevance_time:.2f}s"
    )

    print(
        "\n========== RELEVANCE CHECK =========="
    )

    print(
        f"Relevant: "
        f"{relevance['relevant']}"
    )

    print(
        f"Score: "
        f"{relevance['score']:.4f}"
    )

    print(
        f"Reason: "
        f"{relevance['reason']}"
    )

    print(
        "====================================="
    )

    if not relevance["relevant"]:

        total_time = (
            time.perf_counter()
            - total_start
        )

        print(
            f"[TIMING] TOTAL: "
            f"{total_time:.2f}s"
        )

        return {
            "decision": "DENY",
            "answer": (
                "I could not find the answer "
                "in the provided document."
            ),
            "reason": relevance["reason"],
            "relevance_score": relevance["score"],
            "evidence_score": 0.0,
            "sources": get_sources(
                reranked
            ),
        }

    # =====================================================
    # 4. ROLE-AWARE DISCLOSURE
    # =====================================================

    disclosure_start = time.perf_counter()

    disclosure = control_disclosure(
        user=user,
        query=query,
        evidence=evidence,
    )

    disclosure_time = (
        time.perf_counter()
        - disclosure_start
    )

    print(
        f"[TIMING] Disclosure control: "
        f"{disclosure_time:.2f}s"
    )

    # -----------------------------------------------------
    # Explicit disclosure denial
    # -----------------------------------------------------

    if disclosure.decision == "DENY":

        total_time = (
            time.perf_counter()
            - total_start
        )

        print(
            f"[TIMING] TOTAL: "
            f"{total_time:.2f}s"
        )

        return {
            "decision": "DENY",
            "answer": (
                "I cannot provide that information "
                "under the current security policy."
            ),
            "reason": "; ".join(
                disclosure.reasons
            ),
            "risk_level": disclosure.risk_level,
            "risk_score": disclosure.risk_score,
            "relevance_score": relevance["score"],
            "evidence_score": 0.0,
            "sources": get_sources(
                reranked
            ),
        }

    sanitized_evidence = (
        disclosure.sanitized_evidence
    )

    # -----------------------------------------------------
    # No evidence after disclosure
    # -----------------------------------------------------

    if not sanitized_evidence.strip():

        total_time = (
            time.perf_counter()
            - total_start
        )

        print(
            f"[TIMING] TOTAL: "
            f"{total_time:.2f}s"
        )

        return {
            "decision": "DENY",
            "answer": (
                "I cannot provide that information "
                "under the current security policy."
            ),
            "reason": (
                "No permitted evidence remained "
                "after disclosure control."
            ),
            "relevance_score": relevance["score"],
            "evidence_score": 0.0,
            "sources": get_sources(
                reranked
            ),
        }

        # =====================================================
    # 5. SECURITY DISCLOSURE GATE
    # =====================================================

    evidence_was_redacted = (
        sanitized_evidence.strip()
        != evidence.strip()
    )

    # -----------------------------------------------------
    # LIMITED DISCLOSURE
    #
    # Sensitive fields may have been removed from the
    # evidence. The sanitized evidence is still safe to
    # use for answering non-sensitive questions.
    #
    # IMPORTANT:
    # The original evidence is NEVER passed to the LLM
    # when disclosure is LIMITED.
    # -----------------------------------------------------

    if disclosure.decision == "LIMITED":

        print(
            "\n========== LIMITED DISCLOSURE =========="
        )

        print(
            "Restricted information was "
            "redacted for this user."
        )

        print(
            f"User: {user.user_id}"
        )

        print(
            f"Role: {user.role.value}"
        )

        print(
            "Only sanitized evidence will be "
            "provided to the generator."
        )

        print(
            "Original restricted evidence will "
            "NOT be sent to the LLM."
        )

        print(
            "========================================"
        )

    # -----------------------------------------------------
    # ALLOW
    #
    # No restricted information was removed.
    # Normal generation can proceed.
    # -----------------------------------------------------

    elif disclosure.decision == "ALLOW":

        print(
            "\n========== DISCLOSURE ALLOWED =========="
        )

        print(
            "No unauthorized sensitive information "
            "was detected."
        )

        print(
            "========================================"
        )

    # -----------------------------------------------------
    # Defensive fallback
    #
    # The controller should already return DENY above.
    # This protects the pipeline if a future disclosure
    # state is introduced unexpectedly.
    # -----------------------------------------------------

    else:

        total_time = (
            time.perf_counter()
            - total_start
        )

        print(
            "[SECURITY] Unknown disclosure state. "
            "Failing closed."
        )

        return {
            "decision": "DENY",
            "answer": (
                "I cannot provide that information "
                "under the current security policy."
            ),
            "reason": (
                "Unsupported disclosure decision. "
                "Security pipeline failed closed."
            ),
            "risk_level": disclosure.risk_level,
            "risk_score": disclosure.risk_score,
            "relevance_score": relevance["score"],
            "evidence_score": 0.0,
            "sources": get_sources(
                reranked
            ),
        }

    # =====================================================
    # 6. SANITIZED EVIDENCE
    # =====================================================

    if DEBUG_EVIDENCE:

        print(
            "\n========== SANITIZED EVIDENCE =========="
        )

        print(
            sanitized_evidence
        )

        print(
            "\n========================================="
        )

    # =====================================================
    # 7. LLM GENERATION
    # =====================================================

    generation_start = time.perf_counter()

    answer = generate_answer(
        query=query,
        context=sanitized_evidence,
    )

    generation_time = (
        time.perf_counter()
        - generation_start
    )

    print(
        f"[TIMING] LLM generation: "
        f"{generation_time:.2f}s"
    )

    print(
        "\n========== GENERATED ANSWER =========="
    )

    print(
        answer
    )

    print(
        "======================================="
    )
    if answer.strip() == FALLBACK_ANSWER:
        return {
            "decision": "DENY",
            "answer": FALLBACK_ANSWER,
            "reason": "No supporting information found in the permitted evidence.",
        }
    # =====================================================
    # 8. SEMANTIC VERIFICATION
    # =====================================================

    verification_start = time.perf_counter()

    verification = verify_answer_claims(
        answer=answer,
        evidence=sanitized_evidence,
        checker=get_evidence_checker(),
    )

    verification_time = (
        time.perf_counter()
        - verification_start
    )

    print(
        f"[TIMING] NLI verification: "
        f"{verification_time:.2f}s"
    )

    # =====================================================
    # 9. VERIFICATION DETAILS
    # =====================================================

    print(
        "\n========== EVIDENCE VERIFICATION =========="
    )

    for claim in verification["claims"]:

        print(
            f"\nClaim: "
            f"{claim['claim']}"
        )

        print(
            f"Supported: "
            f"{claim['supported']}"
        )

        print(
            f"Semantic Label: "
            f"{claim['label']}"
        )

        print(
            f"Semantic Score: "
            f"{claim['semantic_score']:.4f}"
        )

        print(
            f"Reason: "
            f"{claim['reason']}"
        )

        if (
            DEBUG_EVIDENCE
            and claim["evidence"]
        ):

            print(
                "\nSupporting evidence:"
            )

            print(
                claim["evidence"][:1000]
            )

    print(
        "\nOverall verification:"
    )

    print(
        verification["supported"]
    )

    print(
        f"Reason: "
        f"{verification['reason']}"
    )

    print(
        "============================================"
    )

    # =====================================================
    # 10. VERIFICATION FAILURE
    # =====================================================

    if not verification["supported"]:

        total_time = (
            time.perf_counter()
            - total_start
        )

        print(
            f"[TIMING] TOTAL: "
            f"{total_time:.2f}s"
        )

        return {
            "decision": "DENY",
            "answer": (
                "I could not verify the generated "
                "answer against the permitted evidence."
            ),
            "reason": verification["reason"],
            "evidence_score": 0.0,
            "relevance_score": relevance["score"],
            "sources": get_sources(
                reranked
            ),
        }

    # =====================================================
    # 11. FINAL ALLOW
    # =====================================================

    total_time = (
        time.perf_counter()
        - total_start
    )

    print(
        "\n============================================"
    )

    print(
        "[TIMING] PERFORMANCE SUMMARY"
    )

    print(
        f"[TIMING] Retrieval: "
        f"{retrieval_time:.2f}s"
    )

    print(
        f"[TIMING] Reranking: "
        f"{rerank_time:.2f}s"
    )

    print(
        f"[TIMING] Evidence relevance: "
        f"{relevance_time:.2f}s"
    )

    print(
        f"[TIMING] Disclosure control: "
        f"{disclosure_time:.2f}s"
    )

    print(
        f"[TIMING] LLM generation: "
        f"{generation_time:.2f}s"
    )

    print(
        f"[TIMING] NLI verification: "
        f"{verification_time:.2f}s"
    )

    print(
        f"[TIMING] TOTAL: "
        f"{total_time:.2f}s"
    )

    print(
    "=============================================="
)

    final_decision = disclosure.decision
    final_reason = "; ".join(
        disclosure.reasons
    )

    if (
            final_decision == "LIMITED"
            and "[REDACTED]" not in answer
    ):
            final_decision = "ALLOW"
            final_reason = ""

    return {
            "decision": final_decision,
            "answer": answer,
            "reason": final_reason,
            "risk_level": disclosure.risk_level,
            "risk_score": disclosure.risk_score,
            "evidence_score": 1.0,
            "relevance_score": relevance["score"],
            "sources": get_sources(
                reranked
            ),
            "timings": {
                "retrieval": retrieval_time,
                "reranking": rerank_time,
                "evidence_relevance": relevance_time,
                "disclosure": disclosure_time,
                "llm_generation": generation_time,
                "nli_verification": verification_time,
                "total": total_time,
            },
        }


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    PROJECT_DIR = (
        Path(__file__).resolve().parents[2]
    )

    VECTOR_DB = (
        PROJECT_DIR / "vector_db"
    )

    print(
        "\n=========================================="
    )

    print(
        "        SECURE ASKRAG PIPELINE"
    )

    print(
        "=========================================="
    )

    # =====================================================
    # LOGIN
    # =====================================================

    print(
        "\n========== LOGIN =========="
    )

    user_id = input(
        "Username: "
    ).strip()

    password = input(
        "Password: "
    ).strip()

    user = authenticate(
        user_id=user_id,
        password=password,
    )

    if user is None:

        print(
            "\nAuthentication FAILED."
        )

        print(
            "Invalid username or password."
        )

        print(
            "Access denied."
        )

        raise SystemExit(1)

    print(
        "\nAuthentication SUCCESS."
    )

    print(
        f"User: {user.user_id}"
    )

    print(
        f"Role: {user.role.value}"
    )

    # =====================================================
    # AVAILABLE DOCUMENTS
    # =====================================================

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

        print(
            error
        )

        raise SystemExit(1)

    if not available_documents:

        print(
            "\nNo documents are currently available."
        )

        print(
            "Please ingest a document first."
        )

        raise SystemExit(1)

    # =====================================================
    # DOCUMENT LOOP
    # =====================================================

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

        if selection == "0":

            selected_filename = None

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

        if selected_filename:

            print(
                "\nSelected document: "
                f"{selected_filename}"
            )

        else:

            print(
                "\nSelected: ALL DOCUMENTS"
            )

        # =================================================
        # QUESTION LOOP
        # =================================================

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

                result = secure_ask(
                    user=user,
                    query=query,
                    vector_db=str(
                        VECTOR_DB
                    ),
                    filename=selected_filename,
                )

                print(
                    "\n========== SECURITY RESULT =========="
                )

                print(
                    f"Decision: "
                    f"{result['decision']}"
                )

                if "risk_level" in result:

                    print(
                        f"Risk: "
                        f"{result['risk_level']} "
                        f"({result['risk_score']})"
                    )

                if "relevance_score" in result:

                    print(
                        "Relevance score: "
                        f"{result['relevance_score']:.4f}"
                    )

                if "evidence_score" in result:

                    print(
                        "Evidence score: "
                        f"{result['evidence_score']:.4f}"
                    )

                print(
                    "\nAnswer:"
                )

                print(
                    result["answer"]
                )

                if result.get("sources"):

                    print(
                        "\nSource:"
                    )

                    for source in result[
                        "sources"
                    ]:

                        print(
                            f"- {source}"
                        )

                print(
                    "\nReason:"
                )

                print(
                    result["reason"]
                )

                print(
                    "=====================================\n"
                )

            except Exception as error:

                print(
                    "\nPipeline error:"
                )

                print(
                    error
                )