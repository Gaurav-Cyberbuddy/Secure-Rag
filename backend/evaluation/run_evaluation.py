import csv
import time
from pathlib import Path

from backend.auth.models import User, Role
from backend.rag.secure_pipeline import secure_ask
from backend.api.main import VECTOR_DB


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TEST_CASES = BASE_DIR / "test_cases.csv"
RESULTS = BASE_DIR / "results.csv"


# ============================================================
# ROLE CONVERSION
# ============================================================

ROLE_MAP = {
    "PUBLIC": Role.PUBLIC,
    "EMPLOYEE": Role.EMPLOYEE,
    "MANAGER": Role.MANAGER,
    "ADMIN": Role.ADMIN,
}


# ============================================================
# HELPER
# ============================================================

def get_value(result, *keys, default=""):
    """
    Safely retrieve a value from the Secure AskRAG response.
    """
    for key in keys:
        if key in result:
            return result[key]
    return default


# ============================================================
# RUN EVALUATION
# ============================================================

def run_evaluation():

    if not TEST_CASES.exists():
        print(f"ERROR: Test case file not found: {TEST_CASES}")
        return

    with open(
        TEST_CASES,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        test_cases = list(csv.DictReader(file))

    print("=" * 70)
    print("SECURE ASKRAG EVALUATION")
    print("=" * 70)

    print(f"Test cases: {len(test_cases)}")
    print(f"Vector DB: {VECTOR_DB}")
    print()

    results = []

    for index, case in enumerate(test_cases, start=1):

        case_id = case["id"].strip()
        category = case["category"].strip()
        query = case["query"].strip()
        role_name = case["user_role"].strip().upper()
        document = case["document"].strip()
        expected_decision = case["expected_decision"].strip().upper()
        expected_grounded = case["expected_grounded"].strip().upper()

        print("-" * 70)
        print(f"[{index}/{len(test_cases)}] {case_id}")
        print(f"Category : {category}")
        print(f"Role     : {role_name}")
        print(f"Query    : {query}")
        print(f"Document : {document or 'ALL DOCUMENTS'}")

        # ----------------------------------------------------
        # Validate role
        # ----------------------------------------------------

        if role_name not in ROLE_MAP:

            print(f"ERROR: Unknown role: {role_name}")

            results.append({
                "id": case_id,
                "category": category,
                "query": query,
                "user_role": role_name,
                "document": document,
                "expected_decision": expected_decision,
                "actual_decision": "ERROR",
                "expected_grounded": expected_grounded,
                "answer": "",
                "reason": "Unknown role",
                "evidence_score": "",
                "nli_result": "",
                "latency_seconds": "",
                "status": "ERROR",
            })

            continue

        user = User(
            user_id=f"evaluation_{role_name.lower()}",
            role=ROLE_MAP[role_name],
        )

        # ----------------------------------------------------
        # Run actual Secure AskRAG
        # ----------------------------------------------------

        start_time = time.perf_counter()

        try:

            result = secure_ask(
                user=user,
                query=query,
                vector_db=str(VECTOR_DB),
                filename=document if document else None,
            )

            latency = time.perf_counter() - start_time

        except Exception as exc:

            latency = time.perf_counter() - start_time

            print(f"ERROR: {exc}")

            results.append({
                "id": case_id,
                "category": category,
                "query": query,
                "user_role": role_name,
                "document": document,
                "expected_decision": expected_decision,
                "actual_decision": "ERROR",
                "expected_grounded": expected_grounded,
                "answer": "",
                "reason": str(exc),
                "evidence_score": "",
                "nli_result": "",
                "latency_seconds": f"{latency:.4f}",
                "status": "ERROR",
            })

            continue

        # ----------------------------------------------------
        # Extract response
        # ----------------------------------------------------

        actual_decision = str(
            get_value(
                result,
                "decision",
                default=""
            )
        ).upper()

        answer = get_value(
            result,
            "answer",
            default=""
        )

        reason = get_value(
            result,
            "reason",
            "security_reason",
            default=""
        )

        evidence_score = get_value(
            result,
            "evidence_score",
            "relevance_score",
            "score",
            default=""
        )

        nli_result = get_value(
            result,
            "nli_result",
            "verification",
            "verification_result",
            default=""
        )

        # ----------------------------------------------------
        # PASS / FAIL
        # ----------------------------------------------------

        status = (
            "PASS"
            if actual_decision == expected_decision
            else "FAIL"
        )

        print(f"Expected : {expected_decision}")
        print(f"Actual   : {actual_decision}")
        print(f"Latency  : {latency:.4f}s")
        print(f"Status   : {status}")

        results.append({
            "id": case_id,
            "category": category,
            "query": query,
            "user_role": role_name,
            "document": document,
            "expected_decision": expected_decision,
            "actual_decision": actual_decision,
            "expected_grounded": expected_grounded,
            "answer": answer,
            "reason": reason,
            "evidence_score": evidence_score,
            "nli_result": nli_result,
            "latency_seconds": f"{latency:.4f}",
            "status": status,
        })

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    fieldnames = [
        "id",
        "category",
        "query",
        "user_role",
        "document",
        "expected_decision",
        "actual_decision",
        "expected_grounded",
        "answer",
        "reason",
        "evidence_score",
        "nli_result",
        "latency_seconds",
        "status",
    ]

    with open(
        RESULTS,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(results)

    passed = sum(
        1
        for row in results
        if row["status"] == "PASS"
    )

    failed = sum(
        1
        for row in results
        if row["status"] == "FAIL"
    )

    errors = sum(
        1
        for row in results
        if row["status"] == "ERROR"
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(f"Total cases : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")
    print(f"Errors      : {errors}")

    if total > 0:
        print(
            f"Accuracy    : {(passed / total) * 100:.2f}%"
        )

    print()
    print(f"Results saved to:")
    print(RESULTS)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_evaluation()