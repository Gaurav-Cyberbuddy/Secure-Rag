from backend.security.semantic_evidence_check import SemanticEvidenceChecker

checker = SemanticEvidenceChecker()

premises = {
    "identical":  "Administrator email: admin-test@example.com",
    "raw_doc":    "SECURITY TEST DOCUMENT\nAdministrator email: admin-test@example.com\nAPI key: TEST-API-KEY-12345\n\nThe project uses Python and PyTorch.",
    "flattened":  "SECURITY TEST DOCUMENT, Administrator email: admin-test@example.com, API key: TEST-API-KEY-12345.\nThe project uses Python and PyTorch.",
}

claims = {
    "fragment": "Administrator email: admin-test@example.com",
    "sentence": "The administrator's email address is admin-test@example.com.",
}

for pn, p in premises.items():
    for cn, c in claims.items():
        r = checker.check_claim(
            claim=c,
            evidence=p,
            threshold=0.70,
        )

        print(
            f"{pn:10} | {cn:9} | {r.label:13} "
            f"E={r.entailment_score:.3f} "
            f"C={r.contradiction_score:.3f} "
            f"N={r.neutral_score:.3f}"
        )