from dataclasses import dataclass
import re


@dataclass(frozen=True)
class EvidenceCheckResult:
    supported: bool
    score: float
    reason: str


def normalize_text(text: str) -> set[str]:
    """
    Convert text into normalized word tokens.
    """

    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )
    )


def check_answer_against_evidence(
    answer: str,
    evidence: str,
    minimum_score: float = 0.20,
) -> EvidenceCheckResult:
    """
    Perform a lightweight lexical evidence-support check.

    This is a baseline check. It measures how much of the
    answer's vocabulary is supported by the supplied evidence.
    """

    answer_words = normalize_text(answer)
    evidence_words = normalize_text(evidence)

    if not answer_words:
        return EvidenceCheckResult(
            supported=False,
            score=0.0,
            reason="Generated answer is empty.",
        )

    if not evidence_words:
        return EvidenceCheckResult(
            supported=False,
            score=0.0,
            reason="No evidence is available.",
        )

    supported_words = answer_words.intersection(
        evidence_words
    )

    score = len(supported_words) / len(answer_words)

    if score >= minimum_score:

        return EvidenceCheckResult(
            supported=True,
            score=score,
            reason="Answer contains sufficient evidence-supported content.",
        )

    return EvidenceCheckResult(
        supported=False,
        score=score,
        reason="Answer contains insufficient evidence-supported content.",
    )


if __name__ == "__main__":

    print("\n========== EVIDENCE CHECK TEST ==========")

    evidence = """
    The internship project used AWS S3 for object storage.
    The project focused on secure cloud storage.
    """

    supported_answer = (
        "The project used AWS S3 for object storage."
    )

    unsupported_answer = (
        "The project used Microsoft Azure Kubernetes "
        "with a blockchain-based architecture."
    )

    result_1 = check_answer_against_evidence(
        answer=supported_answer,
        evidence=evidence,
    )

    result_2 = check_answer_against_evidence(
        answer=unsupported_answer,
        evidence=evidence,
    )

    print("\nSupported answer:")
    print(result_1)

    print("\nUnsupported answer:")
    print(result_2)

    print("\n=========================================")