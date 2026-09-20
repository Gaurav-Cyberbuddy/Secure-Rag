"""
Semantic verification behavior tests.

Covers:
  - supported complete answer
  - hallucinated technology
  - partially supported answer
  - unrelated question / unsupported claim
  - structured list evidence
  - incomplete generated answer (still must be entailed if true)
"""

import pytest

from backend.security.semantic_evidence_check import (
    SemanticEvidenceChecker,
)
from backend.security.verification_evidence import (
    select_verification_evidence,
)
from backend.rag.secure_pipeline import (
    verify_answer_claims,
    split_into_claims,
)


SOFTWARE_EVIDENCE = """
All experiments were conducted using Python and the PyTorch deep learning framework.
The implementation was developed and executed in Visual Studio Code on a Windows-based system.
The primary software components used in this research include:
* Python 3.x
* PyTorch
* Torchvision
* Matplotlib
* NumPy
* Scikit-learn
"""


@pytest.fixture(scope="module")
def checker():
    return SemanticEvidenceChecker()


def test_supported_complete_answer(checker):
    claim = (
        "The research used Python 3.x, PyTorch, Torchvision, "
        "Matplotlib, NumPy, Scikit-learn, and Visual Studio Code "
        "on a Windows-based system."
    )

    evidence = select_verification_evidence(
        claim=claim,
        evidence=SOFTWARE_EVIDENCE,
    )

    result = checker.check_claim(
        claim=claim,
        evidence=evidence,
    )

    assert result.supported is True
    assert result.label == "ENTAILMENT"
    assert result.score >= 0.70


def test_hallucinated_technology_is_rejected(checker):
    claim = "The research used TensorFlow and Keras."

    evidence = select_verification_evidence(
        claim=claim,
        evidence=SOFTWARE_EVIDENCE,
    ) or SOFTWARE_EVIDENCE

    result = checker.check_claim(
        claim=claim,
        evidence=evidence,
    )

    assert result.supported is False
    assert result.label in {"NEUTRAL", "CONTRADICTION"}


def test_partially_supported_answer_with_hallucination(checker):
    claim = (
        "The research used Python, PyTorch, and TensorFlow."
    )

    evidence = select_verification_evidence(
        claim=claim,
        evidence=SOFTWARE_EVIDENCE,
    )

    result = checker.check_claim(
        claim=claim,
        evidence=evidence,
    )

    assert result.supported is False


def test_unrelated_claim_is_rejected(checker):
    claim = (
        "The research used blockchain technology for "
        "secure model training."
    )

    result = checker.check_claim(
        claim=claim,
        evidence=SOFTWARE_EVIDENCE,
    )

    assert result.supported is False


def test_structured_list_evidence_supports_full_claim(checker):
    """
    Regression for the original failure mode:

    targeted evidence must not shrink a six-item software list
    down to a single bullet before NLI.
    """

    claim = (
        "The primary software components included Python 3.x, "
        "PyTorch, Torchvision, Matplotlib, NumPy, and "
        "Scikit-learn."
    )

    evidence = select_verification_evidence(
        claim=claim,
        evidence=SOFTWARE_EVIDENCE,
    )

    for item in (
        "Python 3.x",
        "PyTorch",
        "Torchvision",
        "Matplotlib",
        "NumPy",
        "Scikit-learn",
    ):
        assert item in evidence

    result = checker.check_claim(
        claim=claim,
        evidence=evidence,
    )

    assert result.supported is True
    assert result.label == "ENTAILMENT"


def test_incomplete_but_true_answer_can_be_supported(checker):
    """
    An incomplete answer that only states true subset facts
    may be entailed. Completeness is a generation concern;
    verification checks support, not exhaustiveness.
    """

    claim = "The research used Python and PyTorch."

    evidence = select_verification_evidence(
        claim=claim,
        evidence=SOFTWARE_EVIDENCE,
    )

    result = checker.check_claim(
        claim=claim,
        evidence=evidence,
    )

    assert result.supported is True
    assert result.label == "ENTAILMENT"


def test_verify_answer_claims_allows_supported_answer(checker):
    answer = (
        "The research used Python 3.x, PyTorch, Torchvision, "
        "Matplotlib, NumPy, Scikit-learn, and Visual Studio Code "
        "on a Windows-based system."
    )

    verification = verify_answer_claims(
        answer=answer,
        evidence=SOFTWARE_EVIDENCE,
        checker=checker,
    )

    assert verification["supported"] is True
    assert split_into_claims(answer)


def test_verify_answer_claims_denies_hallucination(checker):
    answer = (
        "The research used Microsoft Azure and blockchain tools."
    )

    verification = verify_answer_claims(
        answer=answer,
        evidence=SOFTWARE_EVIDENCE,
        checker=checker,
    )

    assert verification["supported"] is False
