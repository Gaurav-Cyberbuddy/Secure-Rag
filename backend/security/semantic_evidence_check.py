from dataclasses import dataclass

import torch
from sentence_transformers import CrossEncoder


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

DEFAULT_THRESHOLD = 0.70

# Minimum advantage that entailment should have over
# the next strongest NLI class.
DEFAULT_MARGIN = 0.10


# =========================================================
# MODEL CACHE
# =========================================================

_model = None


def get_model(model_name: str = MODEL_NAME) -> CrossEncoder:
    """
    Load the NLI model only once and reuse it.
    """

    global _model

    if _model is None:

        print(
            "\n[MODEL] Loading semantic evidence model..."
        )

        _model = CrossEncoder(
            model_name
        )

        print(
            "[MODEL] Semantic evidence model loaded."
        )

    return _model


# =========================================================
# RESULT
# =========================================================

@dataclass(frozen=True)
class SemanticEvidenceResult:

    supported: bool
    score: float
    label: str
    reason: str

    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    neutral_score: float = 0.0


# =========================================================
# SEMANTIC EVIDENCE CHECKER
# =========================================================

class SemanticEvidenceChecker:
    """
    Semantic evidence verifier using Natural Language Inference.

    The model checks whether the supplied evidence semantically
    supports the generated claim.

    Security principle:

        ENTAILMENT
            -> potentially supported

        CONTRADICTION
            -> rejected

        NEUTRAL
            -> rejected

    Lexical similarity is NOT treated as semantic support.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ):

        self.model = get_model(
            model_name
        )

        # -----------------------------------------------------
        # Detect model labels
        # -----------------------------------------------------

        self.label_mapping = {
            int(index): str(label).lower()
            for index, label
            in self.model.model.config.id2label.items()
        }

        # -----------------------------------------------------
        # Find NLI class indexes
        # -----------------------------------------------------

        self.entailment_index = self._find_label(
            "entailment"
        )

        self.contradiction_index = self._find_label(
            "contradiction"
        )

        self.neutral_index = self._find_label(
            "neutral"
        )

        print(
            f"Entailment index: "
            f"{self.entailment_index}"
        )

        print(
            f"Contradiction index: "
            f"{self.contradiction_index}"
        )

        print(
            f"Neutral index: "
            f"{self.neutral_index}"
        )

    # =========================================================
    # FIND LABEL
    # =========================================================

    def _find_label(
        self,
        target_label: str,
    ) -> int:

        for index, label in self.label_mapping.items():

            normalized_label = (
                label
                .strip()
                .lower()
            )

            if normalized_label == target_label:

                return index

        raise ValueError(
            f"Could not find NLI label: "
            f"{target_label}. "
            f"Available labels: "
            f"{self.label_mapping}"
        )

    # =========================================================
    # CHECK CLAIM
    # =========================================================

    def check_claim(
        self,
        claim: str,
        evidence: str,
        threshold: float = DEFAULT_THRESHOLD,
        margin: float = DEFAULT_MARGIN,
    ) -> SemanticEvidenceResult:

        # =====================================================
        # INPUT VALIDATION
        # =====================================================

        if not claim or not claim.strip():

            return SemanticEvidenceResult(
                supported=False,
                score=0.0,
                label="INVALID",
                reason="Claim is empty.",
            )

        if not evidence or not evidence.strip():

            return SemanticEvidenceResult(
                supported=False,
                score=0.0,
                label="NO_EVIDENCE",
                reason="No evidence was provided.",
            )

        claim = claim.strip()
        evidence = evidence.strip()

        # =====================================================
        # NLI MODEL
        # =====================================================

        try:

            scores = self.model.predict(
                [
                    (
                        evidence,
                        claim,
                    )
                ],
                batch_size=1,
                show_progress_bar=False,
                apply_softmax=True,
            )

        except Exception as error:

            return SemanticEvidenceResult(
                supported=False,
                score=0.0,
                label="ERROR",
                reason=(
                    "Semantic evidence verification "
                    f"failed: {error}"
                ),
            )

        # =====================================================
        # CONVERT SCORES
        # =====================================================

        probabilities = torch.tensor(
            scores[0],
            dtype=torch.float32,
        )

        entailment_probability = float(
            probabilities[
                self.entailment_index
            ]
        )

        contradiction_probability = float(
            probabilities[
                self.contradiction_index
            ]
        )

        neutral_probability = float(
            probabilities[
                self.neutral_index
            ]
        )

        # =====================================================
        # FIND STRONGEST CLASS
        # =====================================================

        class_scores = {
            "ENTAILMENT": entailment_probability,
            "CONTRADICTION": contradiction_probability,
            "NEUTRAL": neutral_probability,
        }

        strongest_label = max(
            class_scores,
            key=class_scores.get,
        )

        strongest_score = class_scores[
            strongest_label
        ]

        # =====================================================
        # SECOND STRONGEST SCORE
        # =====================================================

        sorted_scores = sorted(
            class_scores.values(),
            reverse=True,
        )

        second_best_score = (
            sorted_scores[1]
            if len(sorted_scores) > 1
            else 0.0
        )

        confidence_margin = (
            strongest_score
            - second_best_score
        )

        # =====================================================
        # CONTRADICTION
        # =====================================================

        if (
            strongest_label == "CONTRADICTION"
            and contradiction_probability >= threshold
        ):

            return SemanticEvidenceResult(
                supported=False,
                score=contradiction_probability,
                label="CONTRADICTION",
                reason=(
                    "The evidence semantically "
                    "contradicts the claim."
                ),
                entailment_score=(
                    entailment_probability
                ),
                contradiction_score=(
                    contradiction_probability
                ),
                neutral_score=(
                    neutral_probability
                ),
            )

        # =====================================================
        # STRONG ENTAILMENT
        # =====================================================

        if (
            strongest_label == "ENTAILMENT"
            and entailment_probability >= threshold
            and confidence_margin >= margin
        ):

            return SemanticEvidenceResult(
                supported=True,
                score=entailment_probability,
                label="ENTAILMENT",
                reason=(
                    "The evidence semantically "
                    "supports the claim."
                ),
                entailment_score=(
                    entailment_probability
                ),
                contradiction_score=(
                    contradiction_probability
                ),
                neutral_score=(
                    neutral_probability
                ),
            )

        # =====================================================
        # NEUTRAL / INSUFFICIENT SUPPORT
        # =====================================================

        return SemanticEvidenceResult(
            supported=False,

            # Report entailment probability as the support
            # score instead of taking the maximum class.
            score=entailment_probability,

            label="NEUTRAL",

            reason=(
                "The evidence does not provide "
                "sufficient semantic support for "
                "the claim."
            ),

            entailment_score=(
                entailment_probability
            ),

            contradiction_score=(
                contradiction_probability
            ),

            neutral_score=(
                neutral_probability
            ),
        )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print(
        "\n============================================"
    )

    print(
        "     SEMANTIC EVIDENCE VERIFICATION TEST"
    )

    print(
        "============================================"
    )

    checker = SemanticEvidenceChecker()

    # =====================================================
    # TEST EVIDENCE
    # =====================================================

    evidence = """
    The research was conducted using Python and
    the PyTorch deep learning framework.

    The implementation was developed and executed
    in Visual Studio Code on a Windows-based system.

    The primary software components used in this
    research include Python 3.x, PyTorch, Torchvision,
    Matplotlib, NumPy, and Scikit-learn.
    """

    # =====================================================
    # TEST CASES
    # =====================================================

    tests = [

        (
            "The research used Python and PyTorch.",
            "SUPPORTED",
        ),

        (
            "The implementation used Visual Studio Code.",
            "SUPPORTED",
        ),

        (
            "The research used PyTorch and TensorFlow.",
            "PARTIALLY UNSUPPORTED",
        ),

        (
            "The research used Microsoft Azure.",
            "UNSUPPORTED",
        ),

        (
            "The research used blockchain technology.",
            "UNSUPPORTED",
        ),

        (
            "The primary software components included "
            "Python, PyTorch, Torchvision, Matplotlib, "
            "NumPy, and Scikit-learn.",
            "SUPPORTED",
        ),

        (
            "The research was developed using PyCharm.",
            "UNSUPPORTED",
        ),

        (
            "The research used EVIDENCE as a software tool.",
            "UNSUPPORTED",
        ),
    ]

    # =====================================================
    # RUN TESTS
    # =====================================================

    for claim, description in tests:

        print(
            "\n--------------------------------------------"
        )

        print(
            f"Test: {description}"
        )

        print(
            f"Claim: {claim}"
        )

        result = checker.check_claim(
            claim=claim,
            evidence=evidence,
        )

        print(
            f"Label: {result.label}"
        )

        print(
            f"Supported: {result.supported}"
        )

        print(
            f"Score: {result.score:.4f}"
        )

        print(
            f"Entailment: "
            f"{result.entailment_score:.4f}"
        )

        print(
            f"Contradiction: "
            f"{result.contradiction_score:.4f}"
        )

        print(
            f"Neutral: "
            f"{result.neutral_score:.4f}"
        )

        print(
            f"Reason: {result.reason}"
        )

    print(
        "\n============================================"
    )

    print(
        "          SEMANTIC TEST COMPLETE"
    )

    print(
        "============================================\n"
    )