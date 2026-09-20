from sentence_transformers import SentenceTransformer
import re


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SEMANTIC_THRESHOLD = 0.15


# =========================================================
# MODEL CACHE
# =========================================================

_model = None


def get_model() -> SentenceTransformer:
    """
    Load the embedding model only once and reuse it.
    """

    global _model

    if _model is None:
        print("\n[MODEL] Loading evidence relevance model...")

        _model = SentenceTransformer(
            MODEL_NAME
        )

        print(
            "[MODEL] Evidence relevance model loaded."
        )

    return _model


# =========================================================
# EVIDENCE RELEVANCE CHECKER
# =========================================================

class EvidenceRelevanceChecker:

    def __init__(self):
        self.model = get_model()

    # =====================================================
    # KEYWORD EXTRACTION
    # =====================================================

    def _extract_keywords(
        self,
        text: str,
    ) -> set:

        stop_words = {
            "what",
            "is",
            "are",
            "was",
            "were",
            "the",
            "a",
            "an",
            "of",
            "to",
            "in",
            "for",
            "and",
            "on",
            "during",
            "this",
            "that",
            "who",
            "how",
            "which",
        }

        words = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )

        return {
            word
            for word in words
            if word not in stop_words
            and len(word) > 2
        }

    # =====================================================
    # CHECK RELEVANCE
    # =====================================================

    def check(
        self,
        query: str,
        evidence_chunks: list[str],
    ) -> dict:

        if not evidence_chunks:

            return {
                "relevant": False,
                "score": 0.0,
                "reason": "No evidence was retrieved.",
            }

        query_lower = query.lower()

        evidence_text = " ".join(
            evidence_chunks
        ).lower()

        # =================================================
        # 1. SENSITIVE INFORMATION QUESTIONS
        # =================================================

        sensitive_terms = {
            "password",
            "passwords",
            "credential",
            "credentials",
            "secret",
            "secrets",
            "api key",
            "api keys",
            "access token",
            "private key",
        }

        requested_sensitive_term = None

        for term in sensitive_terms:

            if term in query_lower:

                requested_sensitive_term = term

                break

        if requested_sensitive_term:

            if requested_sensitive_term not in evidence_text:

                return {
                    "relevant": False,
                    "score": 0.0,
                    "reason": (
                        "The evidence does not explicitly "
                        "contain the requested sensitive "
                        "information."
                    ),
                }

        # =================================================
        # 2. SEMANTIC SIMILARITY
        # =================================================

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        chunk_embeddings = self.model.encode(
            evidence_chunks,
            normalize_embeddings=True,
        )

        scores = chunk_embeddings @ query_embedding

        best_index = int(
            scores.argmax()
        )

        semantic_score = float(
            scores[best_index]
        )

        best_chunk = evidence_chunks[
            best_index
        ]

        # =================================================
        # 3. TECHNOLOGY QUESTIONS
        # =================================================

        technology_terms = {
            "technology",
            "technologies",
            "python",
            "stix",
            "kali",
            "software",
            "tool",
            "tools",
            "framework",
            "platform",
            "pipeline",
        }

        technology_question = any(
            term in query_lower
            for term in technology_terms
        )

        if technology_question:

            technology_found = any(
                term in evidence_text
                for term in technology_terms
            )

            if technology_found:

                return {
                    "relevant": True,
                    "score": semantic_score,
                    "reason": (
                        "The evidence contains "
                        "technology-related information "
                        "relevant to the question."
                    ),
                }

        # =================================================
        # 4. KEYWORD OVERLAP
        # =================================================

        query_keywords = self._extract_keywords(
            query
        )

        evidence_keywords = self._extract_keywords(
            best_chunk
        )

        overlap = (
            query_keywords.intersection(
                evidence_keywords
            )
        )

        # =================================================
        # 5. GENERAL RELEVANCE
        # =================================================

        if semantic_score >= SEMANTIC_THRESHOLD:

            return {
                "relevant": True,
                "score": semantic_score,
                "reason": (
                    "Relevant evidence was found "
                    "for the question."
                ),
            }

        if overlap:

            return {
                "relevant": True,
                "score": semantic_score,
                "reason": (
                    "The question shares relevant "
                    "terms with the evidence."
                ),
            }

        return {
            "relevant": False,
            "score": semantic_score,
            "reason": (
                "The retrieved evidence does not "
                "appear relevant to the question."
            ),
        }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    checker = EvidenceRelevanceChecker()

    evidence = [
        """
        The Security Analyst Internship carried out at
        Tata Consultancy Services resulted in a
        comprehensive learning experience.

        The internship involved development of an
        Automated CTI Pipeline using technologies
        such as STIX 2.1, Python, and Kali Linux.
        """
    ]

    tests = [
        "What technologies were used?",
        "What is the password?",
        "What was the purpose of the internship?",
    ]

    print(
        "\n========== RELEVANCE TEST =========="
    )

    for query in tests:

        result = checker.check(
            query=query,
            evidence_chunks=evidence,
        )

        print(
            f"\nQuestion: {query}"
        )

        print(
            f"Relevant: {result['relevant']}"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Reason: {result['reason']}"
        )

    print(
        "\n===================================="
    )