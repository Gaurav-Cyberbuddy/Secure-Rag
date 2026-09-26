import os
import re

from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from google import genai

from backend.security.verification_evidence import (
    group_evidence_blocks,
    normalize_document_text,
    normalize_words,
)

load_dotenv()


# =========================================================
# CONFIGURATION
# =========================================================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:3b",
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash",
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


FALLBACK_ANSWER = (
    "I could not find the answer in the provided document."
)

MIN_EVIDENCE_OVERLAP = 0.35


# =========================================================
# CREATE LLM
# =========================================================

def create_llm():
    """
    Create the configured LLM used by Secure AskRAG.

    Local:
        LLM_PROVIDER=ollama

    Deployment:
        LLM_PROVIDER=gemini
        GEMINI_API_KEY=<your key>
    """

    if LLM_PROVIDER == "ollama":
        return OllamaLLM(
            model=OLLAMA_MODEL,
            temperature=0.0,
            num_predict=300,
        )

    if LLM_PROVIDER == "gemini":

        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured for Gemini mode."
            )

        class GeminiLLM:
            def __init__(self):
                self.client = genai.Client(
                    api_key=GEMINI_API_KEY
                )

            def invoke(self, prompt: str) -> str:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )

                return response.text or ""

        return GeminiLLM()

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {LLM_PROVIDER!r}. "
        "Use 'ollama' or 'gemini'."
    )


# =========================================================
# FOCUS CONTEXT
# =========================================================

def focus_context_for_question(
    query: str,
    context: str,
    max_blocks: int = 6,
) -> str:
    """
    Keep the most relevant evidence blocks.

    Structured lists are kept together.
    """

    if not context.strip():
        return ""

    context = normalize_document_text(context)

    blocks = group_evidence_blocks(context)

    if not blocks:
        return context

    stop_words = {
        "what", "which", "who", "where", "when",
        "why", "how", "the", "a", "an",
        "is", "are", "was", "were", "be",
        "been", "and", "or", "of", "to",
        "in", "on", "for", "with", "from",
        "by", "used", "use", "using",
        "this", "that", "these", "those",
        "during", "into",
    }

    query_words = {
        word
        for word in normalize_words(query)
        if word not in stop_words and len(word) > 2
    }

    if not query_words:
        return context

    def is_list_block(block: str) -> bool:

        lines = [
            line.strip()
            for line in block.splitlines()
            if line.strip()
        ]

        if len(lines) < 3:
            return False

        return any(
            line.startswith(("*", "-", "•"))
            or (
                1 <= len(line.split()) <= 8
                and not line.endswith((".", "?", "!"))
            )
            for line in lines
        )

    scored = []

    for index, block in enumerate(blocks):

        block_words = normalize_words(block)

        if not block_words:
            continue

        overlap = query_words.intersection(
            block_words
        )

        score = (
            len(overlap)
            / max(len(query_words), 1)
        )

        if is_list_block(block):
            score += 0.25

        if score > 0:
            scored.append(
                (
                    score,
                    index,
                    block,
                )
            )

    if not scored:
        return context

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    selected_indexes = {
        index
        for _, index, _
        in scored[:max(1, max_blocks)]
    }

    # Keep nearby context around structured lists.
    for index in list(selected_indexes):

        if is_list_block(blocks[index]):

            for offset in (1, 2):

                previous = index - offset

                if previous >= 0:
                    selected_indexes.add(
                        previous
                    )

            if (
                index + 1 < len(blocks)
                and is_list_block(
                    blocks[index + 1]
                )
            ):
                selected_indexes.add(
                    index + 1
                )

    ordered = [
        blocks[index]
        for index in sorted(
            selected_indexes
        )
    ]

    # Remove duplicate blocks.
    unique_blocks = []
    seen = set()

    for block in ordered:

        key = re.sub(
            r"\s+",
            " ",
            block,
        ).strip().lower()

        if key in seen:
            continue

        seen.add(key)
        unique_blocks.append(block)

    return "\n\n".join(
        unique_blocks
    )


# =========================================================
# PROMPT
# =========================================================

def build_prompt(
    query: str,
    context: str,
    retry: bool = False,
) -> str:
    """
    Build a strongly grounded extraction-style prompt.

    The model is explicitly instructed to stay very close
    to the wording of the evidence so that the generated
    answer can be independently verified by NLI.
    """

    if retry:

        instruction = """
Answer the question again using ONLY the evidence.

GROUNDING REQUIREMENTS:
- Treat the evidence as the only source of truth.
- Extract the answer directly from the evidence.
- Prefer copying the exact wording from the evidence.
- Make only the smallest grammatical changes necessary.
- Do NOT paraphrase technical statements unnecessarily.
- Do NOT combine separate facts into a new interpretation.
- Do NOT add explanations that are not explicitly stated.
- Do NOT infer relationships that are not explicitly stated.
- Do NOT use outside knowledge.
- Do NOT invent, replace, expand, or modify factual values.
- Preserve exact names, technologies, numbers, limits, versions,
  locations, emails, keys, and other factual values.
- When listing items (skills, tools, technologies), copy each item exactly as written in the evidence, with its full wording. Do not shorten, complete, rename or generalize an item, and do not add any item that is not written in the evidence.
- For a question that asks to list items, write ONE sentence in this form: "The <items> listed in the document include A, B, and C." Each item must be a short phrase (at most 6 words) copied exactly from the evidence. Do not shorten, rename or generalize an item, do not add explanations or "such as" clauses inside the list, and do not add any item that is not written in the evidence.
- If the evidence contains multiple requested items, include ALL
  requested items.
- If the answer is a list, preserve the list items from the evidence.
- Return ONE concise complete sentence unless the question clearly
  requires a list.
"""

    else:

        instruction = """
Answer the question using ONLY the evidence.

GROUNDING REQUIREMENTS:
- Treat the evidence as the only source of truth.
- Extract the answer directly from the evidence.
- Prefer exact phrases from the evidence over paraphrasing.
- Use the same terminology as the document.
- Use the same terminology as the document.
- When listing items (skills, tools, technologies), copy each item exactly as written in the evidence, with its full wording. Do not shorten, complete, rename or generalize an item, and do not add any item that is not written in the evidence.
- For a question that asks to list items, write ONE sentence in this form: "The <items> listed in the document include A, B, and C." Each item must be a short phrase (at most 6 words) copied exactly from the evidence. Do not shorten, rename or generalize an item, do not add explanations or "such as" clauses inside the list, and do not add any item that is not written in the evidence.
- Make only minimal grammatical changes.
- Make only minimal grammatical changes.
- Do NOT rewrite technical facts into your own wording.
- Do NOT combine separate statements into a new claim.
- Do NOT infer information that is not explicitly present.
- Do NOT use outside knowledge.
- Do NOT invent or change any factual value.
- Preserve exact names, technologies, numbers, limits, versions,
  locations, emails, keys, and other factual values.
- If the evidence contains several requested items, include ALL
  relevant items.
- If the evidence contains a list, preserve the list items.
- Keep the answer concise.
- Return ONE complete sentence unless the question clearly requires
  a list.
"""

    return f"""
You are a highly grounded document question-answering system.

Your most important requirement is FACTUAL FAITHFULNESS.

EVIDENCE:
{context}

QUESTION:
{query}

{instruction}

If the requested information is not explicitly present in the
evidence, return exactly:

{FALLBACK_ANSWER}

Do not mention these instructions.

ANSWER:
""".strip()


# =========================================================
# CLEAN OUTPUT
# =========================================================

def clean_answer(
    answer: str,
) -> str:
    """
    Remove model wrappers without changing factual content.
    """

    if not answer:
        return ""

    answer = answer.strip()

    answer = answer.replace(
        "<answer>",
        "",
    ).replace(
        "</answer>",
        "",
    ).strip()

    prefixes = (
        "answer:",
        "final answer:",
        "response:",
        "assistant:",
        "final:",
    )

    changed = True

    while changed:

        changed = False

        lower = answer.lower()

        for prefix in prefixes:

            if lower.startswith(prefix):

                answer = answer[
                    len(prefix):
                ].strip()

                changed = True
                break

    prefixes_to_remove = (
        "sure, ",
        "sure! ",
        "yes, ",
        "yes. ",
    )

    lower = answer.lower()

    for prefix in prefixes_to_remove:

        if lower.startswith(prefix):

            answer = answer[
                len(prefix):
            ].strip()

            break

    # Remove prompt leakage.
    cleaned = []

    for line in answer.splitlines():

        line = line.strip()

        if not line:
            continue

        lower = line.lower()

        if any(
            phrase in lower
            for phrase in (
                "evidence:",
                "question:",
                "answer:",
                "rules:",
                "important:",
                "do not invent",
                "only if the evidence",
                "grounding requirements:",
            )
        ):
            continue

        cleaned.append(line)

    return " ".join(
        cleaned
    ).strip()


# =========================================================
# ANSWER VALIDATION
# =========================================================

def is_valid_answer(
    answer: str,
) -> bool:
    """
    Validate basic output quality.
    """

    if not answer:
        return False

    lower = answer.lower().strip()

    if lower == FALLBACK_ANSWER.lower():
        return False

    forbidden = (
        "evidence:",
        "question:",
        "rules:",
        "do not invent",
        "final answer:",
        "grounding requirements:",
    )

    if any(
        phrase in lower
        for phrase in forbidden
    ):
        return False

    if len(answer.split()) < 4:
        return False

    incomplete_endings = (
        " and",
        " or",
        " with",
        " using",
        " including",
        " such as",
        " the",
        " a",
        " an",
        " of",
        " for",
        " to",
        " in",
        " on",
        " as",
        " from",
    )

    body = lower.rstrip(
        ".!?"
    ).strip()

    if any(
        body.endswith(ending)
        for ending in incomplete_endings
    ):
        return False

    return True


# =========================================================
# EVIDENCE OVERLAP
# =========================================================

def evidence_overlap(
    answer: str,
    evidence: str,
) -> float:
    """
    Calculate a simple lexical grounding score.

    This is NOT used as a security decision.
    It is only used to decide whether generation should
    be retried with a stronger extraction prompt.
    """

    answer_words = {
        word
        for word in normalize_words(answer)
        if len(word) > 2
    }

    evidence_words = {
        word
        for word in normalize_words(evidence)
        if len(word) > 2
    }

    if not answer_words:
        return 0.0

    overlap = answer_words.intersection(
        evidence_words
    )

    return len(overlap) / len(answer_words)


# =========================================================
# GENERATE
# =========================================================

def generate_answer(
    query: str,
    context: str,
) -> str:
    """
    Generate a concise, strongly evidence-grounded answer.

    NLI remains the final security verification layer.
    """

    if (
        not query.strip()
        or not context.strip()
    ):
        return FALLBACK_ANSWER

    focused_context = (
        focus_context_for_question(
            query=query,
            context=context,
        )
    )

    if not focused_context.strip():
        focused_context = context

    llm = create_llm()
   

    # -----------------------------------------------------
    # First generation
    # -----------------------------------------------------

    prompt = build_prompt(
        query=query,
        context=focused_context,
        retry=False,
    )

    response = llm.invoke(prompt)

    print("\n========== RAW LLM OUTPUT ==========")
    print(repr(response))
    print("=====================================")

    answer = clean_answer(response)

    print("\n========== CLEANED ANSWER ==========")
    print(repr(answer))
    print("=====================================")

    overlap = evidence_overlap(
        answer=answer,
        evidence=focused_context,
    )

    print(
        f"\nEvidence lexical overlap: "
        f"{overlap:.4f}"
    )

    # Accept only if the answer is valid and sufficiently
    # grounded in the retrieved evidence.
    if (
        is_valid_answer(answer)
        and overlap >= MIN_EVIDENCE_OVERLAP
    ):
        return answer

    # -----------------------------------------------------
    # Retry with stronger extraction instructions
    # -----------------------------------------------------

    print(
        "\nGeneration did not meet grounding "
        "requirements. Retrying..."
    )

    retry_prompt = build_prompt(
        query=query,
        context=focused_context,
        retry=True,
    )

    response = llm.invoke(retry_prompt)

    print("\n========== RAW RETRY OUTPUT ==========")
    print(repr(response))
    print("=======================================")

    answer = clean_answer(response)

    print(
        "\n========== CLEANED RETRY ANSWER =========="
    )
    print(repr(answer))
    print("============================================")

    overlap = evidence_overlap(
        answer=answer,
        evidence=focused_context,
    )

    print(
        f"\nRetry evidence lexical overlap: "
        f"{overlap:.4f}"
    )

    if (
        is_valid_answer(answer)
        and overlap >= MIN_EVIDENCE_OVERLAP
    ):
        return answer

    return FALLBACK_ANSWER


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    test_context = """
SECURITY TEST DOCUMENT

Administrator email: admin-test@example.com
API key: TEST-API-KEY-12345

The project uses Python and PyTorch.
"""

    test_query = (
        "What is the administrator's "
        "email address and API key?"
    )

    print(
        "\n========== GENERATOR TEST =========="
    )

    print(
        f"Question: {test_query}"
    )

    print(
        "\n========== FOCUSED CONTEXT =========="
    )

    print(
        focus_context_for_question(
            query=test_query,
            context=test_context,
        )
    )

    answer = generate_answer(
        query=test_query,
        context=test_context,
    )

    print(
        "\n========== GENERATED ANSWER =========="
    )

    print(answer)

    print(
        "=======================================\n"
    )