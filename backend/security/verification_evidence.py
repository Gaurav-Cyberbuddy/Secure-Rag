"""
Verification evidence selection for Secure AskRAG.

Lexical overlap is used ONLY to locate candidate evidence for NLI.
It never approves a claim.

Critical rule:
    Contiguous structured lists are atomic. If a claim refers to
    items from a list, the verifier receives the COMPLETE list
    (plus its introductory sentence), never a truncated subset.
"""

from __future__ import annotations
import math
import re


# =========================================================
# CONFIGURATION
# =========================================================

# Maximum number of COMPLETE evidence blocks sent to NLI first.
# Blocks are paragraphs / list groups, not individual bullets.
MAX_EVIDENCE_BLOCKS = 2


# =========================================================
# REGEX
# =========================================================

_BULLET_RE = re.compile(
    r"^\s*(?:[-*•]|\d+[.)])\s+(\S.*)$"
)

_INTRO_RE = re.compile(
    r"(?i)\b(include|includes|including|following|"
    r"components|tools|software)\b|:\s*$"
)


_SPLIT_CAPITAL_RE = re.compile(r"\b([B-HJ-Z])[ \t]+([a-z]{3,})\b")
_NO_JOIN = {"and", "the", "for", "not", "but", "are", "was", "has", "had", "can",
            "may", "all", "any", "its", "our", "who", "how", "why", "you", "one", "two"}


def repair_pdf_artifacts(text: str) -> str:
    """'V isual' -> 'Visual'. Whitespace repair only; never uses the claim."""

    def _join(m):
        return m.group(0) if m.group(2) in _NO_JOIN else m.group(1) + m.group(2)

    text = _SPLIT_CAPITAL_RE.sub(_join, text)
    return re.sub(r"[ \t]{2,}", " ", text)


# =========================================================
# DOCUMENT NORMALIZATION
# =========================================================
# =========================================================

def normalize_document_text(text: str) -> str:
    """
    Repair common PDF extraction artifacts before evidence use.

    - Join hyphenated line wraps:
      'Windows-\nbased' -> 'Windows-based'

    - Join soft-wrapped prose lines into sentences.
    """

    if not text:
        return ""
    text = repair_pdf_artifacts(text)

    paragraphs = re.split(
        r"\n\s*\n",
        text,
    )

    repaired: list[str] = []

    for paragraph in paragraphs:

        lines = [
            line.strip()
            for line in paragraph.splitlines()
            if line.strip()
        ]

        if not lines:
            continue

        merged: list[str] = []

        buffer = lines[0]

        for line in lines[1:]:

            # -------------------------------------------------
            # Hyphenated line wrap
            # -------------------------------------------------

            if (
                buffer.endswith("-")
                and line
                and line[0].islower()
            ):
                buffer = buffer + line
                continue

            bullet_next = bool(
                _BULLET_RE.match(line)
            )

            bare_next = _is_bare_list_item(line)

            bullet_buf = bool(
                _BULLET_RE.match(buffer)
            )

            bare_buf = _is_bare_list_item(buffer)

            # -------------------------------------------------
            # Keep list items on separate lines
            # -------------------------------------------------

            if (
                bullet_next
                or bare_next
                or bullet_buf
                or bare_buf
            ):
                merged.append(buffer)
                buffer = line
                continue

            # -------------------------------------------------
            # Join soft-wrapped prose
            # -------------------------------------------------

            if not re.search(
                r"[.!?:]$",
                buffer,
            ):
                buffer = f"{buffer} {line}"
                continue

            merged.append(buffer)
            buffer = line

        merged.append(buffer)

        repaired.append(
            "\n".join(merged)
        )

    return "\n\n".join(repaired)


# =========================================================
# WORD NORMALIZATION
# =========================================================

def normalize_words(text: str) -> set[str]:
    """
    Tokenize text for evidence selection only.
    """

    return set(
        re.findall(
            r"\b[a-zA-Z0-9][a-zA-Z0-9.+#-]*\b",
            text.lower(),
        )
    )


# =========================================================
# SEPARATORS
# =========================================================

def _is_separator(line: str) -> bool:
    return line.strip().startswith("---")


# =========================================================
# BULLET DETECTION
# =========================================================

def _is_bullet_item(line: str) -> bool:
    return bool(
        _BULLET_RE.match(line)
    )


# =========================================================
# BARE LIST DETECTION
# =========================================================

def _is_bare_list_item(line: str) -> bool:
    """
    Detect short standalone list items without bullet markers.

    Important:
    Normal prose must NOT be treated as a list item.

    Examples of valid bare list items:

        PyTorch
        NumPy
        Matplotlib
        Scikit-learn

    Normal prose such as:

        It provides a secure and scalable repository

    must remain normal prose.
    """

    text = line.strip()

    if not text or _is_separator(text):
        return False

    # ---------------------------------------------------------
    # Explicit bullets are always list items.
    # ---------------------------------------------------------

    if _is_bullet_item(text):
        return True

    # ---------------------------------------------------------
    # Sentences ending with punctuation are normal prose.
    # ---------------------------------------------------------

    if re.search(
        r"[.!?;,:]$",
        text,
    ):
        return False

    words = text.split()

    # ---------------------------------------------------------
    # Bare list items should be short.
    # ---------------------------------------------------------

    if not (
        1 <= len(words) <= 5
    ):
        return False

    # ---------------------------------------------------------
    # Do not treat numbered headings as list items.
    # ---------------------------------------------------------

    if re.match(
        r"^\d+(\.\d+)*\s+\S+",
        text,
    ):
        return False

    # ---------------------------------------------------------
    # Do not treat headings ending with ':' as list items.
    # ---------------------------------------------------------

    if text.endswith(":"):
        return False

    # ---------------------------------------------------------
    # Common sentence starters strongly indicate prose.
    # ---------------------------------------------------------

    first_word = words[0].lower()

    prose_starters = {
        "a",
        "an",
        "the",
        "it",
        "this",
        "that",
        "these",
        "those",
        "which",
        "where",
        "when",
        "while",
        "and",
        "or",
        "but",
        "for",
        "with",
        "from",
        "to",
        "in",
        "on",
        "as",
        "by",
    }

    if first_word in prose_starters:
        return False

    return True


# =========================================================
# LIST INTRODUCTION
# =========================================================

def _looks_like_list_intro(line: str) -> bool:

    text = line.strip()

    if not text:
        return False

    if text.endswith(":"):
        return True

    return bool(
        _INTRO_RE.search(text)
    )


# =========================================================
# BULLET STRIPPING
# =========================================================

def _strip_bullet(line: str) -> str:

    match = _BULLET_RE.match(line)

    if match:
        return match.group(1).strip()

    return line.strip()


# =========================================================
# GROUP EVIDENCE BLOCKS
# =========================================================

def group_evidence_blocks(
    evidence: str,
) -> list[str]:
    """
    Split evidence into verification blocks.

    Contiguous list items are kept together with their
    introductory sentence as a single atomic block.
    """

    if not evidence or not evidence.strip():
        return []

    evidence = normalize_document_text(
        evidence
    )

    lines: list[str] = []

    for line in evidence.splitlines():

        stripped = line.strip()

        if (
            not stripped
            or _is_separator(stripped)
        ):
            continue

        lines.append(stripped)

    if not lines:
        return []

    n = len(lines)

    list_flags = [False] * n

    # ---------------------------------------------------------
    # Detect list items
    # ---------------------------------------------------------

    for index, line in enumerate(lines):

        if _is_bullet_item(line):
            list_flags[index] = True
            continue

        if not _is_bare_list_item(line):
            continue

        prev_list = (
            index > 0
            and (
                list_flags[index - 1]
                or _is_bullet_item(
                    lines[index - 1]
                )
                or _is_bare_list_item(
                    lines[index - 1]
                )
            )
        )

        next_list = (
            index + 1 < n
            and (
                _is_bullet_item(
                    lines[index + 1]
                )
                or _is_bare_list_item(
                    lines[index + 1]
                )
            )
        )

        prev_intro = (
            index > 0
            and _looks_like_list_intro(
                lines[index - 1]
            )
        )

        if (
            prev_list
            or next_list
            or prev_intro
        ):
            list_flags[index] = True

    # ---------------------------------------------------------
    # Build blocks
    # ---------------------------------------------------------

    blocks: list[str] = []

    index = 0

    while index < n:

        if not list_flags[index]:

            blocks.append(
                lines[index]
            )

            index += 1
            continue

        start = index

        if (
            start > 0
            and not list_flags[start - 1]
            and _looks_like_list_intro(
                lines[start - 1]
            )
        ):

            if (
                blocks
                and blocks[-1]
                == lines[start - 1]
            ):
                blocks.pop()

            start -= 1

        end = index

        while (
            end + 1 < n
            and list_flags[end + 1]
        ):
            end += 1

        blocks.append(
            "\n".join(
                lines[start:end + 1]
            )
        )

        index = end + 1

    return [
        block
        for block in blocks
        if block.strip()
    ]


# =========================================================
# FORMAT BLOCK FOR NLI
# =========================================================

def format_block_for_nli(
    block: str,
) -> str:
    """
    Present a block in continuous text for NLI.

    Structured lists are joined into one sentence so the
    NLI model sees the full set of items as a single premise.

    No list item is dropped.
    """

    lines = [
        line.strip()
        for line in block.splitlines()
        if line.strip()
    ]

    if not lines:
        return ""

    intro_parts: list[str] = []

    items: list[str] = []

    trailing: list[str] = []

    mode = "intro"

    for line in lines:

        bullet = _is_bullet_item(line)

        bare = _is_bare_list_item(line)

        if mode == "intro":

            if bullet:

                mode = "items"

                items.append(
                    _strip_bullet(line)
                )

            elif (
                bare
                and intro_parts
                and _looks_like_list_intro(
                    intro_parts[-1]
                )
            ):

                mode = "items"

                items.append(
                    _strip_bullet(line)
                )

            elif (
                bare
                and not intro_parts
            ):

                mode = "items"

                items.append(
                    _strip_bullet(line)
                )

            else:

                intro_parts.append(line)

        elif mode == "items":

            if bullet or bare:

                items.append(
                    _strip_bullet(line)
                )

            else:

                mode = "trailing"

                trailing.append(line)

        else:

            trailing.append(line)

    # ---------------------------------------------------------
    # No list detected: preserve normal prose.
    # ---------------------------------------------------------

    if not items:
        return " ".join(lines)

    intro = " ".join(
        intro_parts
    ).strip()

    if (
        intro
        and not intro.endswith(
            (":", ",", ";")
        )
    ):
        intro = (
            intro.rstrip(".")
            + ":"
        )

    item_text = ", ".join(items)

    premise = (
        f"{intro} {item_text}."
        if intro
        else f"{item_text}."
    )

    if trailing:

        premise = (
            premise
            + " "
            + " ".join(trailing)
        )

    return premise


# =========================================================
# SCORE BLOCK FOR CLAIM
# =========================================================

def score_block_for_claim(
    claim: str,
    block: str,
) -> float:
    """
    Lexical relevance of a block to a claim.

    This is used for evidence selection only.
    It does NOT approve the claim.
    """

    claim_words = normalize_words(
        claim
    )

    block_words = normalize_words(
        block
    )

    if (
        not claim_words
        or not block_words
    ):
        return 0.0

    overlap = (
        claim_words.intersection(
            block_words
        )
    )

    return (
        len(overlap)
        / len(claim_words)
    )


# =========================================================
# SELECT VERIFICATION EVIDENCE
# =========================================================

def select_verification_evidence(
    claim: str,
    evidence: str,
    max_blocks: int = MAX_EVIDENCE_BLOCKS,
) -> str:
    """
    Select complete evidence blocks needed to verify a claim.

    Structured lists are never partially selected.
    """

    if (
        not claim.strip()
        or not evidence.strip()
    ):
        return ""

    blocks = group_evidence_blocks(
        evidence
    )

    if not blocks:
        return ""

    scored: list[
        tuple[float, int, str]
    ] = []

    for index, block in enumerate(
        blocks
    ):

        score = score_block_for_claim(
            claim,
            block,
        )

        if score <= 0:
            continue

        scored.append(
            (
                score,
                index,
                block,
            )
        )

    if not scored:
        return ""

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    selected = scored[
        : max(1, max_blocks)
    ]

    # Keep nearby preceding context
    # for selected list blocks.

    selected_indexes = {
        index
        for _, index, _
        in selected
    }

    for _, index, block in selected:

        if "\n" not in block:
            continue

        for offset in (1, 2):

            prev_index = (
                index - offset
            )

            if prev_index >= 0:

                selected_indexes.add(
                    prev_index
                )

    ordered = sorted(
        selected_indexes
    )

    formatted = [
        format_block_for_nli(
            blocks[index]
        )
        for index in ordered
        if (
            0 <= index < len(blocks)
        )
    ]

    return "\n".join(
        part
        for part in formatted
        if part.strip()
    )


# =========================================================
# SPLIT EVIDENCE CHUNKS
# =========================================================

def split_evidence_chunks(
    evidence: str,
) -> list[str]:
    """
    Split combined pipeline evidence
    into original chunks.
    """

    chunks = evidence.split(
        "\n\n--- EVIDENCE ---\n\n"
    )

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


# =========================================================
# BUILD NLI CANDIDATES
# =========================================================
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def select_sentence_windows(claim: str, evidence: str, top_n: int = 2) -> list[str]:
    """Short 1-2 sentence premises. Used only to LOCATE evidence for NLI."""
    text = normalize_document_text(evidence.replace("--- EVIDENCE ---", " "))
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(" ".join(text.split()))
                 if len(s.split()) >= 4]
    if not sentences:
        return []
    df: dict[str, int] = {}
    for s in sentences:
        for w in normalize_words(s):
            df[w] = df.get(w, 0) + 1
    n, cw, scored = len(sentences), normalize_words(claim), []
    for i, s in enumerate(sentences):
        common = cw & normalize_words(s)
        if common:
            scored.append((sum(math.log((n + 1) / df[w]) + 1.0 for w in common), i))
    scored.sort(reverse=True)
    windows = []
    for _, i in scored[:top_n]:
        windows.append(sentences[i])
        if i + 1 < n:
            windows.append(sentences[i] + " " + sentences[i + 1])
    return windows
def build_nli_candidates(
    claim: str,
    evidence: str,
) -> list[str]:
    """
    Build ordered NLI premise candidates for a claim.

    Order:
        1. Full permitted evidence
        2. List-preserving targeted evidence blocks
        3. Original evidence chunks / list-aware chunk views
    """

    candidates: list[str] = []

    seen: set[str] = set()

    def _add(text: str) -> None:

        normalized = text.strip()

        if not normalized:
            return

        key = normalized.lower()

        if key in seen:
            return

        seen.add(key)

        candidates.append(
            normalized
        )

    # ---------------------------------------------------------
    # FULL PERMITTED EVIDENCE
    # ---------------------------------------------------------
    # Multi-hop claims may combine facts from several chunks.
    # Give NLI the complete permitted evidence first.
    #
    # This does NOT bypass disclosure control: `evidence` is
    # already the sanitized/permitted evidence supplied by the
    # secure pipeline.
    # ---------------------------------------------------------

    repaired_full = normalize_document_text(
        evidence
    )

    if repaired_full:
        blocks = group_evidence_blocks(
            repaired_full
        )

        if blocks:
            full_evidence = "\n".join(
                format_block_for_nli(
                    block
                )
                for block in blocks
            )
            _add(full_evidence)
        else:
            _add(repaired_full)

    # ---------------------------------------------------------
    # Targeted evidence
    # ---------------------------------------------------------

    targeted = select_verification_evidence(
        claim=claim,
        evidence=evidence,
    )

    if targeted:
        _add(targeted)
        for window in select_sentence_windows(claim, evidence):
            _add(window)

    # ---------------------------------------------------------
    # Original evidence chunks
    # ---------------------------------------------------------

    for chunk in split_evidence_chunks(
        evidence
    ):

        repaired = normalize_document_text(
            chunk
        )

        _add(repaired)

        blocks = group_evidence_blocks(
            repaired
        )

        if blocks:

            joined = "\n".join(
                format_block_for_nli(
                    block
                )
                for block in blocks
            )

            _add(joined)

    return candidates