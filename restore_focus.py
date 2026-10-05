import re
import shutil
import py_compile
from pathlib import Path

path = Path("backend/rag/generator.py")
src = path.read_text(encoding="utf-8")
shutil.copy(path, "backend/rag/generator.py.bak")

NEW_FUNC = r'''def focus_context_for_question(
    query: str,
    context: str,
    max_blocks: int = 6,
) -> str:
    """
    Keep the most relevant evidence blocks (lexical overlap).

    Structured lists are kept together.
    """

    if not context.strip():
        return ""

    context = normalize_document_text(context)

    if len(context) <= MAX_FULL_CONTEXT_CHARS:
        return context

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

        overlap = query_words.intersection(block_words)

        score = len(overlap) / max(len(query_words), 1)

        if is_list_block(block):
            score += 0.25

        if score > 0:
            scored.append((score, index, block))

    if not scored:
        return context

    scored.sort(key=lambda item: item[0], reverse=True)

    selected_indexes = {
        index
        for _, index, _ in scored[:max(1, max_blocks)]
    }

    # Keep nearby context around structured lists.
    for index in list(selected_indexes):

        if is_list_block(blocks[index]):

            for offset in (1, 2):
                previous = index - offset
                if previous >= 0:
                    selected_indexes.add(previous)

            if (
                index + 1 < len(blocks)
                and is_list_block(blocks[index + 1])
            ):
                selected_indexes.add(index + 1)

    ordered = [blocks[i] for i in sorted(selected_indexes)]

    # Remove duplicate blocks.
    unique_blocks = []
    seen = set()

    for block in ordered:
        key = re.sub(r"\s+", " ", block).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique_blocks.append(block)

    return "\n\n".join(unique_blocks)
'''

start = src.index("def focus_context_for_question(")
m = re.search(r"\n# =+\n# PROMPT", src[start:])
if m:
    end = start + m.start()
else:
    end = src.index("def build_prompt(", start)

src = src[:start] + NEW_FUNC.rstrip() + "\n\n\n" + src[end:].lstrip("\n")
src = re.sub(r"(?m)^SEMANTIC_[A-Z_]+\s*=.*\n", "", src)
src = re.sub(
    r"(?m)^[ \t]*from backend\.security\.evidence_relevance import get_model\s*\n",
    "",
    src,
)

path.write_text(src, encoding="utf-8")
py_compile.compile(str(path), doraise=True)

import backend.rag.generator as g

ctx = "\n\n".join(
    f"Sentence number {i} about pipelines and dashboards." for i in range(400)
)
out = g.focus_context_for_question("What dashboards were built?", ctx)
print("RESTORED OK:", len(ctx), "->", len(out))