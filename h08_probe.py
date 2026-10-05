import re
from backend.api.main import VECTOR_DB
from backend.retrieval.retriever import retrieve_documents
from backend.retrieval.reranker import rerank_with_floor
from backend.rag import secure_pipeline as sp, generator as g
from backend.security.verification_evidence import (
    group_evidence_blocks, normalize_document_text, normalize_words,
)

QUERY = "Summarize the key outcomes of the internship."
FILE = "Internship report.pdf.pdf"
PHRASES = ["responsive analytical dashboards", "automated cti pipeline", "stix 2.1"]


def focus_reference(query, context, max_blocks=6):
    """Original lexical focus, copied from the baseline generator.py."""
    if not context.strip():
        return ""
    context = normalize_document_text(context)
    if len(context) <= g.MAX_FULL_CONTEXT_CHARS:
        return context
    blocks = group_evidence_blocks(context)
    if not blocks:
        return context
    stop_words = {
        "what", "which", "who", "where", "when", "why", "how", "the", "a", "an",
        "is", "are", "was", "were", "be", "been", "and", "or", "of", "to", "in",
        "on", "for", "with", "from", "by", "used", "use", "using", "this", "that",
        "these", "those", "during", "into",
    }
    query_words = {
        w for w in normalize_words(query) if w not in stop_words and len(w) > 2
    }
    if not query_words:
        return context

    def is_list_block(block):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            return False
        return any(
            line.startswith(("*", "-", "•"))
            or (1 <= len(line.split()) <= 8 and not line.endswith((".", "?", "!")))
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
    selected = {i for _, i, _ in scored[:max(1, max_blocks)]}
    for index in list(selected):
        if is_list_block(blocks[index]):
            for offset in (1, 2):
                if index - offset >= 0:
                    selected.add(index - offset)
            if index + 1 < len(blocks) and is_list_block(blocks[index + 1]):
                selected.add(index + 1)
    unique, seen = [], set()
    for i in sorted(selected):
        key = re.sub(r"\s+", " ", blocks[i]).strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(blocks[i])
    return "\n\n".join(unique)


cands = retrieve_documents(query=QUERY, persist_directory=str(VECTOR_DB),
                           top_k=sp.TOP_K_RETRIEVAL, filename=FILE)
reranked = rerank_with_floor(query=QUERY, results=cands,
                             top_k=sp.TOP_K_RERANKED, floor=sp.RERANK_FLOOR)
print("reranked chunks:", [d.metadata.get("chunk_id") for d, _ in reranked])

context = sp.build_context(reranked)
cur = g.focus_context_for_question(query=QUERY, context=context)
ref = focus_reference(QUERY, context)

print(f"\nchars: context={len(context)} current={len(cur)} reference={len(ref)}")
print("current == reference:", cur == ref)
for p in PHRASES:
    print(f"{p!r}: context={p in context.lower()} "
          f"current={p in cur.lower()} reference={p in ref.lower()}")

if cur != ref:
    cb, rb = cur.split("\n\n"), ref.split("\n\n")
    print("\nonly in current  :", [b[:80] for b in cb if b not in rb][:3])
    print("only in reference:", [b[:80] for b in rb if b not in cb][:3])

print("\nanswer with current focus  :", g.generate_answer(QUERY, context))
print("answer with reference focus:", g.generate_answer(QUERY, ref))