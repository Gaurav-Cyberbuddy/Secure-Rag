from backend.api.main import VECTOR_DB
from backend.retrieval.retriever import retrieve_documents
from backend.retrieval.reranker import rerank_with_floor
from backend.rag.secure_pipeline import build_context, TOP_K_RETRIEVAL, TOP_K_RERANKED
from backend.rag.generator import (
    focus_context_for_question, generate_answer, MAX_FULL_CONTEXT_CHARS,
)

QUERY  = "What future features can the framework be expanded to support?"
FILE   = "Internship report.pdf.pdf"
TARGET = "39"
PROBE  = ["osint", "widget"]      # words from the answer sentence (diagnostic only)

cands = retrieve_documents(query=QUERY, persist_directory=str(VECTOR_DB),
                           top_k=TOP_K_RETRIEVAL, filename=FILE)
reranked = rerank_with_floor(query=QUERY, results=cands,
                             top_k=TOP_K_RERANKED, floor=2)

print("reranked chunks:")
for i, (d, s) in enumerate(reranked, 1):
    print(f"  {i}. chunk {d.metadata.get('chunk_id')}  chars={len(d.page_content)}")

context = build_context(reranked)
focused = focus_context_for_question(query=QUERY, context=context)

print(f"\ncontext chars = {len(context)}   MAX_FULL_CONTEXT_CHARS = {MAX_FULL_CONTEXT_CHARS}")
print(f"focus bypassed = {len(context) <= MAX_FULL_CONTEXT_CHARS}   focused chars = {len(focused)}")
for w in PROBE:
    print(f"'{w}': in context={w in context.lower()}  in focused={w in focused.lower()}")

target = next((d for d, _ in reranked if str(d.metadata.get('chunk_id')) == TARGET), None)
print("\nGenerator on FOCUSED context:\n ", generate_answer(QUERY, context))
if target:
    print("\nGenerator on chunk 39 ALONE:\n ", generate_answer(QUERY, target.page_content))