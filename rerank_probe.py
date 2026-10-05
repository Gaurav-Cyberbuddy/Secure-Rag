from backend.api.main import VECTOR_DB
from backend.retrieval.retriever import retrieve_documents
from backend.retrieval.reranker import rerank_documents

QUERY = "What future features can the framework be expanded to support?"
FILE = "Internship report.pdf.pdf"
TARGET = "39"
KEY = "extend functionalities"

cands = retrieve_documents(
    query=QUERY,
    persist_directory=str(VECTOR_DB),
    top_k=20,
    filename=FILE,
)

dense_rank = {
    str(d.metadata.get("chunk_id")): i
    for i, (d, _) in enumerate(cands, 1)
}

reranked = rerank_documents(
    query=QUERY,
    results=cands,
    top_k=20,
)

print("final  chunk  dense_rank  rerank_score  chars  key_offset")

for i, (d, s) in enumerate(reranked, 1):
    cid = str(d.metadata.get("chunk_id"))

    off = (
        d.page_content.find(KEY)
        if cid == TARGET
        else ""
    )

    mark = "   <== TARGET" if cid == TARGET else ""

    print(
        f"{i:5}  {cid:5}  "
        f"{dense_rank.get(cid)!s:10}  "
        f"{s:12.4f}  "
        f"{len(d.page_content):5}  "
        f"{off!s:10}{mark}"
    )
    import copy
from backend.security.verification_evidence import normalize_document_text

target = next(
    d for d, _ in cands
    if str(d.metadata.get("chunk_id")) == TARGET
)

print("\nchars:", len(target.page_content))
print(repr(target.page_content[:600]))
print(repr(target.page_content[-300:]))

clean = copy.copy(target)
clean.page_content = normalize_document_text(target.page_content)

others = [
    c for c in cands
    if c[0] is not target
][:9]

test = rerank_documents(
    query=QUERY,
    results=[(clean, 0.0)] + others,
    top_k=10,
)

print("\nreranked with NORMALIZED chunk 39:")
for i, (d, s) in enumerate(test, 1):
    print(
        i,
        d.metadata.get("chunk_id"),
        f"{s:.4f}"
    )