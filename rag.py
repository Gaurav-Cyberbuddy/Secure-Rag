from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import os
import hashlib
import time
import logging

logging.basicConfig(
    filename="security_logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
def calculate_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        while chunk := f.read(4096):
            sha256.update(chunk)

    return sha256.hexdigest()

  
file_path = "doc/S3 bucket in AWSaa .pdf"
allowed_extensions = [
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".csv"
]

file_extension = os.path.splitext(file_path)[1]
file_path = "doc/S3 bucket in AWSaa.pdf"
file_hash = calculate_file_hash(file_path)
trusted_hashes = [
    "251334bc1ff21404839a6eef15b35614f7694ddeee7cc4ce98358acc69f61a16"
] 
if file_hash not in trusted_hashes:
    print("File integrity verification failed.")
    exit() 
print("\nFile Hash:")
print(file_hash)


if file_extension not in allowed_extensions:
    print("Blocked untrusted file type.")
    exit()
trusted_sources = [
    "official_aws_docs",
    "research_paper",
    "internal_verified_docs"
]

source = input("Enter document source: ")

if source not in trusted_sources:
    print("Untrusted document source blocked.")
    exit()    
# Load PDF
loader = PyPDFLoader("doc/S3 bucket in AWSaa.pdf")
documents = loader.load()

suspicious_patterns = [
    "ignore previous instructions",
    "reveal system prompt",
    "bypass security",
    "developer instructions"
]

full_text = ""

for doc in documents:
    full_text += doc.page_content.lower()

if any(pattern in full_text for pattern in suspicious_patterns):
    print("Suspicious content detected in document.")
    exit()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

# Create embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
db = Chroma.from_documents(
    chunks,
    embedding,
    persist_directory="chroma_db"
)

# Load local model
llm = Ollama(model="tinyllama")

query_count = 0
max_queries = 5

# Chat loop
# Chat loop
while True:

    if query_count >= max_queries:

        logging.warning("Query limit exceeded.")

        print("\nQuery limit exceeded.")
        break
    query = input("\nAsk Question: ")

    # Input Guardrail
    blocked_words = [
        "ignore previous instructions",
        "system prompt",
        "reveal secrets",
        "bypass security",
        "developer instructions"
    ]

    query_lower = query.lower()

    if any(word in query_lower for word in blocked_words):
        logging.warning(f"Blocked suspicious query: {query}")
        print("\nBlocked suspicious prompt.")
        continue
    
    suspicious_patterns = [
        "reveal",
        "hidden",
        "internal",
        "secret",
        "override",
        "bypass",
        "disable",
        "instruction",
        "prompt",
        "confidential",
    ]

    suspicion_score = 0
    for pattern in suspicious_patterns:
        if pattern in query_lower:
            suspicion_score += 1

    if suspicion_score >= 2:
        logging.warning(f"Semantic suspicious query detected: {query}")
        print("\nSuspicious semantic behavior detected.")
        continue

    # Retrieve documents with similarity scores
    results = db.similarity_search_with_score(query, k=5)

    docs = []
    print("\nSimilarity Scores:\n")

    blocked_chunk_patterns = [
        "ignore previous instructions",
        "reveal hidden",
        "admin password",
        "secret password",
        "bypass security",
        "developer instructions",
        "root123",
        "confidential"
    ]

    for doc, score in results:
        print(score)

        chunk_text = doc.page_content.lower()

        poisoned = any(
            pattern in chunk_text
            for pattern in blocked_chunk_patterns
        )

        if poisoned:
            print("\nPoisoned chunk detected and removed.")
            continue

        # Chroma returns distance; lower = more relevant
        if score < 1.0:
            docs.append(doc)

    if not docs:
        print("\nI could not find the answer in the provided document.")
        continue

    context = "\n".join([doc.page_content for doc in docs])
    sensitive_patterns = [
        "password",
        "api key",
        "secret key",
        "private key",
        "admin credentials",
        "token",
    ]

    if any(pattern in context.lower() for pattern in sensitive_patterns):
        print("\nSensitive retrieved content blocked.")
        continue

    prompt = f"""You are a secure RAG assistant.

Security Rules:
1. Answer ONLY using the provided context.
2. Do NOT guess.
3. Do NOT invent information.
4. If the answer is not present in the context, reply exactly:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    blocked_output = [
        "password",
        "secret key",
        "api key",
        "admin credentials",
    ]

    if any(word in response.lower() for word in blocked_output):
        print("\nUnsafe output blocked.")
        continue

    print("\nANSWER:\n")
    print(response)
    query_count += 1