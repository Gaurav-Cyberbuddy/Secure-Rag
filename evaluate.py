from rouge_score import rouge_scorer
from bert_score import score
import textstat
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')


# =========================
# EXPECTED ANSWER
# =========================

reference = """
Amazon S3 is a scalable cloud object storage service provided by AWS.
"""

# =========================
# GENERATED ANSWER
# Paste your RAG output here
# =========================

generated = """
Amazon S3 is a cloud storage service offered by Amazon Web Services (AWS). It provides scalable object storage with security and high availability.
"""

# =========================
# ROUGE SCORE
# =========================

print("\n===== ROUGE SCORE =====\n")

rouge = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True
)

scores = rouge.score(reference, generated)

for metric, value in scores.items():
    print(metric, value)

# =========================
# BERT SCORE
# =========================

print("\n===== BERT SCORE =====\n")

P, R, F1 = score(
    [generated],
    [reference],
    lang="en"
)

print("Precision:", P.mean().item())
print("Recall:", R.mean().item())
print("F1:", F1.mean().item())

# =========================
# TEXTSTAT READABILITY
# =========================

print("\n===== READABILITY =====\n")

print("Reading Ease:",
      textstat.flesch_reading_ease(generated))

print("Grade Level:",
      textstat.flesch_kincaid_grade(generated))

# =========================
# NLTK TOKENIZATION
# =========================

print("\n===== TOKENS =====\n")

tokens = word_tokenize(generated)

print(tokens)