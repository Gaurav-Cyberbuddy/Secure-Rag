from backend.rag.secure_pipeline import verify_answer_claims, get_evidence_checker

EV = ("The internship provided practical use of tools such as Visual Studio Code, "
      "Git, GitHub, Postman, and Kali Linux, which enhanced familiarity with "
      "professional development environments.")

r = verify_answer_claims(
    "The technologies mentioned in the internship experience include Git, Postman, and Metasploit.",
    EV,
    get_evidence_checker()
)

print("EXPECT False ->", r["supported"])
print(r["reason"])
