from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.auth.authentication import authenticate
from backend.rag.secure_pipeline import secure_ask


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

VECTOR_DB = PROJECT_DIR / "vector_db"


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Secure AskRAG API",
    description=(
        "Secure RAG API with authentication, authorization, "
        "disclosure control, and semantic verification."
    ),
    version="1.0.0",
)


# =========================================================
# CORS CONFIGURATION
# =========================================================

# The frontend is served from port 5500 while the API
# runs on port 8000. CORS allows the browser to connect
# these two different origins.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST MODELS
# =========================================================

class LoginRequest(BaseModel):
    user_id: str
    password: str


class AskRequest(BaseModel):
    user_id: str
    password: str
    question: str
    filename: str | None = None


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "Secure AskRAG API",
    }


# =========================================================
# LOGIN
# =========================================================

@app.post("/login")
def login(request: LoginRequest):

    user = authenticate(
        user_id=request.user_id,
        password=request.password,
    )

    if user is None:

        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    return {
        "authenticated": True,
        "user_id": user.user_id,
        "role": user.role.value,
    }


# =========================================================
# ASK QUESTION
# =========================================================

@app.post("/ask")
def ask(request: AskRequest):

    # -----------------------------------------------------
    # Validate question
    # -----------------------------------------------------

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # -----------------------------------------------------
    # Authenticate user
    # -----------------------------------------------------

    user = authenticate(
        user_id=request.user_id,
        password=request.password,
    )

    if user is None:

        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )

    # -----------------------------------------------------
    # Run Secure AskRAG pipeline
    # -----------------------------------------------------

    result = secure_ask(
        user=user,
        query=request.question,
        vector_db=str(VECTOR_DB),
        filename=request.filename,
    )

    # -----------------------------------------------------
    # Return security decision + answer
    # -----------------------------------------------------

    return result