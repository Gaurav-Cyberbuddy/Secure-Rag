from fastapi.testclient import TestClient

from backend.api.main import app
import os


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_login_success():
    response = client.post(
        "/login",
        json={
            "user_id": "employee_001",
            "password": os.getenv("EMPLOYEE_PASSWORD"),
           
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["authenticated"] is True
    assert data["user_id"] == "employee_001"
    assert data["role"] == "EMPLOYEE"


def test_login_failure():
    response = client.post(
        "/login",
        json={
            "user_id": "employee_001",
            "password": "wrongpassword",
        },
    )

    assert response.status_code == 401


def test_empty_question():
    response = client.post(
        "/ask",
        json={
            "user_id": "employee_001",
            "password": os.getenv("EMPLOYEE_PASSWORD"),
            "question": "",
            "filename": "paper (1).pdf",
        },
    )

    assert response.status_code == 400