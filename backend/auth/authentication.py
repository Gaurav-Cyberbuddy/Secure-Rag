import os
from dotenv import load_dotenv

from backend.auth.models import User, Role

load_dotenv()

# ---------------------------------------------------------
# Demo user database
# ---------------------------------------------------------
#
# Credentials are loaded from environment variables so that
# passwords are not stored in the source code.
#
# For local development, create a .env file containing:
#
# EMPLOYEE_PASSWORD=your_password
# MANAGER_PASSWORD=your_password
# ADMIN_PASSWORD=your_password
#
# The .env file must never be committed to Git.
# ---------------------------------------------------------

USERS = {
    "employee_001": {
        "password": os.getenv("EMPLOYEE_PASSWORD"),
        "role": Role.EMPLOYEE,
    },
    "manager_001": {
        "password": os.getenv("MANAGER_PASSWORD"),
        "role": Role.MANAGER,
    },
    "admin_001": {
        "password": os.getenv("ADMIN_PASSWORD"),
        "role": Role.ADMIN,
    },
}


def authenticate(
    user_id: str,
    password: str,
) -> User | None:
    """
    Authenticate a user using user ID and password.

    Returns:
        User object if authentication succeeds.
        None if authentication fails.
    """

    user_record = USERS.get(user_id)

    if user_record is None:
        return None

    stored_password = user_record["password"]

    if stored_password is None:
        return None

    if stored_password != password:
        return None

    return User(
        user_id=user_id,
        role=user_record["role"],
    )


if __name__ == "__main__":

    print("\n========== AUTHENTICATION TEST ==========")

    # Correct credentials
    user = authenticate(
        user_id="employee_001",
        password=os.getenv("EMPLOYEE_PASSWORD", ""),
    )

    if user:
        print("Authentication: SUCCESS")
        print(f"User ID: {user.user_id}")
        print(f"Role: {user.role.value}")
    else:
        print("Authentication: FAILED")

    # Wrong password
    wrong_user = authenticate(
        user_id="employee_001",
        password="wrongpassword",
    )

    if wrong_user:
        print("\nWrong password test: FAILED")
    else:
        print("\nWrong password test: SUCCESS")
        print("Access rejected.")

    print("==========================================")