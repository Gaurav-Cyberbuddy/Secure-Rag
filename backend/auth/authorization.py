from backend.auth.models import Role, SensitivityLevel, User


# Higher number = higher access privilege.
ROLE_ACCESS_LEVEL = {
    Role.PUBLIC: 0,
    Role.EMPLOYEE: 1,
    Role.MANAGER: 2,
    Role.ADMIN: 3,
}


# Higher number = more sensitive information.
SENSITIVITY_LEVEL = {
    SensitivityLevel.PUBLIC: 0,
    SensitivityLevel.INTERNAL: 1,
    SensitivityLevel.CONFIDENTIAL: 2,
    SensitivityLevel.RESTRICTED: 3,
}


def is_authorized(
    user: User,
    sensitivity: SensitivityLevel,
) -> bool:
    """
    Determine whether a user's role is authorized to access
    information at the specified sensitivity level.
    """

    user_level = ROLE_ACCESS_LEVEL[user.role]
    required_level = SENSITIVITY_LEVEL[sensitivity]

    return user_level >= required_level


if __name__ == "__main__":

    test_users = [
        User("public_001", Role.PUBLIC),
        User("employee_001", Role.EMPLOYEE),
        User("manager_001", Role.MANAGER),
        User("admin_001", Role.ADMIN),
    ]

    print("\n========== AUTHORIZATION TEST ==========")

    for user in test_users:

        print(f"\nUser: {user.user_id}")
        print(f"Role: {user.role.value}")

        for sensitivity in SensitivityLevel:

            result = is_authorized(
                user,
                sensitivity,
            )

            status = "ALLOW" if result else "DENY"

            print(
                f"  {sensitivity.value:<12} → {status}"
            )

    print("\n========================================")