from backend.auth.models import Role
from backend.disclosure.sensitivity import SensitivityType


# Information sensitivity policy.
#
# Higher roles inherit the permissions of lower roles.
# This is the policy used by the Disclosure Controller.

ROLE_PERMISSIONS = {
    Role.PUBLIC: {
        SensitivityType.EMAIL: False,
        SensitivityType.PHONE: False,
        SensitivityType.CREDENTIAL: False,
        SensitivityType.API_KEY: False,
        SensitivityType.SECRET: False,
        SensitivityType.PRIVATE_KEY: False,
        SensitivityType.PERSONAL_DATA: False,
    },

    Role.EMPLOYEE: {
        SensitivityType.EMAIL: False,
        SensitivityType.PHONE: False,
        SensitivityType.CREDENTIAL: False,
        SensitivityType.API_KEY: False,
        SensitivityType.SECRET: False,
        SensitivityType.PRIVATE_KEY: False,
        SensitivityType.PERSONAL_DATA: False,
    },

    Role.MANAGER: {
        SensitivityType.EMAIL: True,
        SensitivityType.PHONE: True,
        SensitivityType.CREDENTIAL: False,
        SensitivityType.API_KEY: False,
        SensitivityType.SECRET: False,
        SensitivityType.PRIVATE_KEY: False,
        SensitivityType.PERSONAL_DATA: True,
    },

    Role.ADMIN: {
        SensitivityType.EMAIL: True,
        SensitivityType.PHONE: True,
        SensitivityType.CREDENTIAL: True,
        SensitivityType.API_KEY: True,
        SensitivityType.SECRET: True,
        SensitivityType.PRIVATE_KEY: True,
        SensitivityType.PERSONAL_DATA: True,
    },
}


def is_entity_allowed(
    role: Role,
    entity_type: SensitivityType,
) -> bool:
    """
    Determine whether a specific information type can be
    disclosed to a particular user role.
    """

    permissions = ROLE_PERMISSIONS.get(role, {})

    return permissions.get(entity_type, False)


if __name__ == "__main__":

    print("\n========== DISCLOSURE RULE TEST ==========")

    for role in Role:

        print(f"\nRole: {role.value}")

        for entity_type in SensitivityType:

            allowed = is_entity_allowed(
                role,
                entity_type,
            )

            decision = "ALLOW" if allowed else "REDACT"

            print(
                f"  {entity_type.value:<15} → {decision}"
            )

    print("\n==========================================")