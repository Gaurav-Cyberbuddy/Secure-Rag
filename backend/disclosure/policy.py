from dataclasses import dataclass
from enum import Enum

from backend.auth.models import Role, SensitivityLevel, User
from backend.auth.authorization import is_authorized


class DisclosureDecision(str, Enum):
    ALLOW = "ALLOW"
    LIMITED = "LIMITED"
    DENY = "DENY"


@dataclass(frozen=True)
class DisclosurePolicy:
    """
    Defines how information should be handled for a user.
    """

    role: Role
    sensitivity: SensitivityLevel
    decision: DisclosureDecision


def determine_disclosure_policy(
    user: User,
    sensitivity: SensitivityLevel,
) -> DisclosurePolicy:
    """
    Determine the initial disclosure decision.

    Important:
    This is NOT the final Disclosure Controller.

    It establishes the policy decision that the controller
    will later refine using the actual content and requested
    information.
    """

    authorized = is_authorized(
        user=user,
        sensitivity=sensitivity,
    )

    if authorized:
        return DisclosurePolicy(
            role=user.role,
            sensitivity=sensitivity,
            decision=DisclosureDecision.ALLOW,
        )

    # For now, unauthorized access is classified as LIMITED
    # because the next stage will determine whether some
    # fields/entities can safely be disclosed.
    return DisclosurePolicy(
        role=user.role,
        sensitivity=sensitivity,
        decision=DisclosureDecision.LIMITED,
    )


if __name__ == "__main__":

    users = [
        User("public_001", Role.PUBLIC),
        User("employee_001", Role.EMPLOYEE),
        User("manager_001", Role.MANAGER),
        User("admin_001", Role.ADMIN),
    ]

    print("\n========== DISCLOSURE POLICY TEST ==========")

    for user in users:

        for sensitivity in SensitivityLevel:

            policy = determine_disclosure_policy(
                user=user,
                sensitivity=sensitivity,
            )

            print(
                f"{user.role.value:<10} | "
                f"{sensitivity.value:<12} | "
                f"{policy.decision.value}"
            )

    print("\n============================================")