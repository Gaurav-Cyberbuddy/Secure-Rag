from dataclasses import dataclass

from backend.auth.models import User
from backend.disclosure.risk import assess_query_risk
from backend.disclosure.sensitivity import (
    SensitiveEntity,
    detect_sensitive_entities,
)
from backend.disclosure.rules import is_entity_allowed


class DisclosureDecision:
    ALLOW = "ALLOW"
    LIMITED = "LIMITED"
    DENY = "DENY"


@dataclass(frozen=True)
class DisclosureResult:
    decision: str
    original_evidence: str
    sanitized_evidence: str
    detected_entities: tuple[SensitiveEntity, ...]
    redacted_entities: tuple[SensitiveEntity, ...]
    risk_score: int
    risk_level: str
    reasons: tuple[str, ...]


def redact_entities(
    text: str,
    entities: list[SensitiveEntity],
) -> str:
    """
    Replace selected sensitive entities with [REDACTED].
    """

    if not entities:
        return text

    sanitized = text

    # Work from right to left so entity positions remain valid.
    for entity in sorted(
        entities,
        key=lambda item: item.start,
        reverse=True,
    ):
        sanitized = (
            sanitized[:entity.start]
            + "[REDACTED]"
            + sanitized[entity.end:]
        )

    return sanitized


def control_disclosure(
    user: User,
    query: str,
    evidence: str,
) -> DisclosureResult:
    """
    Main disclosure-control mechanism.

    Decision process:

        Query
          ↓
        Risk assessment
          ↓
        Sensitive entity detection
          ↓
        Role-based policy
          ↓
        ALLOW / LIMITED / DENY
          ↓
        Sanitized evidence
          ↓
        Fail-closed if no useful evidence remains
    """

    # ---------------------------------------------------------
    # 1. Assess query risk
    # ---------------------------------------------------------

    risk = assess_query_risk(query)

    # ---------------------------------------------------------
    # 2. Detect sensitive entities in retrieved evidence
    # ---------------------------------------------------------

    entities = detect_sensitive_entities(evidence)

    # ---------------------------------------------------------
    # 3. Critical query risk
    #
    # A request explicitly targeting secrets, credentials,
    # prompt extraction, or security bypass is denied.
    # ---------------------------------------------------------

    if risk.level.value == "CRITICAL":

        return DisclosureResult(
            decision=DisclosureDecision.DENY,
            original_evidence=evidence,
            sanitized_evidence="",
            detected_entities=tuple(entities),
            redacted_entities=tuple(entities),
            risk_score=risk.score,
            risk_level=risk.level.value,
            reasons=tuple(
                list(risk.reasons)
                + ["Critical-risk request denied."]
            ),
        )

    # ---------------------------------------------------------
    # 4. Determine which sensitive entities are unauthorized
    # ---------------------------------------------------------

    unauthorized_entities = []

    for entity in entities:

        allowed = is_entity_allowed(
            role=user.role,
            entity_type=entity.entity_type,
        )

        if not allowed:
            unauthorized_entities.append(entity)

    # ---------------------------------------------------------
    # 5. Nothing sensitive detected
    # ---------------------------------------------------------

    if not entities:

        return DisclosureResult(
            decision=DisclosureDecision.ALLOW,
            original_evidence=evidence,
            sanitized_evidence=evidence,
            detected_entities=(),
            redacted_entities=(),
            risk_score=risk.score,
            risk_level=risk.level.value,
            reasons=tuple(risk.reasons),
        )

    # ---------------------------------------------------------
    # 6. Everything detected is authorized
    # ---------------------------------------------------------

    if not unauthorized_entities:

        return DisclosureResult(
            decision=DisclosureDecision.ALLOW,
            original_evidence=evidence,
            sanitized_evidence=evidence,
            detected_entities=tuple(entities),
            redacted_entities=(),
            risk_score=risk.score,
            risk_level=risk.level.value,
            reasons=tuple(risk.reasons),
        )

    # ---------------------------------------------------------
    # 7. Redact unauthorized information
    # ---------------------------------------------------------

    sanitized = redact_entities(
        text=evidence,
        entities=unauthorized_entities,
    )

    # ---------------------------------------------------------
    # 8. Fail-closed fallback
    #
    # If redaction leaves no meaningful evidence,
    # do not send anything to the LLM.
    # ---------------------------------------------------------

    remaining_content = sanitized.replace(
        "[REDACTED]",
        "",
    ).strip()

    if not remaining_content:

        return DisclosureResult(
            decision=DisclosureDecision.DENY,
            original_evidence=evidence,
            sanitized_evidence="",
            detected_entities=tuple(entities),
            redacted_entities=tuple(unauthorized_entities),
            risk_score=risk.score,
            risk_level=risk.level.value,
            reasons=tuple(
                list(risk.reasons)
                + ["No useful evidence remained after redaction."]
            ),
        )

    # ---------------------------------------------------------
    # 9. Useful evidence remains
    # ---------------------------------------------------------

    return DisclosureResult(
        decision=DisclosureDecision.LIMITED,
        original_evidence=evidence,
        sanitized_evidence=sanitized,
        detected_entities=tuple(entities),
        redacted_entities=tuple(unauthorized_entities),
        risk_score=risk.score,
        risk_level=risk.level.value,
        reasons=tuple(
            list(risk.reasons)
            + ["Unauthorized sensitive entities were redacted."]
        ),
    )


if __name__ == "__main__":

    from backend.auth.models import Role

    print("\n========== DISCLOSURE CONTROLLER TEST ==========")

    evidence = """
    The project used AWS S3 for object storage.

    Administrator email: admin@example.com

    API key: sk-example123456789

    Project objective: secure cloud storage.
    """

    test_cases = [
        User("employee_001", Role.EMPLOYEE),
        User("manager_001", Role.MANAGER),
        User("admin_001", Role.ADMIN),
    ]

    query = "What technology was used in the project?"

    for user in test_cases:

        result = control_disclosure(
            user=user,
            query=query,
            evidence=evidence,
        )

        print("\n---------------------------------------------")
        print(f"Role: {user.role.value}")
        print(f"Decision: {result.decision}")
        print(f"Risk: {result.risk_score} ({result.risk_level})")

        print("\nDetected entities:")

        for entity in result.detected_entities:
            print(
                f"  {entity.entity_type.value}: "
                f"{entity.value}"
            )

        print("\nSanitized evidence:")
        print(result.sanitized_evidence)

        print("---------------------------------------------")

    print("\n==============================================")