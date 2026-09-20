from backend.disclosure.sensitivity import (
    SensitiveEntity,
    detect_sensitive_entities,
)


REDACTION_TOKEN = "[REDACTED]"


def redact_sensitive_entities(
    text: str,
    entities: list[SensitiveEntity] | None = None,
) -> str:
    """
    Replace detected sensitive entities with [REDACTED].

    Redaction is performed from right to left so that
    replacing one entity does not change the positions
    of entities that appear earlier in the text.
    """

    if entities is None:
        entities = detect_sensitive_entities(text)

    if not entities:
        return text

    sanitized_text = text

    # Process from the end of the text toward the beginning.
    for entity in sorted(
        entities,
        key=lambda item: item.start,
        reverse=True,
    ):
        sanitized_text = (
            sanitized_text[:entity.start]
            + REDACTION_TOKEN
            + sanitized_text[entity.end:]
        )

    return sanitized_text


if __name__ == "__main__":

    test_text = """
    The project used AWS services.

    The administrator password is: Admin@123

    The API key is: sk-example123456789

    Contact email: example@gmail.com
    """

    print("\n========== REDACTION TEST ==========")

    print("\nORIGINAL:")
    print(test_text)

    entities = detect_sensitive_entities(test_text)

    sanitized = redact_sensitive_entities(
        text=test_text,
        entities=entities,
    )

    print("\nSANITIZED:")
    print(sanitized)

    print("====================================")