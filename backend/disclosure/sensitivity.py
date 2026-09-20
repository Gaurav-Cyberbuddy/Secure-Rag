import re
from dataclasses import dataclass
from enum import Enum


class SensitivityType(str, Enum):
    CREDENTIAL = "CREDENTIAL"
    API_KEY = "API_KEY"
    SECRET = "SECRET"
    PRIVATE_KEY = "PRIVATE_KEY"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    PERSONAL_DATA = "PERSONAL_DATA"


@dataclass(frozen=True)
class SensitiveEntity:
    entity_type: SensitivityType
    value: str
    start: int
    end: int


PATTERNS = {
    SensitivityType.API_KEY: re.compile(
        r"\b(?:api[_ -]?key|apikey)\s*[:=]\s*[A-Za-z0-9_\-]{8,}\b",
        re.IGNORECASE,
    ),

    SensitivityType.CREDENTIAL: re.compile(
        r"\b(?:password|passwd|pwd)\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),

    SensitivityType.SECRET: re.compile(
        r"\b(?:secret|secret[_ -]?key)\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),

    SensitivityType.EMAIL: re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),

    SensitivityType.PHONE: re.compile(
        r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b"
    ),

    SensitivityType.PRIVATE_KEY: re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?"
        r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        re.IGNORECASE | re.DOTALL,
    ),
}


def detect_sensitive_entities(text: str) -> list[SensitiveEntity]:
    """
    Detect potentially sensitive entities in text.

    This detector identifies candidate sensitive information.
    It does NOT decide whether the information should be disclosed.
    """

    entities = []

    for entity_type, pattern in PATTERNS.items():

        for match in pattern.finditer(text):

            entities.append(
                SensitiveEntity(
                    entity_type=entity_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                )
            )

    # Sort entities by their position in the original text.
    entities.sort(key=lambda entity: entity.start)

    return entities


if __name__ == "__main__":

    test_text = """
    Project administrator password: Admin@123
    API key: sk-example123456789
    Contact: example@gmail.com
    Phone: +919876543210
    """

    print("\n========== SENSITIVITY DETECTION TEST ==========")

    entities = detect_sensitive_entities(test_text)

    for entity in entities:

        print(
            f"{entity.entity_type.value:<15} "
            f"→ {entity.value}"
        )

    print("\nTotal entities detected:", len(entities))
    print("================================================")