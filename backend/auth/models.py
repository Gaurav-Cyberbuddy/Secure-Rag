from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    PUBLIC = "PUBLIC"
    EMPLOYEE = "EMPLOYEE"
    MANAGER = "MANAGER"
    ADMIN = "ADMIN"


class SensitivityLevel(str, Enum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"


@dataclass(frozen=True)
class User:
    """
    Represents the identity and role of a user.

    This is the authorization identity that will later
    be used by the Policy Engine and Disclosure Controller.
    """

    user_id: str
    role: Role


@dataclass(frozen=True)
class DocumentSecurityMetadata:
    """
    Security metadata associated with a document.

    This metadata is separate from the document's actual content.
    """

    document_id: str
    sensitivity: SensitivityLevel
    owner_id: str

if __name__ == "__main__":

    user = User(
        user_id="user_001",
        role=Role.EMPLOYEE,
    )

    document = DocumentSecurityMetadata(
        document_id="doc_001",
        sensitivity=SensitivityLevel.CONFIDENTIAL,
        owner_id="admin_001",
    )

    print("========== AUTH MODEL TEST ==========")
    print(f"User ID: {user.user_id}")
    print(f"Role: {user.role.value}")
    print(f"Document: {document.document_id}")
    print(f"Sensitivity: {document.sensitivity.value}")
    print("=====================================")    