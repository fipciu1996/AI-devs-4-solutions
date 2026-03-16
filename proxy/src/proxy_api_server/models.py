"""Typed request/response models for proxy API payload validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class InvalidConversationRequest(ValueError):
    """Raised when the incoming request payload is invalid."""


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise InvalidConversationRequest(f"Field '{field_name}' must be a string.")

    normalized = value.strip()
    if not normalized:
        raise InvalidConversationRequest(
            f"Field '{field_name}' must not be empty."
        )

    return normalized


@dataclass(frozen=True, slots=True)
class ConversationRequest:
    session_id: str
    msg: str

    @classmethod
    def from_payload(cls, payload: Any) -> "ConversationRequest":
        if not isinstance(payload, dict):
            raise InvalidConversationRequest("JSON body must be an object.")

        return cls(
            session_id=_require_non_empty_string(payload.get("sessionID"), "sessionID"),
            msg=_require_non_empty_string(payload.get("msg"), "msg"),
        )

    def to_api_dict(self) -> dict[str, str]:
        return {
            "sessionID": self.session_id,
            "msg": self.msg,
        }


@dataclass(frozen=True, slots=True)
class ConversationResponse:
    msg: str

    def to_api_dict(self) -> dict[str, str]:
        return {"msg": self.msg}
