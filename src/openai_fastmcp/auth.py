from __future__ import annotations

import secrets
from typing import Any

from fastmcp.server.auth.auth import AccessToken, TokenVerifier
from fastmcp.utilities.logging import get_logger


logger = get_logger(__name__)


class StaticAPIKeyAuth(TokenVerifier):
    """Simple token verifier that enforces a single API key for every request."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        required_scopes: list[str] | None = None,
    ) -> None:
        super().__init__(base_url=base_url, required_scopes=required_scopes)
        self._api_key = api_key

    async def verify_token(self, token: str) -> AccessToken | None:  # noqa: D401
        """Validate that the caller provided the shared API key."""

        masked_token = _mask_token(token)
        logger.debug(
            "StaticAPIKeyAuth verifying token",  # noqa: TRY400
            extra={
                "masked_token": masked_token,
                "provided": bool(token),
                "required_scopes": self.required_scopes,
            },
        )

        if token and secrets.compare_digest(token, self._api_key):
            logger.info(
                "API key accepted",  # noqa: TRY400
                extra={
                    "masked_token": masked_token,
                    "scopes": self.required_scopes,
                },
            )
            return AccessToken(
                token=token,
                client_id="static-api-key",
                scopes=self.required_scopes,
                expires_at=None,
                claims={"auth": "api-key"},
            )
        logger.warning(
            "API key rejected",  # noqa: TRY400
            extra={
                "masked_token": masked_token,
                "provided": bool(token),
            },
        )
        return None

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover
        """Mirror the auth provider API for logging without leaking the key."""

        return {
            "base_url": str(self.base_url) if self.base_url else None,
            "required_scopes": self.required_scopes,
            "type": "StaticAPIKeyAuth",
        }


def _mask_token(token: str | None) -> str:
    if not token:
        return "<missing>"
    if len(token) <= 8:
        return f"{token[:2]}***{token[-2:]}"
    return f"{token[:4]}***{token[-4:]}"
