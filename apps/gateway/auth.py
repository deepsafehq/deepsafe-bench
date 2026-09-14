"""Supabase JWT verification for DeepSafe API Gateway.

Replaces the old custom OAuth2 auth system with Supabase Auth
JWT verification. Supports both ES256 (JWKS) and HS256 (shared secret).

Security: The server decides which algorithm to use, never the token.
ES256 is attempted first via JWKS; HS256 is the fallback using the
shared secret.
"""

import logging
import os
from typing import Optional, TypedDict

logger = logging.getLogger(__name__)

import jwt as pyjwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

security = HTTPBearer()

SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    os.getenv("NEXT_PUBLIC_SUPABASE_URL", ""),
)
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")

if not SUPABASE_JWT_SECRET:
    logger.warning(
        "SUPABASE_JWT_SECRET is not set. " "HS256 JWT verification will be unavailable."
    )

# JWKS client for ES256 verification (cached)
_jwks_client = None


def _get_jwks_client() -> PyJWKClient:
    """Get or create a cached JWKS client for the Supabase project."""
    global _jwks_client
    if _jwks_client is None:
        if not SUPABASE_URL:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="SUPABASE_URL not configured",
            )
        jwks_url = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
        _jwks_client = PyJWKClient(jwks_url, cache_jwk_set=True, lifespan=300)
    return _jwks_client


class JwtPayload(TypedDict, total=False):
    """Decoded Supabase JWT payload."""

    sub: str
    email: str
    iss: str
    aud: str
    iat: int
    exp: int
    role: str


def verify_supabase_jwt(token: str, jwt_secret: Optional[str] = None) -> JwtPayload:
    """Verify a Supabase JWT and return the decoded payload.

    Security: The server decides the verification method, never the token.
    ES256 (via JWKS) is attempted first. If the JWKS lookup fails (e.g.,
    no matching key), HS256 with the shared secret is tried as a fallback.
    This prevents JWT algorithm confusion attacks where an attacker crafts
    a token with ``alg: HS256`` and signs it with the ES256 public key.

    Args:
        token: The JWT string from the Authorization header.
        jwt_secret: Optional HS256 secret for testing. Defaults to env var.

    Returns:
        Decoded JWT payload dict with 'sub', 'email', etc.

    Raises:
        HTTPException: If the token is invalid, expired, or malformed.
    """
    _issuer = f"{SUPABASE_URL}/auth/v1" if SUPABASE_URL else None

    try:
        # Strategy 1: Try ES256 via JWKS (preferred — Supabase default).
        if SUPABASE_URL:
            try:
                jwks_client = _get_jwks_client()
                signing_key = jwks_client.get_signing_key_from_jwt(token)
                decode_kwargs = {
                    "algorithms": ["ES256"],
                    "audience": "authenticated",
                }
                if _issuer:
                    decode_kwargs["issuer"] = _issuer
                payload = pyjwt.decode(
                    token,
                    signing_key.key,
                    **decode_kwargs,
                )
                return payload
            except (pyjwt.InvalidTokenError, Exception) as es256_err:
                # ES256 failed — fall through to HS256.
                # This is expected for tokens signed with HS256.
                logger.debug("ES256 verification failed, trying HS256: %s", es256_err)

        # Strategy 2: HS256 with shared secret (fallback).
        secret = jwt_secret or SUPABASE_JWT_SECRET
        if not secret:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service is misconfigured",
            )
        decode_kwargs = {
            "algorithms": ["HS256"],
            "audience": "authenticated",
        }
        if _issuer:
            decode_kwargs["issuer"] = _issuer
        payload = pyjwt.decode(
            token,
            secret,
            **decode_kwargs,
        )
        return payload

    except pyjwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except pyjwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Authentication failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
        ) from e


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> JwtPayload:
    """FastAPI dependency that extracts and verifies the Supabase JWT.

    Returns:
        Dict with 'sub' (user UUID), 'email', and other JWT claims.
    """
    return verify_supabase_jwt(credentials.credentials)
