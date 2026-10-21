"""Cognito access-token check with RunManagementAPI's rules: signed by the user pool (issuer),
token_use is "access", client_id is an allowed UI client, and the route's scope is present.
API Gateway's JWT authorizer checks the token first; this keeps the app safe behind anything else."""

import os
from functools import lru_cache

import jwt
from fastapi import Depends, Header, HTTPException

LOCAL_CLAIMS = {"sub": "local", "client_id": "local", "token_use": "access", "scope": ""}


@lru_cache
def _jwks(authority: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(f"{authority}/.well-known/jwks.json")  # caches signing keys per container


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(401, detail, headers={"WWW-Authenticate": "Bearer"})


def principal(authorization: str = Header("")) -> dict:
    """Validated token claims. AUTH_DISABLED=1 (local runs only) skips the check."""
    if os.getenv("AUTH_DISABLED") == "1":
        return LOCAL_CLAIMS
    authority = os.getenv("AUTH_AUTHORITY", "").rstrip("/")  # https://cognito-idp.<region>.amazonaws.com/<poolId>
    allowed = {c.strip() for c in os.getenv("AUTH_ALLOWED_CLIENT_IDS", "").split(",") if c.strip()}
    if not authority or not allowed:  # fail closed on a misconfigured deploy
        raise HTTPException(503, "authentication is not configured")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _unauthorized("missing bearer token")
    try:
        key = _jwks(authority).get_signing_key_from_jwt(token.strip()).key
        claims = jwt.decode(token.strip(), key, algorithms=["RS256"], issuer=authority,
                            options={"verify_aud": False,  # Cognito access tokens carry client_id, not aud
                                     "require": ["exp", "iss", "sub", "token_use", "client_id"]})
    except jwt.PyJWTError:
        raise _unauthorized("invalid token")
    if claims["token_use"] != "access" or claims["client_id"] not in allowed:
        raise _unauthorized("not an approved Cognito access token")
    return claims


def require_scope(scope: str):
    def check(claims: dict = Depends(principal)) -> dict:
        if claims is not LOCAL_CLAIMS and scope not in claims.get("scope", "").split():
            raise HTTPException(403, "missing required scope")
        return claims

    return Depends(check)
