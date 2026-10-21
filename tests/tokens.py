"""Cognito-like access tokens signed with a test RSA key, and the env that makes promptgen.auth trust them."""

import time

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from promptgen import auth

AUTHORITY = "https://cognito-idp.eu-west-2.amazonaws.com/eu-west-2_test"
UI_CLIENT, ADMIN_CLIENT = "ui-client", "admin-client"
SCOPE = "run-management/run:management"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class FakeJwks:
    def get_signing_key_from_jwt(self, token):
        return type("Key", (), {"key": KEY.public_key()})()


def token(sub="user-1", key=KEY, **overrides) -> str:
    claims = {"sub": sub, "iss": AUTHORITY, "client_id": UI_CLIENT, "token_use": "access", "scope": SCOPE,
              "exp": int(time.time()) + 3600, **overrides}
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, key, algorithm="RS256")


def bearer(**kw) -> dict:
    return {"Authorization": f"Bearer {token(**kw)}"}


def trust(monkeypatch) -> None:
    monkeypatch.delenv("AUTH_DISABLED", raising=False)
    monkeypatch.setenv("AUTH_AUTHORITY", AUTHORITY)
    monkeypatch.setenv("AUTH_ALLOWED_CLIENT_IDS", f"{UI_CLIENT},{ADMIN_CLIENT}")
    monkeypatch.setattr(auth, "_jwks", lambda authority: FakeJwks())
