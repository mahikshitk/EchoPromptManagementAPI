"""Process setup shared by the Lambda entry points (api, worker, database)."""

import json
import logging
import os


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO)  # local runs; a no-op on Lambda, which installs its own handler
    logging.getLogger().setLevel(os.getenv("LOG_LEVEL", "INFO"))


def secret(arn_var: str) -> str | None:
    """Secrets Manager string for the ARN in env var arn_var, or None when the variable is unset."""
    arn = os.getenv(arn_var)
    if not arn:
        return None
    import boto3

    return boto3.client("secretsmanager").get_secret_value(SecretId=arn)["SecretString"].strip()


def load_env_secret(name: str, arn_var: str) -> None:
    """Set env var name from the secret at arn_var once per container, unless it is already set."""
    if not os.getenv(name) and (value := secret(arn_var)):
        os.environ[name] = value


def json_secret(arn_var: str) -> dict:
    value = secret(arn_var)
    if not value:
        raise RuntimeError(f"{arn_var} is not set")
    return json.loads(value)
