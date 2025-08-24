import hmac
import hashlib
import os


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    if not signature:
        return False
    try:
        sha_name, sig_hex = signature.split("=", 1)
    except ValueError:
        return False
    if sha_name != "sha256":
        return False
    mac = hmac.new(secret.encode(), msg=payload, digestmod=hashlib.sha256)
    return hmac.compare_digest(mac.hexdigest(), sig_hex)


def secure_github_webhook(payload: bytes, signature: str) -> bool:
    secret = os.getenv("GITHUB_WEBHOOK_SECRET")
    secret = secret.strip() if secret else None
    if not secret:
        raise ValueError("GITHUB_WEBHOOK_SECRET not set.")
    if not verify_signature(payload, signature, secret):
        raise ValueError("Invalid signature for GitHub webhook payload.")
    return True