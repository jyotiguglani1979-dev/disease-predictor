import base64, hashlib, hmac, json, re, secrets, time, uuid
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from .config import APP_SECRET
from .database import SessionLocal
from .models import UserAccount

bearer = HTTPBearer(auto_error=False)

def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")

def password_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
    return f"{_b64(salt)}${_b64(digest)}"

def verify_password(password: str, stored: str) -> bool:
    try:
        salt_text, digest_text = stored.split("$", 1)
        salt = base64.urlsafe_b64decode(salt_text + "=" * (-len(salt_text) % 4))
        return hmac.compare_digest(password_hash(password, salt).split("$", 1)[1], digest_text)
    except Exception: return False

def make_token(user_id: str, *, anonymous: bool = False) -> str:
    claims = {"sub": user_id, "exp": int(time.time()) + 86400}
    if anonymous: claims["kind"] = "anonymous"
    payload = _b64(json.dumps(claims, separators=(",", ":")).encode())
    signature = _b64(hmac.new(APP_SECRET.encode(), payload.encode(), hashlib.sha256).digest())
    return payload + "." + signature

def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> UserAccount | None:
    if not credentials:
        # Local college-project mode: assessments are anonymous and never tied to an account.
        return None
    try:
        payload, signature = credentials.credentials.split(".", 1)
        expected = _b64(hmac.new(APP_SECRET.encode(), payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected): raise ValueError("signature")
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        if claims["exp"] < time.time(): raise ValueError("expired")
        user_id = claims["sub"]
    except Exception: raise HTTPException(401, "Session expired or invalid. Please sign in again", headers={"WWW-Authenticate": "Bearer"})
    if claims.get("kind") == "anonymous":
        try: uuid.UUID(user_id)
        except (ValueError, TypeError, AttributeError): raise HTTPException(401, "Session expired or invalid")
        return type("AnonymousSession", (), {"id": user_id, "email": "Guest"})()
    db: Session = SessionLocal()
    try:
        user = db.get(UserAccount, user_id)
        if not user: raise HTTPException(401, "Account not found")
        return user
    finally: db.close()

def valid_email(email: str) -> bool:
    return bool(re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email))
