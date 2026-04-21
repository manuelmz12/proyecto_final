import os

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()


def _hash(password: str) -> bytes:
    return bcrypt.hashpw(password.encode()[:72], bcrypt.gensalt())


def _verify(password: str, password_hash: bytes) -> bool:
    return bcrypt.checkpw(password.encode()[:72], password_hash)


USERS_DB: dict[str, dict] = {
    os.environ.get("ADMIN_USER", "admin"): {
        "password_hash": _hash(os.environ.get("ADMIN_PASSWORD", "admin123")),
        "role": "admin",
    },
    os.environ.get("DEMO_USER", "demo"): {
        "password_hash": _hash(os.environ.get("DEMO_PASSWORD", "demo123")),
        "role": "user",
    },
}


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    token = credentials.credentials
    if ":" not in token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Formato inválido. Usa username:password como Bearer token",
        )
    username, password = token.split(":", 1)
    user = USERS_DB.get(username)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuario no existe")
    if not _verify(password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Contraseña incorrecta")
    return {"username": username, "role": user["role"]}


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No tienes permisos de administrador")
    return user
