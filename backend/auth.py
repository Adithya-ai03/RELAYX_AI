from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

SECRET_KEY = os.getenv("SECRET_KEY", "ignite-demo-change-this-secret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")
)

PASSWORD_CONTEXT = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

OAUTH2_SCHEME = OAuth2PasswordBearer(
    tokenUrl="/auth/login"
)

DEMO_USERS = {
    "admin": {
        "password": "admin123",
        "role": "ADMIN"
    },
    "manager": {
        "password": "manager123",
        "role": "MANAGER"
    }
}


def hash_password(password: str) -> str:
    return PASSWORD_CONTEXT.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return PASSWORD_CONTEXT.verify(password, password_hash)


def authenticate_user(
    username: str,
    password: str
) -> dict[str, str] | None:

    username = username.strip().lower()

    user = DEMO_USERS.get(username)

    if user is None:
        return None

    if password != user["password"]:
        return None

    return {
        "username": username,
        "role": user["role"]
    }


def authenticate_database_user(session: Any, username: str, password: str) -> dict[str, str] | None:
    from backend.models import User

    user = session.get(User, username.strip().lower())
    if user is None or not verify_password(password, user.password_hash):
        return None
    return {"username": user.username, "role": user.role}


def create_access_token(
    subject: str,
    role: str
) -> str:

    expires = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": subject,
        "role": role,
        "exp": expires
    }

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )


def current_user(
    token: str = Depends(OAUTH2_SCHEME)
) -> dict[str, str]:

    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired session",
        headers={"WWW-Authenticate": "Bearer"}
    )

    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        username = payload.get("sub")
        role = payload.get("role")

        if not username:
            raise credentials_error

        if role not in {"ADMIN", "MANAGER"}:
            raise credentials_error

        return {
            "username": str(username),
            "role": str(role)
        }

    except JWTError as error:
        raise credentials_error from error


def require_manager(
    user: dict[str, str] = Depends(current_user)
) -> dict[str, str]:

    return user


def require_admin(
    user: dict[str, str] = Depends(current_user)
) -> dict[str, str]:

    if user["role"] != "ADMIN":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required"
        )

    return user