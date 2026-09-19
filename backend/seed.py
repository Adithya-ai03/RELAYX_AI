from __future__ import annotations

from sqlalchemy import select

from backend.auth import hash_password
from backend.database import SessionLocal
from backend.data_loader import load_csv_data
from backend.models import User
from backend.repositories import replace_all_datasets


def main() -> None:
    if SessionLocal is None:
        raise RuntimeError("DATABASE_URL is required. Configure .env before seeding Ignite.")
    session = SessionLocal()
    try:
        for username, password, role in (("admin", "admin123", "ADMIN"), ("manager", "manager123", "MANAGER")):
            user = session.scalar(select(User).where(User.username == username))
            if user is None:
                session.add(User(username=username, password_hash=hash_password(password), role=role))
            else:
                user.role = role
        issues = replace_all_datasets(session, load_csv_data(demo=True))
        if issues:
            raise RuntimeError("CSV seed validation failed: " + " ".join(issues))
        session.commit()
        print("Seeded demo users and CSV datasets.")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
