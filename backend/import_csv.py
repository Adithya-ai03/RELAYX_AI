from __future__ import annotations

import sys
from pathlib import Path

from backend.database import SessionLocal
from backend.data_loader import load_csv_data
from backend.repositories import replace_all_datasets


def main() -> None:
    if SessionLocal is None:
        raise RuntimeError("DATABASE_URL is required. Configure .env before importing CSV data.")
    session = SessionLocal()
    try:
        issues = replace_all_datasets(session, load_csv_data(demo=True))
        if issues:
            session.rollback()
            raise RuntimeError("CSV import validation failed: " + " ".join(issues))
        session.commit()
        print("Imported data/shipments.csv, vehicles.csv, hubs.csv, and routes.csv.")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
