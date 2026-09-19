from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.data_loader import load_csv_data
from backend.engine import allocate_plan
from backend.main import _persist_plan
from backend.models import Base, DecisionAudit, User
from backend.repositories import load_frames, replace_all_datasets


def test_database_schema_import_and_audit_persistence():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(User(username="admin", password_hash="test", role="ADMIN"))
    assert replace_all_datasets(session, load_csv_data(demo=True)) == []
    session.commit()
    frames = load_frames(session)
    assert [len(frame) for frame in frames] == [6, 6, 5, 10]
    plan = allocate_plan(*frames)
    _persist_plan(session, plan, {"username": "admin", "role": "ADMIN"})
    assert session.query(DecisionAudit).count() == len(plan["allocations"])


def test_database_import_rejects_duplicate_ids_without_partial_commit():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    frames = list(load_csv_data(demo=True))
    frames[0] = frames[0].iloc[[0, 0]].copy()
    issues = replace_all_datasets(session, tuple(frames))
    assert issues
    session.rollback()
    assert session.query(User).count() == 0
