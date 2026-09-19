# Ignite Shipment Recovery Control Tower

Ignite is a Streamlit control tower backed by FastAPI, PostgreSQL, and the existing deterministic shipment recovery engine.

## Architecture

```text
Streamlit -> FastAPI + JWT -> Repository layer -> PostgreSQL
                                      |
                                      v
                         deterministic recovery engine
```

The engine remains the source of truth for priority, route feasibility, capacity, deadline, allocation, and recovery-option scoring. PostgreSQL stores master data, users, recovery runs, allocations, rejected options, audit history, simulations, and AI analysis history.

## Requirements

- Python 3.10+
- SQLite for local development, or PostgreSQL 14+ for deployment

## Installation

```powershell
cd "sofware hackathon project final"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set a real PostgreSQL connection string. Never commit `.env`:

```text
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/ignite
JWT_SECRET_KEY=replace-with-a-long-random-secret
OPENAI_API_KEY=
```

## Database initialization

For the simplest local setup, use SQLite. Create `.env` from `.env.example`; the default value stores the database in `ignite.db`:

```text
DATABASE_URL=sqlite:///./ignite.db
```

No database server is required for SQLite.

For PostgreSQL deployment, replace the value with:

```text
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/ignite
```

Run the migration and seed the database:

```powershell
alembic upgrade head
python -m backend.seed
```

`backend.seed` is safe to run repeatedly. It creates or updates the development users and imports the four existing demo CSV datasets. To import the CSV bundle separately, use:

```powershell
python -m backend.import_csv
```

The migration creates these tables:

- `users`
- `shipments`
- `vehicles`
- `hubs`
- `routes`
- `recovery_runs`
- `allocations`
- `candidate_recovery_options`
- `rejected_recovery_options`
- `decision_audit`
- `simulation_runs`
- `ai_analysis`

## Run the system

Start the API:

```powershell
uvicorn backend.main:app --reload --port 8001
```

Start Streamlit in a second terminal:

```powershell
python -m streamlit run frontend/app.py
```

- FastAPI: http://127.0.0.1:8001
- Swagger: http://127.0.0.1:8001/docs
- Streamlit: http://localhost:8501

`GET /health` reports `database: connected` when PostgreSQL is reachable. If `DATABASE_URL` is missing, database-dependent API routes return a clear `503` configuration error and health reports `degraded`.

## Demo users

Development-only seeded accounts:

- `admin` / `admin123`: full master-data, upload, planning, audit, simulation, and AI access
- `manager` / `manager123`: view, planning, audit, simulation, and AI access; cannot modify master data

Passwords are stored as bcrypt hashes in PostgreSQL.

## Data workflow

CSV files remain supported as an import mechanism. Admin uploads are parsed, checked for required columns, duplicate IDs, types, and engine validation errors before a transaction commits. Manual shipment, vehicle, hub, and route forms use the same FastAPI database path. Invalid imports roll back without partial writes.

The frontend never connects directly to PostgreSQL. It calls `frontend/api_client.py`, which calls FastAPI.

## Testing

Run the existing engine and frontend tests plus the database repository regression tests:

```powershell
python -m pytest tests -q
```

The database tests use an isolated SQLAlchemy SQLite session for deterministic CI-style checks. PostgreSQL deployment verification should additionally run `alembic upgrade head`, `python -m backend.seed`, and the live API health/login checks against your configured database.

## AI analyzer

Set `OPENAI_API_KEY` to enable the OpenAI-compatible analyzer. Without a key, Ignite keeps its deterministic fallback analyzer. Keys are read from environment variables and are never stored in PostgreSQL.
