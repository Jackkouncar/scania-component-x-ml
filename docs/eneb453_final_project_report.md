# ENEB453 Final Project Report

## Project Title

SCANIA Component X Predictive Maintenance Dashboard

## Team Members

- Jack
- Talia

## Source Code Repository

```text
https://github.com/Jackkouncar/scania-component-x-ml
```

## Contribution Log

| Team Member | Contributions |
|---|---|
| Jack | Data processing, ML model export, Express API integration, database persistence, documentation, testing |
| Talia | Frontend review, dashboard workflow review, presentation support, usability/testing feedback |

## Problem Statement And Motivation

Fleet maintenance teams need early warning when a vehicle component is approaching failure. The SCANIA Component X dataset provides anonymized operational readouts and repair labels for a predictive maintenance problem. This project turns those data files into a web-based monitoring dashboard that visualizes vehicle telemetry, predicts Component X risk, and lets authenticated users create persistent maintenance work orders.

## System Architecture

The final ENEB453 application is a full-stack Node.js web app:

```text
Browser UI
  HTML/CSS dashboard
  Vanilla JS telemetry simulator and canvas charts
  React menu with Telemetry Data, Maintenance Logs, and ML Model tabs
        |
        | fetch / REST API
        v
Express.js server
  Static site hosting
  Authentication routes
  Vehicle summary routes
  Vehicle, telemetry, maintenance log, and ML analysis routes
        |
        v
SQL.js SQLite database
  users
  app_vehicles
  telemetry_records
  dashboard_configuration
  dashboard_vehicle_metadata
  maintenance_notes
  maintenance_note_history
  audit_log
```

Generated files in `site/data` seed the initial database and contain the deployed model artifacts. At runtime, the browser obtains dashboard telemetry from `/api/dashboard/data`, inserts live telemetry through the authenticated API, and executes the API-served exported LightGBM model. Runtime CRUD data is persisted to `database/eneb453_app.sqlite`.

## Database Schema

### `users`

| Field | Type | Notes |
|---|---|---|
| `id` | INTEGER | Primary key |
| `username` | TEXT | Unique login |
| `password_hash` | TEXT | PBKDF2 hash |
| `salt` | TEXT | Per-user salt |
| `role` | TEXT | Example: `admin` |
| `created_at` | TEXT | ISO timestamp |

### `maintenance_notes`

| Field | Type | Notes |
|---|---|---|
| `id` | INTEGER | Primary key |
| `vehicle_id` | INTEGER | Vehicle receiving work order |
| `risk_class` | INTEGER | 0-4 Component X class |
| `status` | TEXT | `open`, `scheduled`, or `resolved` |
| `note` | TEXT | Maintenance note, max 500 chars |
| `created_by` | INTEGER | Foreign key to `users.id` |
| `created_at` | TEXT | ISO timestamp |
| `updated_at` | TEXT | ISO timestamp |
| `deleted_at` | TEXT | Soft-delete timestamp |

### `maintenance_note_history`

| Field | Type | Notes |
|---|---|---|
| `id` | INTEGER | Primary key |
| `note_id` | INTEGER | Related maintenance note |
| `action` | TEXT | `create`, `update`, or `delete` |
| `previous_status` | TEXT | Status before change |
| `new_status` | TEXT | Status after change |
| `previous_note` | TEXT | Note text before change |
| `new_note` | TEXT | Note text after change |
| `actor` | TEXT | Username |
| `created_at` | TEXT | ISO timestamp |

### `app_vehicles`

| Field | Type | Notes |
|---|---|---|
| `id` | INTEGER | Primary key used by telemetry rows |
| `vehicle_key` | TEXT | Unique split/vehicle id key |
| `vehicle_id` | INTEGER | Dashboard vehicle id |
| `split` | TEXT | `train`, `validation`, `test`, or `manual` |
| `risk_class` | INTEGER | 0-4 Component X class |
| `timeline_kind` | TEXT | Data source/category |
| `latest_time_step` | REAL | Most recent telemetry time |
| `active` | INTEGER | Soft archive flag |

### `telemetry_records`

| Field | Type | Notes |
|---|---|---|
| `id` | INTEGER | Primary key |
| `vehicle_db_id` | INTEGER | References `app_vehicles.id` |
| `time_step` | REAL | Relative telemetry time |
| `source` | TEXT | Seeded/manual source |
| `counter_*` | REAL | Eight anonymized counter values |
| `created_at` | TEXT | ISO timestamp |
| `updated_at` | TEXT | ISO timestamp |
| `deleted_at` | TEXT | Soft-delete timestamp for history preservation |

### `dashboard_configuration` And `dashboard_vehicle_metadata`

These tables store the non-editable dashboard configuration and each vehicle's anonymized specifications/failure-display metadata so the main dashboard payload can be rebuilt through the API together with the persisted telemetry rows.

### `audit_log`

| Field | Type | Notes |
|---|---|---|
| `id` | INTEGER | Primary key |
| `actor` | TEXT | Username or system |
| `action` | TEXT | Login/create/update/delete |
| `entity_type` | TEXT | Entity affected |
| `entity_id` | INTEGER | Optional affected row id |
| `created_at` | TEXT | ISO timestamp |

## API Documentation

Full API documentation is provided in `docs/eneb453_api_documentation.md`.

Implemented API groups:

- `GET /api/health`
- `GET /api/dashboard/summary`
- `GET /api/dashboard/data`
- `GET /api/dashboard/model-output`
- `GET /api/dashboard/lightgbm-model`
- `GET /api/vehicles`
- `GET /api/ml/summary`
- `POST /api/ml/run-analysis`
- `GET/POST/PUT/DELETE /api/app-vehicles`
- `GET/POST/PUT/DELETE /api/telemetry-records`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/notes`
- `POST /api/notes`
- `PUT /api/notes/:id`
- `DELETE /api/notes/:id`
- `GET /api/notes/history`

## Frontend Design

The frontend contains:

- Responsive dashboard layout using HTML5 and CSS3
- Canvas-based telemetry timeline and histogram visualizations
- Interactive vehicle lookup, filtering, replay controls, and live telemetry insert form
- Full-set model comparison visualization
- React-based menu with Telemetry Data, Maintenance Logs, and ML Model tabs
- Tabular telemetry and vehicle management with filters by vehicle, split, and class
- Maintenance log form plus active-log and history tables
- ML analysis screen with model comparison and analysis-completed marker
- Form validation and visible API error/status feedback

## Backend Design

The backend uses:

- Node.js
- Express.js
- SQL.js SQLite persistence
- RESTful JSON API routes
- Bearer-token authentication
- Role-based authorization with admin and technician demo users
- Server-side input validation
- Guarded deletes so a vehicle cannot be removed while telemetry or active maintenance logs reference it
- Soft-delete telemetry records so inserted operational history is preserved
- Soft-delete maintenance logs with history preservation
- Password hashing with PBKDF2
- Basic security headers and environment-variable configuration

## Technologies Used

- HTML5
- CSS3
- JavaScript
- React
- Node.js
- Express.js
- SQL.js / SQLite
- esbuild
- TensorFlow.js for optional LSTM experiment
- pptxgenjs/docx for generated submission artifacts

## Deployment Instructions

Local deployment:

```bash
npm install
npm run build:frontend
npm run serve
```

Open:

```text
http://localhost:5173
```

Environment variables are documented in `.env.example`.

Docker deployment:

```bash
docker compose up --build
```

Open the same local URL after the container starts.

## Testing Procedures

The implementation was tested with:

- JavaScript syntax checks for server/frontend scripts
- React bundle build through esbuild
- Express health route smoke test
- Authentication login smoke test
- Vehicle CRUD smoke test
- Telemetry CRUD smoke test
- Guarded delete test returning `409` when records still reference a vehicle
- Create/read/update/soft-delete maintenance note smoke test
- Maintenance note history smoke test
- ML model training run with Random Forest, kNN sweep, Gaussian Naive Bayes, nearest-centroid, and LSTM comparisons
- Docker Compose config validation
- Database-backed dashboard API smoke test
- Live telemetry insert/read/soft-delete persistence test through the API
- Deployed LightGBM artifact endpoint check
- JSON parse checks for generated dashboard/model files

Representative API test flow:

```text
GET /api/health
POST /api/auth/login
POST /api/notes
GET /api/notes
PUT /api/notes/:id
DELETE /api/notes/:id
GET /api/notes/history
```

## Limitations And Future Work

- The demo uses a local SQLite file; cloud deployment would need persistent mounted storage or a managed database.
- Session tokens are in memory, so users must log in again after server restart.
- The dashboard uses anonymized Component X data, so the physical component name cannot be shown.
- Future work could add self-service user management, cloud deployment, WebSocket telemetry, and richer work-order reporting.

## AI Usage Disclosure

Generative AI was used as a coding and documentation assistant. All generated code and documentation were reviewed and integrated into the project by the team, and the team remains responsible for correctness and final submission quality.
