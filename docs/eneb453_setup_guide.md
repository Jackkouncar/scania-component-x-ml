# ENEB453 Setup Guide

## Project

SCANIA Component X Predictive Maintenance Dashboard

## Requirements

- Node.js 20 or newer
- npm
- Local project files, including the `site/data/*.json` database seed/model exports

The Express server seeds its SQLite dashboard tables from the checked-in dashboard export on first run, then the browser reads through the live API. The raw SCANIA dataset folder is only needed if rebuilding the ML/dashboard exports from the original CSV files.

## Install

```bash
npm install
```

## Environment

Copy `.env.example` to `.env` for deployment or local credential changes.

Default local demo values:

```text
PORT=5173
APP_DB_FILE=database/eneb453_app.sqlite
DEMO_ADMIN_USERNAME=demoadmin
DEMO_USER_USERNAME=demouser
DEMO_PASSWORD=ComponentX453!
```

Use a long random `SESSION_SECRET` before deploying publicly.

## Build Frontend Bundle

```bash
npm run build:frontend
```

This bundles the React admin menu into `site/react_admin.bundle.js`.

## Start Full-Stack App

```bash
npm run serve
```

Open:

```text
http://localhost:5173
```

## Docker

Build and run with Docker Compose:

```bash
docker compose up --build
```

The app will be available at:

```text
http://localhost:5173
```

The compose file uses a named volume for the SQLite database so maintenance records persist across container restarts.

## Optional Data/ML Rebuild

```bash
npm run build:ml-data
npm run train:lstm
npm run train:model
npm run export:dashboard
npm run build:frontend
npm run serve
```

## Demo Logins

```text
Admin:
Username: demoadmin
Password: ComponentX453!

Technician:
Username: demouser
Password: ComponentX453!
```

The admin role can access Telemetry Data, Maintenance Logs, and ML Model tabs. The technician role can access Maintenance Logs only.

To demonstrate database-backed live telemetry, sign in as the admin, change the simulator to `Real-time insert`, and click `Insert Live Record`. The new row is created through `/api/telemetry-records` and immediately appears in `/api/dashboard/data`.

## Generated Database

The Express backend creates and persists:

```text
database/eneb453_app.sqlite
```

To inspect the backend tables from the terminal:

```bash
npm run db:app
npm run db:users
npm run db:notes:active
npm run db:notes:deleted
npm run db:history
```

To view rows from one table:

```bash
npm run db:app --table=telemetry_records --limit=10
```

Tables:

- `users`
- `app_vehicles`
- `telemetry_records`
- `dashboard_configuration`
- `dashboard_vehicle_metadata`
- `maintenance_notes`
- `maintenance_note_history`
- `audit_log`

The database file is local runtime state and is intentionally excluded from Git.
