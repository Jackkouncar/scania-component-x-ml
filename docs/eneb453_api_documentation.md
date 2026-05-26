# ENEB453 API Documentation

Base URL:

```text
http://localhost:5173
```

Authenticated endpoints use:

```text
Authorization: Bearer <token>
```

Demo users:

```text
demoadmin / ComponentX453! / admin
demouser  / ComponentX453! / technician
```

Admin-only API groups:

- `/api/ml/*`
- `/api/app-vehicles`
- `/api/telemetry-records`

Technician users can still create, update, soft-delete, and review maintenance logs.

## Health

### `GET /api/health`

Returns server/database status.

Response:

```json
{
  "status": "ok",
  "database": "database/eneb453_app.sqlite",
  "generatedAt": "2026-05-07T00:00:00.000Z"
}
```

## Dashboard Data API

### `GET /api/dashboard/summary`

Returns a compact dashboard/model summary.

### `GET /api/dashboard/data`

Returns the main dashboard vehicle and telemetry payload from the SQLite-backed `app_vehicles`, `telemetry_records`, and dashboard metadata tables. The browser dashboard uses this route instead of loading the seed JSON directly.

### `GET /api/dashboard/model-output`

Returns the generated model comparison report used by the dashboard header and comparison table.

### `GET /api/dashboard/lightgbm-model`

Returns the exported LightGBM tree artifact used by the live prediction controls. The default live prediction mode is `LightGBM cost-sensitive`.

### `GET /api/vehicles`

Query parameters:

- `search`: optional vehicle id substring
- `split`: `all`, `train`, `validation`, or `test`
- `risk`: `all`, `0`, `1`, `2`, `3`, or `4`
- `limit`: integer from `1` to `100`

Example:

```text
GET /api/vehicles?split=train&risk=4&limit=10
```

## ML Analysis API

### `GET /api/ml/summary`

Requires admin authentication. Returns model comparison metrics, selected k value, validation/test results, and k sweep output.

### `POST /api/ml/run-analysis`

Requires admin authentication. Runs the current ML analysis script on the server and refreshes the exported model output.

## Vehicle And Telemetry CRUD

### `GET /api/app-vehicles`

Requires admin authentication. Supports `search`, `split`, `risk`, `limit`, and `offset` query parameters.

### `POST /api/app-vehicles`

Requires admin authentication. Creates a manually managed vehicle row.

### `PUT /api/app-vehicles/:id`

Requires admin authentication. Updates risk class, split, timeline kind, or latest time step.

### `DELETE /api/app-vehicles/:id`

Requires admin authentication. The backend blocks deletion when telemetry records or active maintenance logs still reference the vehicle.

### `GET /api/telemetry-records`

Requires admin authentication. Displays telemetry rows in tabular form and supports filtering by `vehicle_id` and `risk`.

### `POST /api/telemetry-records`

Requires admin authentication. Creates a telemetry row for an existing vehicle. The main dashboard `Insert Live Record` control calls this route with `source = live_insert`, so new telemetry is persisted in SQLite before LightGBM re-scores it.

### `PUT /api/telemetry-records/:id`

Requires admin authentication. Updates telemetry counters, source, vehicle, or time step.

### `DELETE /api/telemetry-records/:id`

Requires admin authentication. Soft-deletes a telemetry row by setting `deleted_at` and records the action in the audit log. This preserves history while removing the row from active dashboard queries.

## Authentication

### `POST /api/auth/login`

Body:

```json
{
  "username": "demoadmin",
  "password": "ComponentX453!"
}
```

Response:

```json
{
  "token": "...",
  "expiresAt": "2026-05-07T20:00:00.000Z",
  "user": {
    "id": 1,
    "username": "demoadmin",
    "role": "admin"
  }
}
```

Passwords are hashed with PBKDF2 and salted before storage.

### `POST /api/auth/logout`

Requires authentication. Invalidates the current session token.

## Maintenance Notes CRUD

### `GET /api/notes`

Requires authentication. Returns up to 100 recent maintenance work orders.

### `POST /api/notes`

Requires authentication.

Body:

```json
{
  "vehicle_id": 228,
  "risk_class": 4,
  "status": "open",
  "note": "Schedule Component X inspection."
}
```

Validation:

- `vehicle_id`: positive integer
- `risk_class`: integer from `0` to `4`
- `status`: `open`, `scheduled`, or `resolved`
- `note`: required, max 500 characters

### `PUT /api/notes/:id`

Requires authentication. Updates one or more fields.

Body example:

```json
{
  "status": "scheduled"
}
```

### `DELETE /api/notes/:id`

Requires authentication. Soft-deletes a work order from the active list while preserving maintenance history.

### `GET /api/notes/history`

Requires authentication. Returns create/update/delete history for maintenance logs.

## Error Format

```json
{
  "error": "Authentication required."
}
```

## Security Controls

- Express JSON body limit
- Password hashing with PBKDF2
- Bearer-token authorization for CRUD routes
- Parameterized SQL statements
- Server-side input validation
- Security headers for content type, frame protection, referrer policy, permissions policy, and CSP
- Environment-variable based credential/session configuration
