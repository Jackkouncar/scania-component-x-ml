# Predictive Maintenance Dashboard

Members: Jack and Talia

Goal: Visualize SCANIA Component X equipment health and predict failure risk before maintenance events happen.

GitHub repository for review and resubmission:

```text
https://github.com/Jackkouncar/scania-component-x-ml
```

## Corrected Runtime Path

- The main dashboard now loads vehicle and telemetry records from `GET /api/dashboard/data`, backed by SQLite.
- `Insert Live Record` posts to the authenticated telemetry API and persists a real database row.
- Clearing a live record performs a soft delete so its database history is preserved.
- The live ML prediction selector defaults to the exported LightGBM cost-sensitive model, served by `GET /api/dashboard/lightgbm-model`.

## What Is Here

- `2024-34-2/2024-34-2/data/`: Optional original SCANIA CSV location when raw files are supplied separately.
- `2024-34-2/2024-34-2/documentation/`: Optional original dataset/challenge PDF location.
- `docs/data_understanding.md`: Notes from the PDFs plus local CSV profiling.
- `docs/article_findings_and_time.md`: Professor-facing explanation of anonymization, time units, and failure windows.
- `docs/wireframe.md`: First UI wireframe for the dashboard.
- `docs/erd_3nf.md`: Preliminary 3NF database schema and Mermaid ERD.
- `docs/demo_queries.sql`: SQL queries to demonstrate the populated database.
- `tools/build_demo_database.js`: Builds a balanced demo SQLite database from the local CSVs.
- `tools/run_demo_queries.js`: Runs the demo queries against the generated database.
- `tools/serve_fullstack.js`: Express API and SQLite-backed dashboard server.
- `site/data/lightgbm_model.json`: exported LightGBM model used for live dashboard predictions.
- `ml/ml_dataset.json.gz.part*`: compressed split archive of the current processed ML dataset for GitHub distribution.

## Quick Start

The checked-in dashboard/model exports are sufficient to run the corrected app without the large raw CSV files:

```bash
npm install
npm run build:frontend
npm run serve
```

Then open:

```text
http://localhost:5173
```

## Telemetry Demo

The webapp simulates telemetry by replaying real SCANIA vehicle readouts one packet at a time.

Demo flow:

1. Open `http://localhost:5173`.
2. Select a vehicle such as `228` from the train split.
3. Click `Reset Stream`.
4. Click `Send Next Telemetry` or `Auto Play`.
5. Watch the Component X health prediction, event log, telemetry packet, and trend graph update.

The dashboard defaults to `LightGBM cost-sensitive`, the selected model from the full validation comparison. The centroid, kNN, probability-trend, expected-cost, and baseline modes remain available only as comparisons.

Use the `Auto speed` dropdown to speed up playback. Use `Jump to time_step`, `Jump to display date`, or `Jump Near Failure Window` to skip through the simulated telemetry stream.

Use the `Telemetry mode` dropdown to switch from historical replay to `Real-time insert`. Sign in as the demo admin in Operations Console, enter a new `time_step` and counter values, and click `Insert Live Record`. The button creates a row through `POST /api/telemetry-records`; the timeline and LightGBM prediction refresh from that persisted telemetry.

The listed telemetry values are not separate parts failing together. They are anonymized input signals from the same vehicle readout. The predicted failure target is one component: Component X.

## ENEB453 Full-Stack Web App

The `npm run serve` command starts the Express.js full-stack server. It serves the dashboard and exposes REST APIs for dashboard data, the deployed LightGBM artifact, authentication, telemetry CRUD, vehicle CRUD, maintenance log CRUD, maintenance history, and ML analysis.

Demo logins:

```text
admin:      demoadmin / ComponentX453!
technician: demouser  / ComponentX453!
```

The admin role can access Telemetry Data, Maintenance Logs, and ML Model. The technician role only sees Maintenance Logs and receives a `403` response if it calls admin-only API routes.

Inspect the live app database:

```bash
npm run db:app
npm run db:users
npm run db:notes:active
npm run db:notes:deleted
npm run db:history
```

Inspect a specific backend table:

```bash
npm run db:app --table=app_vehicles --limit=10
```

453-specific documents:

```text
docs/eneb453_setup_guide.md
docs/eneb453_api_documentation.md
docs/eneb453_final_project_report.md
```

Docker run:

```bash
docker compose up --build
```

Then open `http://localhost:5173`.

## Machine Learning

For the GitHub submission, the expanded processed feature table is stored as compressed split files because the single JSON file exceeds GitHub's direct-file limit. Restore it once before rerunning training:

```bash
npm run restore:ml-data
```

Alternatively, rebuild the supervised ML feature table from the raw SCANIA CSV files:

```bash
npm run build:ml-data
```

Train the baseline ML model:

```bash
npm run train:model
```

This combines the comparison artifacts and exports the dashboard report. The deployed live scorer is the exported LightGBM model created by `npm run train:tabular`; centroid and kNN are retained as comparison options. To force the archived kNN sweep to select by raw accuracy or lowest cost instead:

```bash
npm run train:model:accuracy
```

```bash
npm run train:model:cost
```

Train the optional LSTM sequence experiment:

```bash
npm run train:lstm
```

Run a longer LSTM training pass for presentation/testing:

```bash
npm run train:lstm:deep
```

Train the Python tabular comparators:

```bash
py -m pip install -r requirements-ml.txt
npm run train:tabular
npm run train:model
```

Generated files:

```text
ml/ml_dataset.json
ml/model_report.md
site/data/lstm_model_output.json
site/data/tabular_model_output.json
site/data/model_output.json
site/data/lightgbm_model.json
```

Current recommended fair model:

```text
LightGBM cost-sensitive
```

The browser-live scorer executes the exported LightGBM trees by default, with centroid, kNN, and baseline modes kept as explainable comparison options. The fair model-selection table recommends LightGBM because its validation SCANIA cost is `35,599`, below the all-class-0 validation baseline of `57,400`. Accuracy is still shown beside cost, but cost drives the choice because the dataset is heavily imbalanced.

This is an explainable supervised baseline. It compares each incoming telemetry packet with the learned class pattern for each risk class using engineered telemetry features:

- current counter values
- counter deltas
- counter rates
- relative time_step
- vehicle specification categories

The dashboard can toggle between `ML model` and `Baseline demo` prediction modes. In ML mode, the dashboard also uses temporal smoothing so a single noisy packet cannot make the displayed health jump from critical straight back to healthy.

The current report compares all-class-0, nearest-centroid, Gaussian Naive Bayes, Random Forest, distance-weighted kNN, Logistic Regression, and LightGBM on the complete validation/test sets. The optional TensorFlow.js LSTM sequence experiment is listed separately until it is rebuilt on the same full validation/test scope. The dashboard now keeps the first screen focused on the fair model comparison instead of the archived kNN sweep.

The normal full rebuild finishes quickly for class demos. The `train:lstm:deep` script uses longer sequences, more samples, more LSTM units, and more epochs, so it is the legitimate longer-running training option.

Final-project rubric notes and an IEEE-style paper draft are in:

```text
docs/final_project_rubric_checklist.md
docs/ieee_paper_draft.md
```
