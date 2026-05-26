const crypto = require('crypto');
const { execFile } = require('child_process');
const fs = require('fs');
const path = require('path');
const express = require('express');
const initSqlJs = require('sql.js');

const ROOT = path.resolve(__dirname, '..');
loadDotEnv(path.join(ROOT, '.env'));

const SITE_DIR = path.join(ROOT, 'site');
const DATA_DIR = path.join(SITE_DIR, 'data');
const DB_FILE = path.resolve(process.env.APP_DB_FILE || path.join(ROOT, 'database', 'eneb453_app.sqlite'));
const PORT = Number(process.env.PORT || 5173);
const SESSION_SECRET = process.env.SESSION_SECRET || 'local-dev-session-secret-change-for-deployment';
const DEMO_ADMIN_USERNAME = process.env.DEMO_ADMIN_USERNAME || 'demoadmin';
const DEMO_USER_USERNAME = process.env.DEMO_USER_USERNAME || 'demouser';
const DEMO_PASSWORD = process.env.DEMO_PASSWORD || process.env.DEMO_ADMIN_PASSWORD || 'ComponentX453!';
const TOKEN_TTL_MS = Number(process.env.SESSION_TTL_MS || 8 * 60 * 60 * 1000);
const COUNTER_COLUMNS = [
  ['100_0', 'counter_100_0'],
  ['171_0', 'counter_171_0'],
  ['309_0', 'counter_309_0'],
  ['370_0', 'counter_370_0'],
  ['427_0', 'counter_427_0'],
  ['666_0', 'counter_666_0'],
  ['835_0', 'counter_835_0'],
  ['837_0', 'counter_837_0']
];

let db;
const sessions = new Map();
let analysisRunning = false;

function loadDotEnv(filePath) {
  if (!fs.existsSync(filePath)) return;
  const lines = fs.readFileSync(filePath, 'utf8').split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#') || !trimmed.includes('=')) continue;
    const [key, ...valueParts] = trimmed.split('=');
    if (!key || process.env[key] !== undefined) continue;
    process.env[key] = valueParts.join('=').replace(/^["']|["']$/g, '');
  }
}

function nowIso() {
  return new Date().toISOString();
}

function sanitizeInt(value, { min = Number.MIN_SAFE_INTEGER, max = Number.MAX_SAFE_INTEGER } = {}) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < min || parsed > max) return null;
  return parsed;
}

function sanitizeText(value, maxLength) {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  if (!trimmed || trimmed.length > maxLength) return null;
  return trimmed;
}

function hashPassword(password, salt = crypto.randomBytes(16).toString('hex')) {
  const hash = crypto.pbkdf2Sync(password, salt, 120000, 32, 'sha256').toString('hex');
  return { salt, hash };
}

function verifyPassword(password, salt, expectedHash) {
  const { hash } = hashPassword(password, salt);
  return crypto.timingSafeEqual(Buffer.from(hash, 'hex'), Buffer.from(expectedHash, 'hex'));
}

function signToken(user) {
  const nonce = crypto.randomBytes(24).toString('hex');
  const expiresAt = Date.now() + TOKEN_TTL_MS;
  const payload = `${user.id}.${expiresAt}.${nonce}`;
  const signature = crypto.createHmac('sha256', SESSION_SECRET).update(payload).digest('hex');
  const token = `${payload}.${signature}`;
  sessions.set(token, {
    user: { id: user.id, username: user.username, role: user.role },
    expiresAt
  });
  return { token, expiresAt: new Date(expiresAt).toISOString() };
}

function run(sql, params = []) {
  const stmt = db.prepare(sql);
  try {
    stmt.run(params);
  } finally {
    stmt.free();
  }
}

function all(sql, params = []) {
  const stmt = db.prepare(sql);
  const rows = [];
  try {
    stmt.bind(params);
    while (stmt.step()) rows.push(stmt.getAsObject());
    return rows;
  } finally {
    stmt.free();
  }
}

function one(sql, params = []) {
  return all(sql, params)[0] || null;
}

function tableColumns(tableName) {
  return all(`PRAGMA table_info(${tableName})`).map(row => row.name);
}

function addColumnIfMissing(tableName, columnName, definition) {
  if (!tableColumns(tableName).includes(columnName)) {
    run(`ALTER TABLE ${tableName} ADD COLUMN ${columnName} ${definition}`);
  }
}

function saveDatabase() {
  fs.mkdirSync(path.dirname(DB_FILE), { recursive: true });
  fs.writeFileSync(DB_FILE, Buffer.from(db.export()));
}

function upsertDemoUser(username, password, role) {
  const { salt, hash } = hashPassword(password);
  const existing = one('SELECT id, role FROM users WHERE username = ?', [username]);
  if (existing) {
    run(
      'UPDATE users SET password_hash = ?, salt = ?, role = ? WHERE id = ?',
      [hash, salt, role, existing.id]
    );
    return existing.id;
  }
  run(
    'INSERT INTO users (username, password_hash, salt, role, created_at) VALUES (?, ?, ?, ?, ?)',
    [username, hash, salt, role, nowIso()]
  );
  return one('SELECT last_insert_rowid() AS id').id;
}

function seedDemoUsers() {
  const legacyDemo = one('SELECT id FROM users WHERE username = ?', ['demo']);
  const admin = one('SELECT id FROM users WHERE username = ?', [DEMO_ADMIN_USERNAME]);
  if (legacyDemo && !admin) {
    run(
      'UPDATE users SET username = ?, role = ? WHERE id = ?',
      [DEMO_ADMIN_USERNAME, 'admin', legacyDemo.id]
    );
  }

  const adminId = upsertDemoUser(DEMO_ADMIN_USERNAME, DEMO_PASSWORD, 'admin');
  const technicianId = upsertDemoUser(DEMO_USER_USERNAME, DEMO_PASSWORD, 'technician');
  audit('system', 'seed demo admin', 'user', adminId);
  audit('system', 'seed demo technician', 'user', technicianId);
}

async function initDatabase() {
  const SQL = await initSqlJs({
    locateFile: file => path.join(ROOT, 'node_modules', 'sql.js', 'dist', file)
  });
  db = fs.existsSync(DB_FILE)
    ? new SQL.Database(fs.readFileSync(DB_FILE))
    : new SQL.Database();

  run(`
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT NOT NULL UNIQUE,
      password_hash TEXT NOT NULL,
      salt TEXT NOT NULL,
      role TEXT NOT NULL DEFAULT 'technician',
      created_at TEXT NOT NULL
    )
  `);
  run(`
    CREATE TABLE IF NOT EXISTS maintenance_notes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      vehicle_id INTEGER NOT NULL,
      risk_class INTEGER NOT NULL CHECK (risk_class BETWEEN 0 AND 4),
      status TEXT NOT NULL CHECK (status IN ('open', 'scheduled', 'resolved')),
      note TEXT NOT NULL,
      created_by INTEGER NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      deleted_at TEXT,
      FOREIGN KEY (created_by) REFERENCES users(id)
    )
  `);
  addColumnIfMissing('maintenance_notes', 'deleted_at', 'TEXT');
  run(`
    CREATE TABLE IF NOT EXISTS maintenance_note_history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      note_id INTEGER NOT NULL,
      action TEXT NOT NULL,
      previous_status TEXT,
      new_status TEXT,
      previous_note TEXT,
      new_note TEXT,
      actor TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
  `);
  run(`
    CREATE TABLE IF NOT EXISTS app_vehicles (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      vehicle_key TEXT NOT NULL UNIQUE,
      vehicle_id INTEGER NOT NULL,
      split TEXT NOT NULL,
      risk_class INTEGER CHECK (risk_class BETWEEN 0 AND 4),
      timeline_kind TEXT NOT NULL,
      latest_time_step REAL,
      active INTEGER NOT NULL DEFAULT 1,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
  `);
  run(`
    CREATE TABLE IF NOT EXISTS telemetry_records (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      vehicle_db_id INTEGER NOT NULL,
      time_step REAL NOT NULL,
      source TEXT NOT NULL DEFAULT 'dashboard_seed',
      counter_100_0 REAL,
      counter_171_0 REAL,
      counter_309_0 REAL,
      counter_370_0 REAL,
      counter_427_0 REAL,
      counter_666_0 REAL,
      counter_835_0 REAL,
      counter_837_0 REAL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      deleted_at TEXT,
      FOREIGN KEY (vehicle_db_id) REFERENCES app_vehicles(id)
    )
  `);
  addColumnIfMissing('telemetry_records', 'deleted_at', 'TEXT');
  run(`
    CREATE TABLE IF NOT EXISTS dashboard_configuration (
      id INTEGER PRIMARY KEY CHECK (id = 1),
      payload_json TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
  `);
  run(`
    CREATE TABLE IF NOT EXISTS dashboard_vehicle_metadata (
      vehicle_db_id INTEGER PRIMARY KEY,
      payload_json TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      FOREIGN KEY (vehicle_db_id) REFERENCES app_vehicles(id)
    )
  `);
  run(`
    CREATE TABLE IF NOT EXISTS audit_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      actor TEXT NOT NULL,
      action TEXT NOT NULL,
      entity_type TEXT NOT NULL,
      entity_id INTEGER,
      created_at TEXT NOT NULL
    )
  `);

  seedDemoUsers();

  seedVehicleAndTelemetryTables();
  saveDatabase();
}

function seedVehicleAndTelemetryTables() {
  const dashboardFile = path.join(DATA_DIR, 'dashboard_data.json');
  if (!fs.existsSync(dashboardFile)) return;

  const data = JSON.parse(fs.readFileSync(dashboardFile, 'utf8'));
  const timestamp = nowIso();
  const { vehicles, ...configuration } = data;
  run(
    'INSERT OR REPLACE INTO dashboard_configuration (id, payload_json, updated_at) VALUES (1, ?, ?)',
    [JSON.stringify(configuration), timestamp]
  );

  for (const vehicle of data.vehicles || []) {
    let storedVehicle = one('SELECT id FROM app_vehicles WHERE vehicle_key = ?', [vehicle.key]);
    if (!storedVehicle) {
      run(
        `INSERT INTO app_vehicles
          (vehicle_key, vehicle_id, split, risk_class, timeline_kind, latest_time_step, active, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)`,
        [
          vehicle.key,
          vehicle.vehicle_id,
          vehicle.split,
          Number.isFinite(vehicle.riskClass) ? vehicle.riskClass : null,
          vehicle.timelineKind || 'unknown',
          vehicle.latest?.t ?? null,
          timestamp,
          timestamp
        ]
      );
      storedVehicle = one('SELECT last_insert_rowid() AS id');
      for (const point of vehicle.series || []) {
        const values = COUNTER_COLUMNS.map(([counterName]) => point.counters?.[counterName] ?? null);
        run(
          `INSERT INTO telemetry_records
            (vehicle_db_id, time_step, source, ${COUNTER_COLUMNS.map(([, column]) => column).join(', ')}, created_at, updated_at)
           VALUES (?, ?, ?, ${COUNTER_COLUMNS.map(() => '?').join(', ')}, ?, ?)`,
          [
            storedVehicle.id,
            point.t,
            point.source || 'dashboard_seed',
            ...values,
            timestamp,
            timestamp
          ]
        );
      }
    }

    const metadata = {
      specs: vehicle.specs || {},
      label: vehicle.label ?? null,
      repair: vehicle.repair ?? null,
      latestHistograms: vehicle.latestHistograms || {},
      failure: vehicle.failure || null
    };
    run(
      'INSERT OR REPLACE INTO dashboard_vehicle_metadata (vehicle_db_id, payload_json, updated_at) VALUES (?, ?, ?)',
      [storedVehicle.id, JSON.stringify(metadata), timestamp]
    );
  }
  audit('system', 'seed dashboard vehicles and telemetry', 'app_vehicles');
}

function securityHeaders(req, res, next) {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('Referrer-Policy', 'same-origin');
  res.setHeader('Permissions-Policy', 'geolocation=(), microphone=(), camera=()');
  res.setHeader('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; base-uri 'self'; form-action 'self'");
  next();
}

function requireAuth(req, res, next) {
  const header = req.get('authorization') || '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : '';
  const session = sessions.get(token);
  if (!session || session.expiresAt < Date.now()) {
    if (token) sessions.delete(token);
    res.status(401).json({ error: 'Authentication required.' });
    return;
  }
  req.user = session.user;
  next();
}

function requireAdmin(req, res, next) {
  if (req.user?.role !== 'admin') {
    res.status(403).json({ error: 'Admin role required.' });
    return;
  }
  next();
}

function validateNotePayload(body, partial = false) {
  const errors = [];
  const payload = {};

  if (!partial || body.vehicle_id !== undefined) {
    const vehicleId = sanitizeInt(body.vehicle_id, { min: 1 });
    if (vehicleId === null) errors.push('vehicle_id must be a positive integer.');
    else payload.vehicle_id = vehicleId;
  }

  if (!partial || body.risk_class !== undefined) {
    const riskClass = sanitizeInt(body.risk_class, { min: 0, max: 4 });
    if (riskClass === null) errors.push('risk_class must be an integer from 0 to 4.');
    else payload.risk_class = riskClass;
  }

  if (!partial || body.status !== undefined) {
    const status = sanitizeText(body.status, 24);
    if (!['open', 'scheduled', 'resolved'].includes(status)) errors.push('status must be open, scheduled, or resolved.');
    else payload.status = status;
  }

  if (!partial || body.note !== undefined) {
    const note = sanitizeText(body.note, 500);
    if (!note) errors.push('note is required and must be 500 characters or fewer.');
    else payload.note = note;
  }

  return { errors, payload };
}

function validateVehiclePayload(body, partial = false) {
  const errors = [];
  const payload = {};

  if (!partial || body.vehicle_id !== undefined) {
    const vehicleId = sanitizeInt(body.vehicle_id, { min: 1 });
    if (vehicleId === null) errors.push('vehicle_id must be a positive integer.');
    else payload.vehicle_id = vehicleId;
  }

  if (!partial || body.split !== undefined) {
    const split = sanitizeText(body.split, 24);
    if (!['train', 'validation', 'test', 'manual'].includes(split)) errors.push('split must be train, validation, test, or manual.');
    else payload.split = split;
  }

  if (!partial || body.risk_class !== undefined) {
    const riskClass = sanitizeInt(body.risk_class, { min: 0, max: 4 });
    if (riskClass === null) errors.push('risk_class must be an integer from 0 to 4.');
    else payload.risk_class = riskClass;
  }

  if (!partial || body.timeline_kind !== undefined) {
    const timelineKind = sanitizeText(body.timeline_kind, 60);
    if (!timelineKind) errors.push('timeline_kind is required.');
    else payload.timeline_kind = timelineKind;
  }

  if (body.latest_time_step !== undefined) {
    const latest = Number(body.latest_time_step);
    if (!Number.isFinite(latest) || latest < 0) errors.push('latest_time_step must be a non-negative number.');
    else payload.latest_time_step = latest;
  }

  return { errors, payload };
}

function validateTelemetryPayload(body, partial = false) {
  const errors = [];
  const payload = {};

  if (!partial || body.vehicle_db_id !== undefined) {
    const vehicleDbId = sanitizeInt(body.vehicle_db_id, { min: 1 });
    if (vehicleDbId === null) errors.push('vehicle_db_id must be a positive integer.');
    else payload.vehicle_db_id = vehicleDbId;
  }

  if (!partial || body.time_step !== undefined) {
    const timeStep = Number(body.time_step);
    if (!Number.isFinite(timeStep) || timeStep < 0) errors.push('time_step must be a non-negative number.');
    else payload.time_step = timeStep;
  }

  if (body.source !== undefined) {
    const source = sanitizeText(body.source, 60);
    if (!source) errors.push('source must be a short text value.');
    else payload.source = source;
  } else if (!partial) {
    payload.source = 'manual';
  }

  for (const [, column] of COUNTER_COLUMNS) {
    if (body[column] === undefined) continue;
    const value = Number(body[column]);
    if (!Number.isFinite(value) || value < 0) errors.push(`${column} must be a non-negative number.`);
    else payload[column] = value;
  }

  return { errors, payload };
}

function logNoteHistory(noteId, action, previous, next, actor) {
  run(
    `INSERT INTO maintenance_note_history
      (note_id, action, previous_status, new_status, previous_note, new_note, actor, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      noteId,
      action,
      previous?.status ?? null,
      next?.status ?? null,
      previous?.note ?? null,
      next?.note ?? null,
      actor,
      nowIso()
    ]
  );
}

function audit(actor, action, entityType, entityId = null) {
  run(
    'INSERT INTO audit_log (actor, action, entity_type, entity_id, created_at) VALUES (?, ?, ?, ?, ?)',
    [actor, action, entityType, entityId, nowIso()]
  );
}

function readJsonFile(fileName) {
  return JSON.parse(fs.readFileSync(path.join(DATA_DIR, fileName), 'utf8'));
}

function refreshVehicleLatestTimeStep(vehicleDbId) {
  if (!vehicleDbId) return;
  const latest = one(
    'SELECT MAX(time_step) AS latest_time_step FROM telemetry_records WHERE vehicle_db_id = ? AND deleted_at IS NULL',
    [vehicleDbId]
  );
  run(
    'UPDATE app_vehicles SET latest_time_step = ?, updated_at = ? WHERE id = ?',
    [latest?.latest_time_step ?? null, nowIso(), vehicleDbId]
  );
}

function dashboardDataPayload() {
  const configurationRow = one('SELECT payload_json FROM dashboard_configuration WHERE id = 1');
  const configuration = configurationRow ? JSON.parse(configurationRow.payload_json) : {};
  const vehicleRows = all(
    `SELECT id, vehicle_key, vehicle_id, split, risk_class, timeline_kind, latest_time_step
     FROM app_vehicles
     WHERE active = 1
     ORDER BY split, vehicle_id`
  );
  const metadataRows = all('SELECT vehicle_db_id, payload_json FROM dashboard_vehicle_metadata');
  const metadataByVehicle = new Map(
    metadataRows.map(row => [Number(row.vehicle_db_id), JSON.parse(row.payload_json)])
  );
  const telemetryRows = all(
    `SELECT id, vehicle_db_id, time_step, source,
      ${COUNTER_COLUMNS.map(([, column]) => column).join(', ')}, created_at
     FROM telemetry_records
     WHERE deleted_at IS NULL
     ORDER BY vehicle_db_id, time_step, id`
  );
  const seriesByVehicle = new Map();
  for (const row of telemetryRows) {
    const vehicleDbId = Number(row.vehicle_db_id);
    const series = seriesByVehicle.get(vehicleDbId) || [];
    const counters = Object.fromEntries(
      COUNTER_COLUMNS.map(([counterName, column]) => [counterName, row[column] ?? 0])
    );
    series.push({
      id: Number(row.id),
      t: Number(row.time_step),
      counters,
      source: row.source,
      receivedAt: row.created_at
    });
    seriesByVehicle.set(vehicleDbId, series);
  }

  const vehicles = vehicleRows.map(row => {
    const metadata = metadataByVehicle.get(Number(row.id)) || {};
    const series = seriesByVehicle.get(Number(row.id)) || [];
    const latest = series[series.length - 1] || null;
    return {
      key: row.vehicle_key,
      dbId: Number(row.id),
      vehicle_id: Number(row.vehicle_id),
      split: row.split,
      specs: metadata.specs || {},
      label: metadata.label ?? null,
      repair: metadata.repair ?? null,
      riskClass: row.risk_class === null ? null : Number(row.risk_class),
      timelineKind: row.timeline_kind,
      series,
      latest: latest ? { t: latest.t, counters: latest.counters, source: latest.source } : null,
      latestHistograms: metadata.latestHistograms || {},
      failure: metadata.failure || null
    };
  });

  return { ...configuration, vehicles };
}

function modelSummaryPayload() {
  const model = readJsonFile('model_output.json');
  const comparison = model.modelComparison || [];
  const recommendedModelName = model.modelSelection?.recommendedModel || model.modelName;
  const recommendedModel = comparison.find(row => row.name === recommendedModelName) || null;
  return {
    generatedAt: model.generatedAt,
    modelName: recommendedModelName,
    modelType: recommendedModel?.type || model.modelType,
    liveModelName: model.modelName,
    liveModelType: model.modelType,
    modelSelection: model.modelSelection || null,
    recommendedModel,
    selectedK: model.knn?.k,
    selectionMetric: model.knnTuning?.selectionMetric,
    comparison,
    supplementalComparison: model.supplementalModelComparison || [],
    tabularModelExperiment: model.tabularModelExperiment || null,
    knnTuning: model.knnTuning || null,
    validation: recommendedModel?.validation || model.evaluation?.validation || null,
    test: recommendedModel?.test || model.evaluation?.test || null,
    topFeatures: model.topFeaturesClass0VsClass4 || []
  };
}

async function main() {
  await initDatabase();

  const app = express();
  app.disable('x-powered-by');
  app.use(securityHeaders);
  app.use(express.json({ limit: '100kb' }));

  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', database: path.relative(ROOT, DB_FILE), generatedAt: nowIso() });
  });

  app.get('/api/dashboard/summary', (req, res) => {
    const data = dashboardDataPayload();
    const model = modelSummaryPayload();
    res.json({
      vehicles: data.vehicles.length,
      rawDataset: data.rawDataset,
      modelName: model.modelName,
      liveModelName: model.liveModelName,
      selectedK: model.selectedK,
      validationCost: model.validation?.totalCost,
      testCost: model.test?.totalCost
    });
  });

  app.get('/api/dashboard/data', (req, res) => {
    res.json(dashboardDataPayload());
  });

  app.get('/api/dashboard/model-output', (req, res) => {
    res.json(readJsonFile('model_output.json'));
  });

  app.get('/api/dashboard/lightgbm-model', (req, res) => {
    res.json(readJsonFile('lightgbm_model.json'));
  });

  app.get('/api/ml/summary', requireAuth, requireAdmin, (req, res) => {
    res.json(modelSummaryPayload());
  });

  app.post('/api/ml/run-analysis', requireAuth, requireAdmin, (req, res) => {
    if (analysisRunning) {
      res.status(409).json({ error: 'ML analysis is already running.' });
      return;
    }

    analysisRunning = true;
    execFile(process.execPath, [path.join(ROOT, 'tools', 'train_baseline_model.js')], {
      cwd: ROOT,
      timeout: 180000,
      maxBuffer: 1024 * 1024 * 8
    }, (error, stdout, stderr) => {
      analysisRunning = false;
      if (error) {
        res.status(500).json({ error: 'ML analysis failed.', details: stderr || error.message });
        return;
      }
      audit(req.user.username, 'run analysis', 'ml_model');
      saveDatabase();
      res.json({
        completedAt: nowIso(),
        output: stdout.trim().split(/\r?\n/).slice(-12),
        summary: modelSummaryPayload()
      });
    });
  });

  app.get('/api/vehicles', (req, res) => {
    const search = String(req.query.search || '').trim();
    const split = String(req.query.split || 'all');
    const risk = String(req.query.risk || 'all');
    const limit = Math.min(sanitizeInt(req.query.limit || 25, { min: 1, max: 100 }) || 25, 100);
    const conditions = ['active = 1'];
    const params = [];
    if (search) {
      conditions.push('CAST(vehicle_id AS TEXT) LIKE ?');
      params.push(`%${search}%`);
    }
    if (split !== 'all') {
      conditions.push('split = ?');
      params.push(split);
    }
    if (risk !== 'all') {
      const parsedRisk = sanitizeInt(risk, { min: 0, max: 4 });
      if (parsedRisk === null) {
        res.status(400).json({ error: 'risk must be all or a class from 0 to 4.' });
        return;
      }
      conditions.push('risk_class = ?');
      params.push(parsedRisk);
    }
    const vehicles = all(
      `SELECT vehicle_key AS key, vehicle_id, split, risk_class AS riskClass,
        latest_time_step AS latestTimeStep, timeline_kind AS timelineKind
       FROM app_vehicles
       WHERE ${conditions.join(' AND ')}
       ORDER BY risk_class DESC, latest_time_step DESC
       LIMIT ?`,
      [...params, limit]
    );
    res.json({ vehicles });
  });

  app.get('/api/app-vehicles', requireAuth, requireAdmin, (req, res) => {
    const search = String(req.query.search || '').trim();
    const split = String(req.query.split || 'all');
    const risk = String(req.query.risk || 'all');
    const limit = Math.min(sanitizeInt(req.query.limit || 100, { min: 1, max: 500 }) || 100, 500);
    const offset = sanitizeInt(req.query.offset || 0, { min: 0, max: 1000000 }) || 0;
    const conditions = ['active = 1'];
    const params = [];
    if (search) {
      conditions.push('CAST(vehicle_id AS TEXT) LIKE ?');
      params.push(`%${search}%`);
    }
    if (split !== 'all') {
      conditions.push('split = ?');
      params.push(split);
    }
    if (risk !== 'all') {
      conditions.push('risk_class = ?');
      params.push(sanitizeInt(risk, { min: 0, max: 4 }));
    }
    const where = conditions.join(' AND ');
    const rows = all(
      `SELECT id, vehicle_key, vehicle_id, split, risk_class, timeline_kind, latest_time_step, updated_at
       FROM app_vehicles
       WHERE ${where}
       ORDER BY risk_class DESC, latest_time_step DESC
       LIMIT ? OFFSET ?`,
      [...params, limit, offset]
    );
    const total = one(`SELECT COUNT(*) AS count FROM app_vehicles WHERE ${where}`, params);
    res.json({ vehicles: rows, total: Number(total?.count || 0) });
  });

  app.post('/api/app-vehicles', requireAuth, requireAdmin, (req, res) => {
    const { errors, payload } = validateVehiclePayload(req.body);
    if (errors.length) {
      res.status(400).json({ error: errors.join(' ') });
      return;
    }
    const vehicleKey = `${payload.split}:${payload.vehicle_id}`;
    if (one('SELECT id FROM app_vehicles WHERE vehicle_key = ? AND active = 1', [vehicleKey])) {
      res.status(409).json({ error: 'That active vehicle already exists.' });
      return;
    }
    const timestamp = nowIso();
    run(
      `INSERT INTO app_vehicles
        (vehicle_key, vehicle_id, split, risk_class, timeline_kind, latest_time_step, active, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)`,
      [vehicleKey, payload.vehicle_id, payload.split, payload.risk_class, payload.timeline_kind, payload.latest_time_step ?? null, timestamp, timestamp]
    );
    const created = one('SELECT last_insert_rowid() AS id');
    audit(req.user.username, 'create', 'app_vehicle', created.id);
    saveDatabase();
    res.status(201).json({ vehicle: one('SELECT * FROM app_vehicles WHERE id = ?', [created.id]) });
  });

  app.put('/api/app-vehicles/:id', requireAuth, requireAdmin, (req, res) => {
    const id = sanitizeInt(req.params.id, { min: 1 });
    if (id === null) {
      res.status(400).json({ error: 'Invalid vehicle id.' });
      return;
    }
    const current = one('SELECT * FROM app_vehicles WHERE id = ? AND active = 1', [id]);
    if (!current) {
      res.status(404).json({ error: 'Vehicle not found.' });
      return;
    }
    const { errors, payload } = validateVehiclePayload(req.body, true);
    if (errors.length) {
      res.status(400).json({ error: errors.join(' ') });
      return;
    }
    const updated = {
      vehicle_id: payload.vehicle_id ?? current.vehicle_id,
      split: payload.split ?? current.split,
      risk_class: payload.risk_class ?? current.risk_class,
      timeline_kind: payload.timeline_kind ?? current.timeline_kind,
      latest_time_step: payload.latest_time_step ?? current.latest_time_step
    };
    const vehicleKey = `${updated.split}:${updated.vehicle_id}`;
    const duplicate = one('SELECT id FROM app_vehicles WHERE vehicle_key = ? AND id <> ? AND active = 1', [vehicleKey, id]);
    if (duplicate) {
      res.status(409).json({ error: 'Another active vehicle already uses that split and vehicle_id.' });
      return;
    }
    run(
      `UPDATE app_vehicles
       SET vehicle_key = ?, vehicle_id = ?, split = ?, risk_class = ?, timeline_kind = ?, latest_time_step = ?, updated_at = ?
       WHERE id = ?`,
      [vehicleKey, updated.vehicle_id, updated.split, updated.risk_class, updated.timeline_kind, updated.latest_time_step, nowIso(), id]
    );
    audit(req.user.username, 'update', 'app_vehicle', id);
    saveDatabase();
    res.json({ vehicle: one('SELECT * FROM app_vehicles WHERE id = ?', [id]) });
  });

  app.delete('/api/app-vehicles/:id', requireAuth, requireAdmin, (req, res) => {
    const id = sanitizeInt(req.params.id, { min: 1 });
    if (id === null) {
      res.status(400).json({ error: 'Invalid vehicle id.' });
      return;
    }
    const current = one('SELECT * FROM app_vehicles WHERE id = ? AND active = 1', [id]);
    if (!current) {
      res.status(404).json({ error: 'Vehicle not found.' });
      return;
    }
    const telemetryCount = Number(one('SELECT COUNT(*) AS count FROM telemetry_records WHERE vehicle_db_id = ?', [id])?.count || 0);
    const noteCount = Number(one('SELECT COUNT(*) AS count FROM maintenance_notes WHERE vehicle_id = ? AND deleted_at IS NULL', [current.vehicle_id])?.count || 0);
    if (telemetryCount || noteCount) {
      res.status(409).json({
        error: 'Vehicle cannot be deleted because telemetry records or maintenance logs still reference it.',
        telemetryCount,
        noteCount
      });
      return;
    }
    run('UPDATE app_vehicles SET active = 0, updated_at = ? WHERE id = ?', [nowIso(), id]);
    audit(req.user.username, 'archive', 'app_vehicle', id);
    saveDatabase();
    res.json({ ok: true });
  });

  app.get('/api/telemetry-records', requireAuth, requireAdmin, (req, res) => {
    const vehicleId = req.query.vehicle_id ? sanitizeInt(req.query.vehicle_id, { min: 1 }) : null;
    const risk = String(req.query.risk || 'all');
    const limit = Math.min(sanitizeInt(req.query.limit || 100, { min: 1, max: 500 }) || 100, 500);
    const offset = sanitizeInt(req.query.offset || 0, { min: 0, max: 1000000 }) || 0;
    const conditions = ['v.active = 1', 't.deleted_at IS NULL'];
    const params = [];
    if (vehicleId) {
      conditions.push('v.vehicle_id = ?');
      params.push(vehicleId);
    }
    if (risk !== 'all') {
      conditions.push('v.risk_class = ?');
      params.push(sanitizeInt(risk, { min: 0, max: 4 }));
    }
    const where = conditions.join(' AND ');
    const rows = all(
      `SELECT t.id, t.vehicle_db_id, v.vehicle_id, v.split, v.risk_class, t.time_step, t.source,
        ${COUNTER_COLUMNS.map(([, column]) => `t.${column}`).join(', ')}, t.updated_at
       FROM telemetry_records t
       JOIN app_vehicles v ON v.id = t.vehicle_db_id
       WHERE ${where}
       ORDER BY v.vehicle_id ASC, t.time_step DESC
       LIMIT ? OFFSET ?`,
      [...params, limit, offset]
    );
    const total = one(
      `SELECT COUNT(*) AS count
       FROM telemetry_records t
       JOIN app_vehicles v ON v.id = t.vehicle_db_id
       WHERE ${where}`,
      params
    );
    res.json({ telemetry: rows, total: Number(total?.count || 0) });
  });

  app.post('/api/telemetry-records', requireAuth, requireAdmin, (req, res) => {
    const { errors, payload } = validateTelemetryPayload(req.body);
    if (errors.length) {
      res.status(400).json({ error: errors.join(' ') });
      return;
    }
    if (!one('SELECT id FROM app_vehicles WHERE id = ? AND active = 1', [payload.vehicle_db_id])) {
      res.status(404).json({ error: 'Vehicle not found for telemetry record.' });
      return;
    }
    const timestamp = nowIso();
    run(
      `INSERT INTO telemetry_records
        (vehicle_db_id, time_step, source, ${COUNTER_COLUMNS.map(([, column]) => column).join(', ')}, created_at, updated_at)
       VALUES (?, ?, ?, ${COUNTER_COLUMNS.map(() => '?').join(', ')}, ?, ?)`,
      [
        payload.vehicle_db_id,
        payload.time_step,
        payload.source || 'manual',
        ...COUNTER_COLUMNS.map(([, column]) => payload[column] ?? null),
        timestamp,
        timestamp
      ]
    );
    const created = one('SELECT last_insert_rowid() AS id');
    refreshVehicleLatestTimeStep(payload.vehicle_db_id);
    audit(req.user.username, 'create', 'telemetry_record', created.id);
    saveDatabase();
    res.status(201).json({ telemetry: one('SELECT * FROM telemetry_records WHERE id = ?', [created.id]) });
  });

  app.put('/api/telemetry-records/:id', requireAuth, requireAdmin, (req, res) => {
    const id = sanitizeInt(req.params.id, { min: 1 });
    if (id === null) {
      res.status(400).json({ error: 'Invalid telemetry id.' });
      return;
    }
    const current = one('SELECT * FROM telemetry_records WHERE id = ? AND deleted_at IS NULL', [id]);
    if (!current) {
      res.status(404).json({ error: 'Telemetry record not found.' });
      return;
    }
    const { errors, payload } = validateTelemetryPayload(req.body, true);
    if (errors.length) {
      res.status(400).json({ error: errors.join(' ') });
      return;
    }
    const vehicleDbId = payload.vehicle_db_id ?? current.vehicle_db_id;
    if (!one('SELECT id FROM app_vehicles WHERE id = ? AND active = 1', [vehicleDbId])) {
      res.status(404).json({ error: 'Vehicle not found for telemetry record.' });
      return;
    }
    const updatedValues = COUNTER_COLUMNS.map(([, column]) => payload[column] ?? current[column]);
    run(
      `UPDATE telemetry_records
       SET vehicle_db_id = ?, time_step = ?, source = ?, ${COUNTER_COLUMNS.map(([, column]) => `${column} = ?`).join(', ')}, updated_at = ?
       WHERE id = ?`,
      [
        vehicleDbId,
        payload.time_step ?? current.time_step,
        payload.source ?? current.source,
        ...updatedValues,
        nowIso(),
        id
      ]
    );
    refreshVehicleLatestTimeStep(current.vehicle_db_id);
    if (vehicleDbId !== current.vehicle_db_id) refreshVehicleLatestTimeStep(vehicleDbId);
    audit(req.user.username, 'update', 'telemetry_record', id);
    saveDatabase();
    res.json({ telemetry: one('SELECT * FROM telemetry_records WHERE id = ?', [id]) });
  });

  app.delete('/api/telemetry-records/:id', requireAuth, requireAdmin, (req, res) => {
    const id = sanitizeInt(req.params.id, { min: 1 });
    if (id === null) {
      res.status(400).json({ error: 'Invalid telemetry id.' });
      return;
    }
    const existing = one('SELECT id, vehicle_db_id FROM telemetry_records WHERE id = ? AND deleted_at IS NULL', [id]);
    if (!existing) {
      res.status(404).json({ error: 'Telemetry record not found.' });
      return;
    }
    run('UPDATE telemetry_records SET deleted_at = ?, updated_at = ? WHERE id = ?', [nowIso(), nowIso(), id]);
    refreshVehicleLatestTimeStep(existing.vehicle_db_id);
    audit(req.user.username, 'soft delete', 'telemetry_record', id);
    saveDatabase();
    res.json({ ok: true });
  });

  app.post('/api/auth/login', (req, res) => {
    const username = sanitizeText(req.body?.username, 80);
    const password = sanitizeText(req.body?.password, 200);
    if (!username || !password) {
      res.status(400).json({ error: 'Username and password are required.' });
      return;
    }

    const user = one('SELECT id, username, password_hash, salt, role FROM users WHERE username = ?', [username]);
    if (!user || !verifyPassword(password, user.salt, user.password_hash)) {
      res.status(401).json({ error: 'Invalid username or password.' });
      return;
    }

    const session = signToken(user);
    audit(user.username, 'login', 'session');
    saveDatabase();
    res.json({
      token: session.token,
      expiresAt: session.expiresAt,
      user: { id: user.id, username: user.username, role: user.role }
    });
  });

  app.post('/api/auth/logout', requireAuth, (req, res) => {
    const token = (req.get('authorization') || '').slice(7);
    sessions.delete(token);
    audit(req.user.username, 'logout', 'session');
    saveDatabase();
    res.json({ ok: true });
  });

  app.get('/api/notes', requireAuth, (req, res) => {
    const rows = all(`
      SELECT n.id, n.vehicle_id, n.risk_class, n.status, n.note, n.created_at, n.updated_at, u.username AS created_by
      FROM maintenance_notes n
      JOIN users u ON u.id = n.created_by
      WHERE n.deleted_at IS NULL
      ORDER BY n.updated_at DESC, n.id DESC
      LIMIT 100
    `);
    res.json({ notes: rows });
  });

  app.post('/api/notes', requireAuth, (req, res) => {
    const { errors, payload } = validateNotePayload(req.body);
    if (errors.length) {
      res.status(400).json({ error: errors.join(' ') });
      return;
    }
    if (!one('SELECT id FROM app_vehicles WHERE vehicle_id = ? AND active = 1', [payload.vehicle_id])) {
      res.status(404).json({ error: 'Vehicle must exist before creating a maintenance log.' });
      return;
    }

    const timestamp = nowIso();
    run(
      'INSERT INTO maintenance_notes (vehicle_id, risk_class, status, note, created_by, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
      [payload.vehicle_id, payload.risk_class, payload.status, payload.note, req.user.id, timestamp, timestamp]
    );
    const created = one('SELECT last_insert_rowid() AS id');
    logNoteHistory(created.id, 'create', null, { status: payload.status, note: payload.note }, req.user.username);
    audit(req.user.username, 'create', 'maintenance_note', created.id);
    saveDatabase();
    res.status(201).json({ note: one('SELECT * FROM maintenance_notes WHERE id = ?', [created.id]) });
  });

  app.put('/api/notes/:id', requireAuth, (req, res) => {
    const id = sanitizeInt(req.params.id, { min: 1 });
    if (id === null) {
      res.status(400).json({ error: 'Invalid note id.' });
      return;
    }
    const existing = one('SELECT id FROM maintenance_notes WHERE id = ? AND deleted_at IS NULL', [id]);
    if (!existing) {
      res.status(404).json({ error: 'Maintenance note not found.' });
      return;
    }

    const { errors, payload } = validateNotePayload(req.body, true);
    if (errors.length) {
      res.status(400).json({ error: errors.join(' ') });
      return;
    }

    const current = one('SELECT * FROM maintenance_notes WHERE id = ? AND deleted_at IS NULL', [id]);
    const updated = {
      vehicle_id: payload.vehicle_id ?? current.vehicle_id,
      risk_class: payload.risk_class ?? current.risk_class,
      status: payload.status ?? current.status,
      note: payload.note ?? current.note
    };
    if (!one('SELECT id FROM app_vehicles WHERE vehicle_id = ? AND active = 1', [updated.vehicle_id])) {
      res.status(404).json({ error: 'Vehicle must exist before updating a maintenance log.' });
      return;
    }
    run(
      'UPDATE maintenance_notes SET vehicle_id = ?, risk_class = ?, status = ?, note = ?, updated_at = ? WHERE id = ?',
      [updated.vehicle_id, updated.risk_class, updated.status, updated.note, nowIso(), id]
    );
    logNoteHistory(id, 'update', current, updated, req.user.username);
    audit(req.user.username, 'update', 'maintenance_note', id);
    saveDatabase();
    res.json({ note: one('SELECT * FROM maintenance_notes WHERE id = ?', [id]) });
  });

  app.delete('/api/notes/:id', requireAuth, (req, res) => {
    const id = sanitizeInt(req.params.id, { min: 1 });
    if (id === null) {
      res.status(400).json({ error: 'Invalid note id.' });
      return;
    }
    const existing = one('SELECT * FROM maintenance_notes WHERE id = ? AND deleted_at IS NULL', [id]);
    if (!existing) {
      res.status(404).json({ error: 'Maintenance note not found.' });
      return;
    }
    run('UPDATE maintenance_notes SET deleted_at = ?, updated_at = ? WHERE id = ?', [nowIso(), nowIso(), id]);
    logNoteHistory(id, 'delete', existing, { status: existing.status, note: existing.note }, req.user.username);
    audit(req.user.username, 'soft delete', 'maintenance_note', id);
    saveDatabase();
    res.json({ ok: true });
  });

  app.get('/api/notes/history', requireAuth, (req, res) => {
    const noteId = req.query.note_id ? sanitizeInt(req.query.note_id, { min: 1 }) : null;
    const params = [];
    const where = noteId ? 'WHERE note_id = ?' : '';
    if (noteId) params.push(noteId);
    const rows = all(
      `SELECT id, note_id, action, previous_status, new_status, previous_note, new_note, actor, created_at
       FROM maintenance_note_history
       ${where}
       ORDER BY created_at DESC, id DESC
       LIMIT 200`,
      params
    );
    res.json({ history: rows });
  });

  app.use(express.static(SITE_DIR, {
    extensions: ['html'],
    maxAge: process.env.NODE_ENV === 'production' ? '1h' : 0
  }));

  app.use((req, res) => {
    res.status(404).json({ error: 'Route not found.' });
  });

  app.listen(PORT, () => {
    console.log(`Full-stack dashboard running at http://localhost:${PORT}`);
    console.log(`Demo admin: ${DEMO_ADMIN_USERNAME} / ${DEMO_PASSWORD}`);
    console.log(`Demo technician: ${DEMO_USER_USERNAME} / ${DEMO_PASSWORD}`);
    console.log(`SQLite database: ${DB_FILE}`);
  });
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
