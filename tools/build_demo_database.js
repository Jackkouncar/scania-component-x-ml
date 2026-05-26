const fs = require('fs');
const path = require('path');
const readline = require('readline');
const initSqlJs = require('sql.js');

const ROOT = path.resolve(__dirname, '..');
const DATA_DIR = path.join(ROOT, '2024-34-2', '2024-34-2', 'data');
const OUT_DIR = path.join(ROOT, 'database');
const OUT_FILE = path.join(OUT_DIR, 'scania_pdm_demo.sqlite');

const SPLITS = [
  {
    id: 1,
    name: 'train',
    specs: 'train_specifications.csv',
    operational: 'train_operational_readouts.csv',
    target: 'train_tte.csv'
  },
  {
    id: 2,
    name: 'validation',
    specs: 'validation_specifications.csv',
    operational: 'validation_operational_readouts.csv',
    labels: 'validation_labels.csv'
  },
  {
    id: 3,
    name: 'test',
    specs: 'test_specifications.csv',
    operational: 'test_operational_readouts.csv',
    labels: 'test_labels.csv'
  }
];

const TRAIN_HEALTHY_LIMIT = Number(process.env.TRAIN_HEALTHY_LIMIT || 30);
const TRAIN_REPAIRED_LIMIT = Number(process.env.TRAIN_REPAIRED_LIMIT || 30);
const EVAL_CLASS0_LIMIT = Number(process.env.EVAL_CLASS0_LIMIT || 30);
const EVAL_POSITIVE_LIMIT = Number(process.env.EVAL_POSITIVE_LIMIT || 30);

function csvPath(file) {
  return path.join(DATA_DIR, file);
}

function parseCsvLine(line) {
  return line.split(',');
}

async function readFirstLine(file) {
  return new Promise((resolve, reject) => {
    const stream = fs.createReadStream(file, { encoding: 'utf8', highWaterMark: 65536 });
    let data = '';

    stream.on('data', chunk => {
      data += chunk;
      const newlineIndex = data.indexOf('\n');
      if (newlineIndex >= 0) {
        stream.destroy();
        resolve(data.slice(0, newlineIndex).replace(/\r$/, ''));
      }
    });

    stream.on('error', reject);
    stream.on('end', () => resolve(data.replace(/\r$/, '')));
  });
}

async function readSmallCsv(file) {
  const rows = [];
  const rl = readline.createInterface({
    input: fs.createReadStream(csvPath(file)),
    crlfDelay: Infinity
  });

  let header = null;
  for await (const line of rl) {
    if (!header) {
      header = parseCsvLine(line);
      continue;
    }
    if (!line) continue;

    const cells = parseCsvLine(line);
    const row = {};
    for (let i = 0; i < header.length; i++) {
      row[header[i]] = cells[i] ?? '';
    }
    rows.push(row);
  }

  return rows;
}

function pickByColumn(rows, column, targets) {
  const picked = new Set();
  const seen = {};

  for (const row of rows) {
    const value = row[column];
    const limit = targets[value];
    if (limit === undefined) continue;

    seen[value] = seen[value] || 0;
    if (seen[value] < limit) {
      picked.add(Number(row.vehicle_id));
      seen[value]++;
    }
  }

  return picked;
}

async function chooseVehicles() {
  const selected = new Map();

  const trainRows = await readSmallCsv('train_tte.csv');
  selected.set(
    'train',
    pickByColumn(trainRows, 'in_study_repair', {
      0: TRAIN_HEALTHY_LIMIT,
      1: TRAIN_REPAIRED_LIMIT
    })
  );

  for (const splitName of ['validation', 'test']) {
    const rows = await readSmallCsv(`${splitName}_labels.csv`);
    selected.set(
      splitName,
      pickByColumn(rows, 'class_label', {
        0: EVAL_CLASS0_LIMIT,
        1: EVAL_POSITIVE_LIMIT,
        2: EVAL_POSITIVE_LIMIT,
        3: EVAL_POSITIVE_LIMIT,
        4: EVAL_POSITIVE_LIMIT
      })
    );
  }

  return selected;
}

async function getOperationalFeatures() {
  const header = parseCsvLine(await readFirstLine(csvPath('train_operational_readouts.csv')));
  const features = header.slice(2).map((featureName, index) => {
    const [variableCode, binIndexText] = featureName.split('_');
    return {
      featureId: index + 1,
      featureName,
      variableCode,
      binIndex: Number(binIndexText)
    };
  });

  const variables = new Map();
  for (const feature of features) {
    const current = variables.get(feature.variableCode) || { variableCode: feature.variableCode, binCount: 0 };
    current.binCount++;
    variables.set(feature.variableCode, current);
  }

  for (const variable of variables.values()) {
    variable.variableKind = variable.binCount === 1 ? 'counter' : 'histogram';
  }

  return { features, variables: [...variables.values()] };
}

function createSchema(db) {
  db.run(`
    PRAGMA foreign_keys = ON;

    CREATE TABLE dataset_split (
      split_id INTEGER PRIMARY KEY,
      split_name TEXT NOT NULL UNIQUE
    );

    CREATE TABLE vehicle (
      vehicle_id INTEGER PRIMARY KEY,
      split_id INTEGER NOT NULL REFERENCES dataset_split(split_id)
    );

    CREATE TABLE spec_attribute (
      spec_attribute_id INTEGER PRIMARY KEY,
      spec_name TEXT NOT NULL UNIQUE
    );

    CREATE TABLE spec_category (
      spec_attribute_id INTEGER NOT NULL REFERENCES spec_attribute(spec_attribute_id),
      category_label TEXT NOT NULL,
      PRIMARY KEY (spec_attribute_id, category_label)
    );

    CREATE TABLE vehicle_specification (
      vehicle_id INTEGER NOT NULL REFERENCES vehicle(vehicle_id),
      spec_attribute_id INTEGER NOT NULL,
      category_label TEXT NOT NULL,
      PRIMARY KEY (vehicle_id, spec_attribute_id),
      FOREIGN KEY (spec_attribute_id, category_label)
        REFERENCES spec_category(spec_attribute_id, category_label)
    );

    CREATE TABLE operational_variable (
      variable_code TEXT PRIMARY KEY,
      variable_kind TEXT NOT NULL CHECK (variable_kind IN ('counter', 'histogram')),
      bin_count INTEGER NOT NULL
    );

    CREATE TABLE operational_feature (
      feature_id INTEGER PRIMARY KEY,
      variable_code TEXT NOT NULL REFERENCES operational_variable(variable_code),
      bin_index INTEGER NOT NULL,
      feature_name TEXT NOT NULL UNIQUE,
      UNIQUE (variable_code, bin_index)
    );

    CREATE TABLE readout (
      readout_id INTEGER PRIMARY KEY,
      vehicle_id INTEGER NOT NULL REFERENCES vehicle(vehicle_id),
      readout_seq INTEGER NOT NULL,
      time_step REAL NOT NULL,
      is_latest INTEGER NOT NULL DEFAULT 0 CHECK (is_latest IN (0, 1)),
      UNIQUE (vehicle_id, readout_seq)
    );

    CREATE TABLE readout_measurement (
      readout_id INTEGER NOT NULL REFERENCES readout(readout_id),
      feature_id INTEGER NOT NULL REFERENCES operational_feature(feature_id),
      measurement_value REAL,
      PRIMARY KEY (readout_id, feature_id)
    );

    CREATE TABLE repair_outcome (
      vehicle_id INTEGER PRIMARY KEY REFERENCES vehicle(vehicle_id),
      length_of_study_time_step REAL NOT NULL,
      in_study_repair INTEGER NOT NULL CHECK (in_study_repair IN (0, 1))
    );

    CREATE TABLE class_window (
      class_label INTEGER PRIMARY KEY,
      window_label TEXT NOT NULL,
      min_time_to_failure REAL,
      max_time_to_failure REAL
    );

    CREATE TABLE vehicle_label (
      vehicle_id INTEGER PRIMARY KEY REFERENCES vehicle(vehicle_id),
      class_label INTEGER NOT NULL REFERENCES class_window(class_label)
    );

    CREATE TABLE model_run (
      model_run_id INTEGER PRIMARY KEY,
      model_name TEXT NOT NULL,
      trained_at TEXT,
      notes TEXT
    );

    CREATE TABLE prediction (
      prediction_id INTEGER PRIMARY KEY,
      model_run_id INTEGER NOT NULL REFERENCES model_run(model_run_id),
      vehicle_id INTEGER NOT NULL REFERENCES vehicle(vehicle_id),
      predicted_class INTEGER NOT NULL REFERENCES class_window(class_label),
      failure_probability REAL,
      created_at TEXT
    );

    CREATE INDEX idx_vehicle_split ON vehicle(split_id);
    CREATE INDEX idx_readout_vehicle_latest ON readout(vehicle_id, is_latest);
    CREATE INDEX idx_measurement_feature ON readout_measurement(feature_id);
    CREATE INDEX idx_vehicle_label_class ON vehicle_label(class_label);
  `);
}

function insertReferenceData(db, operational) {
  const splitStmt = db.prepare('INSERT INTO dataset_split (split_id, split_name) VALUES (?, ?)');
  for (const split of SPLITS) splitStmt.run([split.id, split.name]);
  splitStmt.free();

  const classStmt = db.prepare(`
    INSERT INTO class_window
      (class_label, window_label, min_time_to_failure, max_time_to_failure)
    VALUES (?, ?, ?, ?)
  `);
  [
    [0, 'No imminent failure or more than 48 time units before failure', 48, null],
    [1, '48 to 24 time units before failure', 24, 48],
    [2, '24 to 12 time units before failure', 12, 24],
    [3, '12 to 6 time units before failure', 6, 12],
    [4, '6 to 0 time units before failure', 0, 6]
  ].forEach(row => classStmt.run(row));
  classStmt.free();

  const variableStmt = db.prepare(`
    INSERT INTO operational_variable (variable_code, variable_kind, bin_count)
    VALUES (?, ?, ?)
  `);
  for (const variable of operational.variables) {
    variableStmt.run([variable.variableCode, variable.variableKind, variable.binCount]);
  }
  variableStmt.free();

  const featureStmt = db.prepare(`
    INSERT INTO operational_feature (feature_id, variable_code, bin_index, feature_name)
    VALUES (?, ?, ?, ?)
  `);
  for (const feature of operational.features) {
    featureStmt.run([feature.featureId, feature.variableCode, feature.binIndex, feature.featureName]);
  }
  featureStmt.free();

  const specStmt = db.prepare('INSERT INTO spec_attribute (spec_attribute_id, spec_name) VALUES (?, ?)');
  for (let i = 0; i < 8; i++) {
    specStmt.run([i, `Spec_${i}`]);
  }
  specStmt.free();
}

async function insertSpecifications(db, selectedBySplit) {
  const vehicleStmt = db.prepare('INSERT OR IGNORE INTO vehicle (vehicle_id, split_id) VALUES (?, ?)');
  const categoryStmt = db.prepare(`
    INSERT OR IGNORE INTO spec_category (spec_attribute_id, category_label)
    VALUES (?, ?)
  `);
  const vehicleSpecStmt = db.prepare(`
    INSERT OR REPLACE INTO vehicle_specification (vehicle_id, spec_attribute_id, category_label)
    VALUES (?, ?, ?)
  `);

  for (const split of SPLITS) {
    const selected = selectedBySplit.get(split.name);
    const rl = readline.createInterface({
      input: fs.createReadStream(csvPath(split.specs)),
      crlfDelay: Infinity
    });

    let header = null;
    for await (const line of rl) {
      if (!header) {
        header = parseCsvLine(line);
        continue;
      }
      if (!line) continue;

      const cells = parseCsvLine(line);
      const vehicleId = Number(cells[0]);
      if (!selected.has(vehicleId)) continue;

      vehicleStmt.run([vehicleId, split.id]);

      for (let i = 1; i < header.length; i++) {
        const specAttributeId = i - 1;
        const categoryLabel = cells[i];
        categoryStmt.run([specAttributeId, categoryLabel]);
        vehicleSpecStmt.run([vehicleId, specAttributeId, categoryLabel]);
      }
    }
  }

  vehicleStmt.free();
  categoryStmt.free();
  vehicleSpecStmt.free();
}

async function insertTargets(db, selectedBySplit) {
  const repairStmt = db.prepare(`
    INSERT OR REPLACE INTO repair_outcome
      (vehicle_id, length_of_study_time_step, in_study_repair)
    VALUES (?, ?, ?)
  `);

  const trainRows = await readSmallCsv('train_tte.csv');
  const selectedTrain = selectedBySplit.get('train');
  for (const row of trainRows) {
    const vehicleId = Number(row.vehicle_id);
    if (selectedTrain.has(vehicleId)) {
      repairStmt.run([
        vehicleId,
        Number(row.length_of_study_time_step),
        Number(row.in_study_repair)
      ]);
    }
  }
  repairStmt.free();

  const labelStmt = db.prepare(`
    INSERT OR REPLACE INTO vehicle_label (vehicle_id, class_label)
    VALUES (?, ?)
  `);
  for (const splitName of ['validation', 'test']) {
    const rows = await readSmallCsv(`${splitName}_labels.csv`);
    const selected = selectedBySplit.get(splitName);
    for (const row of rows) {
      const vehicleId = Number(row.vehicle_id);
      if (selected.has(vehicleId)) {
        labelStmt.run([vehicleId, Number(row.class_label)]);
      }
    }
  }
  labelStmt.free();
}

async function insertOperationalReadouts(db, selectedBySplit, features) {
  const featureIds = features.map(feature => feature.featureId);
  let readoutId = 1;
  let insertedReadouts = 0;
  let insertedMeasurements = 0;

  const readoutStmt = db.prepare(`
    INSERT INTO readout (readout_id, vehicle_id, readout_seq, time_step, is_latest)
    VALUES (?, ?, ?, ?, 0)
  `);
  const measurementStmt = db.prepare(`
    INSERT INTO readout_measurement (readout_id, feature_id, measurement_value)
    VALUES (?, ?, ?)
  `);
  const latestStmt = db.prepare('UPDATE readout SET is_latest = 1 WHERE readout_id = ?');

  for (const split of SPLITS) {
    const selected = selectedBySplit.get(split.name);
    const readoutSeqByVehicle = new Map();
    const latestReadoutByVehicle = new Map();
    const rl = readline.createInterface({
      input: fs.createReadStream(csvPath(split.operational)),
      crlfDelay: Infinity
    });

    let header = null;
    for await (const line of rl) {
      if (!header) {
        header = parseCsvLine(line);
        continue;
      }
      if (!line) continue;

      const cells = parseCsvLine(line);
      const vehicleId = Number(cells[0]);
      if (!selected.has(vehicleId)) continue;

      const seq = (readoutSeqByVehicle.get(vehicleId) || 0) + 1;
      readoutSeqByVehicle.set(vehicleId, seq);

      const currentReadoutId = readoutId++;
      readoutStmt.run([currentReadoutId, vehicleId, seq, Number(cells[1])]);
      insertedReadouts++;

      for (let i = 0; i < featureIds.length; i++) {
        const value = cells[i + 2] === '' ? null : Number(cells[i + 2]);
        measurementStmt.run([currentReadoutId, featureIds[i], value]);
        insertedMeasurements++;
      }

      latestReadoutByVehicle.set(vehicleId, currentReadoutId);
    }

    for (const latestReadoutId of latestReadoutByVehicle.values()) {
      latestStmt.run([latestReadoutId]);
    }
  }

  readoutStmt.free();
  measurementStmt.free();
  latestStmt.free();

  return { insertedReadouts, insertedMeasurements };
}

function scalar(db, sql) {
  const result = db.exec(sql);
  return result[0]?.values?.[0]?.[0] ?? 0;
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const SQL = await initSqlJs({
    locateFile: file => require.resolve(`sql.js/dist/${file}`)
  });
  const db = new SQL.Database();

  const selectedBySplit = await chooseVehicles();
  const operational = await getOperationalFeatures();

  db.run('BEGIN TRANSACTION');
  createSchema(db);
  insertReferenceData(db, operational);
  await insertSpecifications(db, selectedBySplit);
  await insertTargets(db, selectedBySplit);
  const operationalCounts = await insertOperationalReadouts(db, selectedBySplit, operational.features);
  db.run('COMMIT');

  const data = db.export();
  fs.writeFileSync(OUT_FILE, Buffer.from(data));

  console.log(`Wrote ${OUT_FILE}`);
  console.log(`Vehicles: ${scalar(db, 'SELECT COUNT(*) FROM vehicle')}`);
  console.log(`Readouts: ${operationalCounts.insertedReadouts}`);
  console.log(`Measurements: ${operationalCounts.insertedMeasurements}`);
  console.log(`Operational features: ${operational.features.length}`);
  console.log('Selected vehicles by split:');
  for (const split of SPLITS) {
    console.log(`- ${split.name}: ${selectedBySplit.get(split.name).size}`);
  }

  db.close();
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
