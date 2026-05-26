const fs = require('fs');
const path = require('path');
const initSqlJs = require('sql.js');

const ROOT = path.resolve(__dirname, '..');
const DB_FILES = {
  app: path.join(ROOT, 'database', 'eneb453_app.sqlite'),
  demo: path.join(ROOT, 'database', 'scania_pdm_demo.sqlite')
};
const PRESETS = {
  users: {
    db: 'app',
    table: 'users',
    columns: ['id', 'username', 'role', 'created_at'],
    orderBy: 'id ASC',
    limit: 20
  },
  'notes-active': {
    db: 'app',
    table: 'maintenance_notes',
    where: 'deleted_at IS NULL',
    orderBy: 'updated_at DESC, id DESC',
    limit: 20
  },
  'notes-deleted': {
    db: 'app',
    table: 'maintenance_notes',
    where: 'deleted_at IS NOT NULL',
    orderBy: 'updated_at DESC, id DESC',
    limit: 20
  },
  history: {
    db: 'app',
    table: 'maintenance_note_history',
    orderBy: 'created_at DESC, id DESC',
    limit: 30
  }
};

function argValue(name, fallback = '') {
  const prefix = `--${name}=`;
  const direct = process.argv.slice(2).find(value => value.startsWith(prefix));
  if (direct) return direct.slice(prefix.length);
  const envValue = process.env[`npm_config_${name.replaceAll('-', '_')}`];
  return envValue === undefined ? fallback : envValue;
}

function hasFlag(name) {
  return process.argv.slice(2).includes(`--${name}`) || process.env[`npm_config_${name.replaceAll('-', '_')}`] === 'true';
}

function toRows(result) {
  if (!result?.length) return [];
  const [{ columns, values }] = result;
  return values.map(row => Object.fromEntries(columns.map((column, index) => [column, row[index]])));
}

function safeIdentifier(value, label) {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) throw new Error(`Invalid ${label}: ${value}`);
  return value;
}

function safeColumnList(columns) {
  if (!columns?.length) return '*';
  return columns.map(column => safeIdentifier(column.trim(), 'column')).join(', ');
}

async function openDatabase(dbKey) {
  const file = DB_FILES[dbKey];
  if (!file) throw new Error(`Unknown database "${dbKey}". Use app or demo.`);
  if (!fs.existsSync(file)) throw new Error(`Database not found: ${file}`);
  const SQL = await initSqlJs();
  return { db: new SQL.Database(fs.readFileSync(file)), file };
}

function listTables(db) {
  return toRows(db.exec(`
    SELECT name
    FROM sqlite_master
    WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
    ORDER BY name
  `)).map(row => row.name);
}

function printTableCounts(db, dbKey, file) {
  const tables = listTables(db);
  const rows = tables.map(table => {
    const count = db.exec(`SELECT COUNT(*) AS count FROM ${safeIdentifier(table, 'table')}`)[0].values[0][0];
    return { database: dbKey, table, rows: count };
  });
  console.log(`Database: ${path.relative(ROOT, file)}`);
  console.table(rows);
}

function printSchema(db, table) {
  const columns = toRows(db.exec(`PRAGMA table_info(${safeIdentifier(table, 'table')})`));
  console.table(columns.map(column => ({
    column: column.name,
    type: column.type,
    required: column.notnull ? 'yes' : 'no',
    primaryKey: column.pk ? 'yes' : 'no'
  })));
}

function buildQuery({ table, columns, where, orderBy, limit }) {
  const safeTable = safeIdentifier(table, 'table');
  const selectedColumns = safeColumnList(columns);
  const whereSql = where ? ` WHERE ${where}` : '';
  const orderSql = orderBy ? ` ORDER BY ${orderBy}` : '';
  return `SELECT ${selectedColumns} FROM ${safeTable}${whereSql}${orderSql} LIMIT ${limit}`;
}

async function main() {
  const presetName = argValue('preset');
  const preset = presetName ? PRESETS[presetName] : null;
  if (presetName && !preset) throw new Error(`Unknown preset "${presetName}".`);

  const dbKey = preset?.db || argValue('db', 'app');
  const { db, file } = await openDatabase(dbKey);

  try {
    if (hasFlag('list')) {
      printTableCounts(db, dbKey, file);
      return;
    }

    const table = preset?.table || argValue('table');
    if (!table) {
      printTableCounts(db, dbKey, file);
      console.log('Examples:');
      console.log('  npm run db:users');
      console.log('  npm run db:notes:active');
      console.log('  npm run db:app --table=maintenance_notes --limit=20');
      return;
    }

    const availableTables = listTables(db);
    if (!availableTables.includes(table)) {
      throw new Error(`Unknown table "${table}". Available tables: ${availableTables.join(', ')}`);
    }

    console.log(`Database: ${path.relative(ROOT, file)}`);
    console.log(`Table: ${table}`);
    if (hasFlag('schema')) printSchema(db, table);

    const limit = Math.max(1, Math.min(200, Number(preset?.limit || argValue('limit', '20')) || 20));
    const columnsText = argValue('columns');
    const columns = preset?.columns || (columnsText ? columnsText.split(',') : null);
    const where = preset?.where || argValue('where');
    const orderBy = preset?.orderBy || argValue('order-by');
    const rows = toRows(db.exec(buildQuery({ table, columns, where, orderBy, limit })));
    console.table(rows);
  } finally {
    db.close();
  }
}

main().catch(error => {
  console.error(error.message);
  process.exit(1);
});
