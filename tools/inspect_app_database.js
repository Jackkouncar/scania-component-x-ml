const fs = require('fs');
const path = require('path');
const initSqlJs = require('sql.js');

const ROOT = path.resolve(__dirname, '..');
const DB_FILE = path.resolve(process.env.APP_DB_FILE || path.join(ROOT, 'database', 'eneb453_app.sqlite'));
const DEFAULT_TABLES = [
  'users',
  'app_vehicles',
  'telemetry_records',
  'maintenance_notes',
  'maintenance_note_history',
  'audit_log'
];

function argValue(name, fallback = '') {
  const prefix = `--${name}=`;
  const item = process.argv.slice(2).find(value => value.startsWith(prefix));
  return item ? item.slice(prefix.length) : fallback;
}

function toRows(result) {
  if (!result?.length) return [];
  const [{ columns, values }] = result;
  return values.map(row => Object.fromEntries(columns.map((column, index) => [column, row[index]])));
}

async function main() {
  if (!fs.existsSync(DB_FILE)) {
    throw new Error(`Database not found: ${DB_FILE}. Start the app once with npm run serve to create it.`);
  }

  const SQL = await initSqlJs();
  const db = new SQL.Database(fs.readFileSync(DB_FILE));
  const tableRows = toRows(db.exec(`
    SELECT name
    FROM sqlite_master
    WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
    ORDER BY name
  `));
  const availableTables = tableRows.map(row => row.name);
  const requestedTable = argValue('table');
  const limit = Math.max(1, Math.min(100, Number(argValue('limit', '10')) || 10));

  console.log(`Database: ${path.relative(ROOT, DB_FILE)}`);

  if (!requestedTable) {
    const counts = DEFAULT_TABLES
      .filter(table => availableTables.includes(table))
      .map(table => {
        const result = db.exec(`SELECT COUNT(*) AS count FROM ${table}`);
        return { table, rows: result[0].values[0][0] };
      });
    console.table(counts);
    console.log(`Use: npm run db:app -- --table=app_vehicles --limit=10`);
    db.close();
    return;
  }

  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(requestedTable) || !availableTables.includes(requestedTable)) {
    throw new Error(`Unknown table "${requestedTable}". Available tables: ${availableTables.join(', ')}`);
  }

  const columns = toRows(db.exec(`PRAGMA table_info(${requestedTable})`));
  console.log(`Columns: ${columns.map(column => column.name).join(', ')}`);

  const rows = toRows(db.exec(`SELECT * FROM ${requestedTable} LIMIT ${limit}`));
  console.table(rows);
  db.close();
}

main().catch(error => {
  console.error(error.message);
  process.exit(1);
});
