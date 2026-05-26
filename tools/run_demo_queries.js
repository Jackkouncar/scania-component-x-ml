const fs = require('fs');
const path = require('path');
const initSqlJs = require('sql.js');

const ROOT = path.resolve(__dirname, '..');
const DB_FILE = path.join(ROOT, 'database', 'scania_pdm_demo.sqlite');

const QUERIES = [
  {
    title: 'Vehicles loaded by split',
    sql: `
      SELECT ds.split_name, COUNT(*) AS vehicles
      FROM vehicle v
      JOIN dataset_split ds ON ds.split_id = v.split_id
      GROUP BY ds.split_name
      ORDER BY ds.split_id;
    `
  },
  {
    title: 'Validation and test class balance',
    sql: `
      SELECT ds.split_name, vl.class_label, COUNT(*) AS vehicles
      FROM vehicle_label vl
      JOIN vehicle v ON v.vehicle_id = vl.vehicle_id
      JOIN dataset_split ds ON ds.split_id = v.split_id
      GROUP BY ds.split_name, vl.class_label
      ORDER BY ds.split_id, vl.class_label;
    `
  },
  {
    title: 'Training repair outcome balance',
    sql: `
      SELECT in_study_repair, COUNT(*) AS vehicles
      FROM repair_outcome
      GROUP BY in_study_repair
      ORDER BY in_study_repair;
    `
  },
  {
    title: 'Highest-risk validation vehicles',
    sql: `
      SELECT v.vehicle_id, vl.class_label, r.time_step AS latest_time_step
      FROM vehicle v
      JOIN dataset_split ds ON ds.split_id = v.split_id
      JOIN vehicle_label vl ON vl.vehicle_id = v.vehicle_id
      JOIN readout r ON r.vehicle_id = v.vehicle_id AND r.is_latest = 1
      WHERE ds.split_name = 'validation'
      ORDER BY vl.class_label DESC, r.time_step DESC
      LIMIT 10;
    `
  },
  {
    title: 'Average latest 427_0 counter by validation class',
    sql: `
      SELECT vl.class_label, ROUND(AVG(rm.measurement_value), 2) AS avg_latest_427_0
      FROM vehicle_label vl
      JOIN vehicle v ON v.vehicle_id = vl.vehicle_id
      JOIN dataset_split ds ON ds.split_id = v.split_id
      JOIN readout r ON r.vehicle_id = v.vehicle_id AND r.is_latest = 1
      JOIN readout_measurement rm ON rm.readout_id = r.readout_id
      JOIN operational_feature f ON f.feature_id = rm.feature_id
      WHERE ds.split_name = 'validation'
        AND f.feature_name = '427_0'
      GROUP BY vl.class_label
      ORDER BY vl.class_label;
    `
  },
  {
    title: 'Spec_1 distribution in the demo database',
    sql: `
      SELECT ds.split_name, vs.category_label, COUNT(*) AS vehicles
      FROM vehicle_specification vs
      JOIN vehicle v ON v.vehicle_id = vs.vehicle_id
      JOIN dataset_split ds ON ds.split_id = v.split_id
      JOIN spec_attribute sa ON sa.spec_attribute_id = vs.spec_attribute_id
      WHERE sa.spec_name = 'Spec_1'
      GROUP BY ds.split_name, vs.category_label
      ORDER BY ds.split_id, vehicles DESC, vs.category_label
      LIMIT 20;
    `
  }
];

function formatResult(result) {
  if (!result) return '(no rows)';

  const widths = result.columns.map((column, index) => {
    const valueWidths = result.values.map(row => String(row[index] ?? '').length);
    return Math.max(column.length, ...valueWidths);
  });

  const formatRow = row => row
    .map((value, index) => String(value ?? '').padEnd(widths[index], ' '))
    .join(' | ');

  const header = formatRow(result.columns);
  const divider = widths.map(width => '-'.repeat(width)).join('-+-');
  const rows = result.values.map(formatRow);
  return [header, divider, ...rows].join('\n');
}

async function main() {
  if (!fs.existsSync(DB_FILE)) {
    throw new Error(`Database not found at ${DB_FILE}. Run npm run build:db first.`);
  }

  const SQL = await initSqlJs({
    locateFile: file => require.resolve(`sql.js/dist/${file}`)
  });
  const db = new SQL.Database(fs.readFileSync(DB_FILE));

  for (const [index, query] of QUERIES.entries()) {
    console.log(`\n${index + 1}. ${query.title}`);
    console.log(formatResult(db.exec(query.sql)[0]));
  }

  db.close();
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
