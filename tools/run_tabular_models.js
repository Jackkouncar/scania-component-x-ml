const { spawnSync } = require('child_process');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const script = path.join(ROOT, 'tools', 'train_tabular_models.py');
const candidates = [
  { command: 'py', args: [script] },
  { command: 'python', args: [script] },
  { command: 'python3', args: [script] }
];

let lastError = null;
for (const candidate of candidates) {
  const result = spawnSync(candidate.command, candidate.args, {
    cwd: ROOT,
    stdio: 'inherit',
    env: process.env
  });

  if (!result.error && result.status === 0) process.exit(0);
  if (result.error && result.error.code === 'ENOENT') {
    lastError = result.error;
    continue;
  }
  process.exit(result.status || 1);
}

console.error('Python was not found. Install Python and then run: pip install -r requirements-ml.txt');
if (lastError) console.error(lastError.message);
process.exit(1);
