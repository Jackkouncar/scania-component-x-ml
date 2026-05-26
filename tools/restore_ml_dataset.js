const fs = require('fs');
const path = require('path');
const { Readable } = require('stream');
const { pipeline } = require('stream/promises');
const zlib = require('zlib');

const ROOT = path.resolve(__dirname, '..');
const ML_DIR = path.join(ROOT, 'ml');
const OUTPUT = path.join(ML_DIR, 'ml_dataset.json');
const PART_PREFIX = 'ml_dataset.json.gz.part';

async function* readParts(parts) {
  for (const fileName of parts) {
    for await (const chunk of fs.createReadStream(path.join(ML_DIR, fileName))) {
      yield chunk;
    }
  }
}

async function main() {
  const parts = fs.readdirSync(ML_DIR)
    .filter(fileName => fileName.startsWith(PART_PREFIX))
    .sort();

  if (!parts.length) {
    if (fs.existsSync(OUTPUT)) {
      console.log('ml/ml_dataset.json already exists; no compressed archive parts found.');
      return;
    }
    throw new Error('No compressed ML dataset archive parts were found in ml/.');
  }

  await pipeline(
    Readable.from(readParts(parts)),
    zlib.createGunzip(),
    fs.createWriteStream(OUTPUT)
  );
  console.log(`Restored ${path.relative(ROOT, OUTPUT)} from ${parts.length} compressed parts.`);
}

main().catch(error => {
  console.error(error.message);
  process.exit(1);
});
