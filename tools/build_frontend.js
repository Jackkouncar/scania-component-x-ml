const path = require('path');
const esbuild = require('esbuild');

const ROOT = path.resolve(__dirname, '..');

esbuild.build({
  entryPoints: [path.join(ROOT, 'site', 'react_admin.jsx')],
  bundle: true,
  minify: true,
  sourcemap: false,
  format: 'iife',
  target: ['es2020'],
  outfile: path.join(ROOT, 'site', 'react_admin.bundle.js')
}).then(result => {
  if (result.errors?.length) {
    process.exitCode = 1;
    return;
  }
  console.log('Wrote site/react_admin.bundle.js');
}).catch(error => {
  console.error(error);
  process.exit(1);
});
