const { chromium } = require('playwright'); const path = require('path');
(async () => { const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const p = await b.newPage(); p.on('pageerror', e => console.log('PAGEERROR', e.message, (e.stack || '').split('\n')[1]));
  p.on('console', m => m.type() === 'error' && console.log('CONSOLE', m.text().slice(0, 200)));
  await p.route(/three@0\.160\.0\/(.*)$/, r => r.fulfill({ path: path.join(__dirname, '../tests/web/node_modules/three', r.request().url().match(/three@0\.160\.0\/(.*)$/)[1]), contentType: 'application/javascript' }));
  await p.route(/fonts\./, r => r.abort());
  await p.goto('http://localhost:8765/index.html'); await p.waitForTimeout(8000);
  console.log('ready?', await p.evaluate(() => window.__neo ? __neo.ready : 'no __neo')); await b.close(); })();
