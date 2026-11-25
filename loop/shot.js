// Screenshots for the human visual review (loop/shots/, not committed).
const { chromium } = require('playwright'); const path = require('path');
const T = path.join(__dirname, '../tests/web/node_modules/three');
const shots = JSON.parse(process.argv[2]);
(async () => { const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  for (const [name, url, w, h, js] of shots) {
    const p = await b.newPage({ viewport: { width: w, height: h } });
    await p.route(/three@0\.160\.0\/(.*)$/, r => r.fulfill({ path: path.join(T, r.request().url().match(/three@0\.160\.0\/(.*)$/)[1]), contentType: 'application/javascript' }));
    await p.route(/fonts\./, r => r.abort());
    await p.goto('http://localhost:8765/' + url);
    if (url.startsWith('index')) await p.waitForFunction(() => window.__neo && __neo.ready, null, { timeout: 60000 }); else await p.waitForTimeout(3000);
    if (js) { await p.evaluate(js); await p.waitForTimeout(2500); }
    await p.screenshot({ path: path.join(__dirname, 'shots', name + '.png') }); await p.close();
  }
  await b.close(); })();
