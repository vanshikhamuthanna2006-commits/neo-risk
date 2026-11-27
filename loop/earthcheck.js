const { chromium } = require('playwright'); const path = require('path');
(async () => { const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  for (const [w, h, dpr] of [[1366, 657, 1], [1536, 730, 1.25], [1280, 609, 1.5], [1920, 937, 1], [1093, 520, 1.25], [1100, 600, 1]]) {
    const ctx = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: dpr }); const p = await ctx.newPage(); const errs = [];
    p.on('pageerror', e => errs.push(e.message));
    await p.route(/three@0\.160\.0\/(.*)$/, r => r.fulfill({ path: path.join(__dirname, '../tests/web/node_modules/three', r.request().url().match(/three@0\.160\.0\/(.*)$/)[1]), contentType: 'application/javascript' }));
    await p.route(/fonts\./, r => r.abort());
    await p.goto('http://localhost:8765/index.html'); await p.waitForFunction(() => window.__neo && __neo.ready, null, { timeout: 90000 });
    const btn = await p.$('#frames [data-fr="earth"]');
    const info = await p.evaluate(() => { const e = document.querySelector('#frames [data-fr="earth"]'); const r = e.getBoundingClientRect();
      const top = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
      const side = document.querySelector('#side'), sr = side.getBoundingClientRect();
      return { btn: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)], visible: r.width > 0 && r.y < innerHeight && r.bottom > 0,
        covered: top !== e && !e.contains(top) ? (top.id || top.className || top.tagName) : null, side: [Math.round(sr.top), Math.round(sr.bottom)], sideScroll: side.scrollHeight > side.clientHeight }; });
    let clicked = 'n/a';
    try { await btn.click({ timeout: 5000 }); await p.waitForTimeout(400); clicked = await p.evaluate(() => __neo.frame); } catch (e) { clicked = 'CLICK FAILED: ' + e.message.split('\n')[0].slice(0, 80); }
    console.log(`${w}x${h}@${dpr}`, JSON.stringify(info), '->', clicked, errs.length ? errs : '');
    await ctx.close(); }
  await b.close(); })();
