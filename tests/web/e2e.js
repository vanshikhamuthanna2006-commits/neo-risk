// End-to-end check of docs/index.html (3-D atlas) and docs/explorer.html.
const { chromium } = require('playwright');
const path = require('path'), fs = require('fs');
const [THREE_DIR, TRUTH, OUT] = process.argv.slice(2);
const truth = JSON.parse(fs.readFileSync(TRUTH));
const results = []; let fails = 0;
const ok = (name, cond, info = '') => { results.push(`${cond ? 'PASS' : 'FAIL'}  ${name}${info ? '  — ' + info : ''}`); if (!cond) fails++; };

async function page(b, vp) {
  const p = await b.newPage({ viewport: vp });
  p.errs = [];
  p.on('pageerror', e => p.errs.push(e.message));
  p.on('console', m => { if (m.type() === 'error' && !/ERR_FAILED/.test(m.text())) p.errs.push(m.text()); });
  await p.route(/three@0\.160\.0\/(.*)$/, r => r.fulfill({ path: path.join(THREE_DIR, r.request().url().match(/three@0\.160\.0\/(.*)$/)[1]), contentType: 'application/javascript' }));
  await p.route(/fonts\.(googleapis|gstatic)\.com/, r => r.abort());
  return p;
}
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

(async () => {
  const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });

  // ------------------------------------------------------------------ atlas
  const p = await page(b, { width: 1440, height: 900 });
  await p.goto('http://localhost:8765/index.html');
  await p.waitForFunction(() => window.__neo && window.__neo.rowOf('99942') >= 0, null, { timeout: 30000 });
  await p.waitForTimeout(6000);                                     // intro finished

  // positional accuracy vs JPL Horizons (heliocentric, au)
  for (const [name, byJd] of Object.entries(truth)) {
    for (const [jd, ref] of Object.entries(byJd)) {
      const got = await p.evaluate(([n, j]) => ['Earth', 'Mars', 'Jupiter'].includes(n) ? __neo.helioPlanet(n, +j) : __neo.helioAst(__neo.rowOf(n), +j), [name, jd]);
      const err = dist(got, ref), rel = err / Math.hypot(...ref);
      const yr = (2000 + (jd - 2451545) / 365.25).toFixed(1);
      // planets: Standish ~1e-3 au class; asteroids: two-body drift grows with time from the epoch
      // featured flyby objects use 30-day JPL element sets -> tight everywhere; others: aligned to
      // 2026 by N-body, then two-body (drift grows with time but must stay well under 0.2 au)
      const featured = ['99942', '2024 YR4'].includes(name);
      const tol = ['Earth', 'Mars'].includes(name) ? 0.002 : name === 'Jupiter' ? 0.02 : featured ? 0.002 : (jd < 2461400 ? 0.01 : 0.2);
      ok(`position ${name.padEnd(8)} ${yr}`, err < tol, `error ${err.toFixed(4)} au (${(rel * 100).toFixed(2)}% of r), tol ${tol}`);
    }
  }

  // search + select
  await p.fill('#q', 'apophis'); await p.waitForTimeout(300);
  ok('search shows results', await p.isVisible('#results button'));
  await p.keyboard.press('Enter'); await p.waitForTimeout(2500);
  const apo = await p.evaluate(() => __neo.rowOf('99942'));
  ok('Enter selects Apophis', await p.evaluate(() => __neo.selected) === apo);
  ok('card shows Apophis', /Apophis/.test(await p.textContent('#card')));
  ok('URL updated', /obj=99942/.test(p.url()));
  // click at screen centre (camera has flown to Apophis) picks something nearby
  await p.mouse.click(720, 450); await p.waitForTimeout(300);
  ok('click-to-pick selects an object', await p.evaluate(() => __neo.selected) >= 0);
  await p.keyboard.press('Escape'); await p.waitForTimeout(200);
  ok('Escape clears selection', await p.evaluate(() => __neo.selected) === -1);

  // filters
  const counts = {};
  for (const f of ['0', '1', '2', '3']) { await p.click(`#filters [data-f="${f}"]`); await p.waitForTimeout(150); counts[f] = await p.textContent('#count'); }
  const n = s => +s.match(/Showing ([\d,]+)/)[1].replace(/,/g, '');
  ok('filter All = 43,160', n(counts[0]) === 43160, counts[0]);
  ok('filter PHAs < All', n(counts[1]) > 1000 && n(counts[1]) < 43160, counts[1]);
  ok('filter Sentry', n(counts[2]) > 1000 && n(counts[2]) < n(counts[0]), counts[2]);
  ok('filter Top 1% ≈ 432', Math.abs(n(counts[3]) - 432) < 60, counts[3]);
  await p.click('#modes [data-m="1"]'); await p.waitForTimeout(150);
  ok('colour by PHA keeps top-1% count', Math.abs(n(await p.textContent('#count')) - 432) < 60);
  await p.click('#filters [data-f="0"]'); await p.click('#modes [data-m="0"]');

  // time controls
  await p.click('#play'); const t1 = await p.evaluate(() => __neo.simJD); await p.waitForTimeout(800);
  ok('pause stops time', await p.evaluate(() => __neo.simJD) === t1);
  await p.click('#play'); await p.click('#speeds [data-s="2629800"]'); const t2 = await p.evaluate(() => __neo.simJD); await p.waitForTimeout(1000);
  const adv = await p.evaluate(() => __neo.simJD) - t2;
  ok('1 mo/s advances ~30 days/s', adv > 15 && adv < 45, `${adv.toFixed(1)} d in 1 s (software renderer)`);
  await p.click('#rev'); const t3 = await p.evaluate(() => __neo.simJD); await p.waitForTimeout(600);
  ok('reverse runs time backwards', await p.evaluate(() => __neo.simJD) < t3);
  await p.click('#rev'); await p.click('#now'); await p.waitForTimeout(300);
  const nowErr = Math.abs(await p.evaluate(() => __neo.simJD - (Date.now() / 86400000 + 2440587.5))) * 86400;
  ok('Now = real time', nowErr < 5, `${nowErr.toFixed(1)} s off`);
  ok('LIVE badge shown', await p.isVisible('#live'));
  await p.screenshot({ path: `${OUT}/e2e_live.png` });

  // every flyby
  const fly = JSON.parse(fs.readFileSync('/home/user/neo-risk/docs/data/flybys.json', 'utf8')).flybys;
  for (let j = 0; j < fly.length; j++) {
    await p.evaluate(j => __neo.enterFlyby(j), j); await p.waitForTimeout(500);
    ok(`flyby ${fly[j].name}: mode`, await p.evaluate(() => __neo.mode) === 'flyby');
    const at = await p.evaluate(jd => __neo.flybyAst(jd), fly[j].jd_ca);
    const d = Math.hypot(...at);
    ok(`flyby ${fly[j].name}: distance at CA`, Math.abs(d - fly[j].dist_km) / fly[j].dist_km < 0.001, `${Math.round(d)} km vs JPL ${Math.round(fly[j].dist_km)} km`);
    await p.click('#f-ca'); await p.waitForTimeout(1500);
    const rd = await p.textContent('#r-d'), rt = await p.textContent('#r-t');
    ok(`flyby ${fly[j].name}: live readouts`, /km/.test(rd) && /^T[−+]/.test(rt), `${rd} · ${rt}`);
    await p.click('#f-exit'); await p.waitForTimeout(300);
    ok(`flyby ${fly[j].name}: exit restores solar view`, await p.evaluate(() => __neo.mode) === 'solar');
  }
  // encounter list button path
  await p.keyboard.press('Escape'); await p.waitForTimeout(200);
  await p.click('#enc-list .enc'); await p.waitForTimeout(400);
  ok('encounter list opens a flyby', await p.evaluate(() => __neo.mode) === 'flyby');
  await p.keyboard.press('Escape'); await p.waitForTimeout(200);
  ok('Escape leaves flyby', await p.evaluate(() => __neo.mode) === 'solar');
  ok('atlas: no page errors', p.errs.length === 0, p.errs.slice(0, 3).join(' | '));
  await p.close();

  // deep links + phone
  const m = await page(b, { width: 390, height: 844 });
  await m.goto('http://localhost:8765/index.html?flyby=2024%20YR4');
  await m.waitForFunction(() => window.__neo && __neo.mode === 'flyby', null, { timeout: 30000 });
  ok('?flyby= deep link opens flyby (phone)', true);
  await m.click('#f-exit'); await m.waitForTimeout(300);
  await m.click('#open-side'); ok('phone: search panel opens', await m.isVisible('#side'));
  await m.click('#open-enc'); ok('phone: encounters panel opens', await m.isVisible('#enc'));
  const sw = await m.evaluate(() => document.documentElement.scrollWidth);
  ok('phone: no horizontal overflow', sw <= 390, `scrollWidth ${sw}`);
  ok('phone: no page errors', m.errs.length === 0, m.errs.slice(0, 3).join(' | '));
  await m.close();

  // ------------------------------------------------------------------ explorer
  const e = await page(b, { width: 1280, height: 900 });
  await e.goto('http://localhost:8765/explorer.html?obj=101955#explore');
  await e.waitForSelector('#loading', { state: 'detached', timeout: 30000 });
  ok('explorer ?obj= selects Bennu', /Bennu/.test(await e.textContent('#detail')));
  for (const t of ['decide', 'results', 'leak', 'about', 'explore']) {
    await e.click(`nav.tabs [data-tab="${t}"]`); await e.waitForTimeout(200);
    ok(`explorer tab ${t} renders`, await e.isVisible(`#tab-${t}`));
  }
  await e.click('#m-pha'); await e.waitForTimeout(200);
  await e.fill('#q', ''); await e.dispatchEvent('#q', 'input'); await e.waitForTimeout(300);
  ok('explorer Mode A list', /43,160 objects/.test(await e.textContent('#count')));
  await e.click('#tbl th[data-k="arc"]'); await e.waitForTimeout(200);
  ok('explorer sort by arc', /arc/.test(await e.textContent('#count')));
  await e.click('nav.tabs [data-tab="decide"]');
  await e.$eval('#ratio', el => { el.value = '1'; el.dispatchEvent(new Event('input')); });
  ok('cost slider updates', /^10 : 1$/.test((await e.textContent('#ratio-v')).trim()), await e.textContent('#ratio-v'));
  await e.$eval('#ah', el => { el.value = '0.10'; el.dispatchEvent(new Event('input')); });
  ok('conformal slider updates', /0\.10/.test(await e.textContent('#ah-v')));
  await e.click('#theme'); await e.waitForTimeout(200);
  ok('theme toggles', await e.evaluate(() => document.documentElement.getAttribute('data-theme')) !== null);
  await e.click('a[href="index.html"]'); await e.waitForURL(/index\.html/);
  ok('explorer → atlas link', true);
  ok('explorer: no page errors', e.errs.length === 0, e.errs.slice(0, 3).join(' | '));
  await e.close();

  await b.close();
  console.log(results.join('\n'));
  console.log(`\n${results.length - fails}/${results.length} checks passed`);
  process.exit(fails ? 1 : 0);
})().catch(err => { console.log(results.join('\n')); console.error('CRASH', err.message); process.exit(2); });
