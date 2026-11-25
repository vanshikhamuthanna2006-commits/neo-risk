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
  const W = 960, H = 600;                                            // fewer pixels: software WebGL is the bottleneck
  const p = await page(b, { width: W, height: H });
  await p.goto('http://localhost:8765/index.html');
  await p.waitForFunction(() => window.__neo && __neo.ready, null, { timeout: 60000 });   // data loaded + intro done

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
  const ov = await p.evaluate(() => { const a = document.querySelector('#brand').getBoundingClientRect(), c = document.querySelector('#card').getBoundingClientRect();
    return c.top >= a.bottom - 0.5 || c.left >= a.right || c.right <= a.left; });
  ok('card does not overlap the brand panel', ov);
  ok('URL updated', /obj=99942/.test(p.url()));
  // W1: "why this score" drivers in the atlas card must equal explain.json for this object
  const X = JSON.parse(fs.readFileSync('/home/user/neo-risk/docs/data/explain.json', 'utf8'));
  await p.waitForSelector('#card .drv', { timeout: 30000 });
  const drv = await p.$$eval('#card .drv', els => els.map(x => +x.dataset.val));
  const want = X.sentry.val.slice(apo * 5, apo * 5 + 5);
  ok('atlas: 5 score drivers shown', drv.length === 5, `${drv.length}`);
  ok('atlas: drivers equal explain.json', drv.every((v, j) => Math.abs(v - want[j]) < 1e-9), `${drv} vs ${want}`);
  await p.click('#modes [data-m="1"]'); await p.waitForTimeout(400);
  const drvP = await p.$$eval('#card .drv', els => els.map(x => +x.dataset.val));
  ok('atlas: drivers follow the colour mode (PHA)', drvP.every((v, j) => Math.abs(v - X.pha.val[apo * 5 + j]) < 1e-9));
  await p.click('#modes [data-m="0"]'); await p.waitForTimeout(200);
  // click at screen centre (camera has flown to Apophis) picks something nearby
  await p.mouse.click(W / 2, H / 2); await p.waitForTimeout(300);
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

  // W2: scale grid + top-risk orbit overlay
  ok('grid toggle present and on by default', await p.isChecked('#t-grid') && await p.evaluate(() => __neo.gridVisible));
  await p.click('#t-grid'); ok('grid toggles off', !(await p.evaluate(() => __neo.gridVisible))); await p.click('#t-grid');
  await p.click('#t-top'); await p.waitForTimeout(300);
  ok('top-risk overlay draws 50 orbits', await p.evaluate(() => __neo.topOrbits.count) === 50);
  const topS = await p.evaluate(() => __neo.topOrbits.rows.slice());
  const wantS = await p.evaluate(() => __neo.topRows(0, 50));
  ok('overlay = 50 highest P(Sentry) objects', JSON.stringify(topS) === JSON.stringify(wantS));
  await p.click('#modes [data-m="1"]'); await p.waitForTimeout(300);
  ok('overlay follows colour mode (PHA)', JSON.stringify(await p.evaluate(() => __neo.topOrbits.rows.slice())) === JSON.stringify(await p.evaluate(() => __neo.topRows(1, 50))));
  await p.click('#modes [data-m="0"]'); await p.click('#t-top');
  ok('overlay toggles off', await p.evaluate(() => __neo.topOrbits.count) === 0);

  // layout guard: no two visible HUD panels may overlap
  const clash = await p.evaluate(() => {
    const ids = ['#brand', '#clock', '#side', '#card', '#timebar', '#timeline', '#enc'];
    const R = ids.map(s => [s, document.querySelector(s)]).filter(([, e]) => e && getComputedStyle(e).display !== 'none' && e.offsetWidth)
      .map(([s, e]) => [s, e.getBoundingClientRect()]);
    const out = [];
    for (let i = 0; i < R.length; i++) for (let j = i + 1; j < R.length; j++) { const a = R[i][1], b = R[j][1];
      if (a.left < b.right - 1 && b.left < a.right - 1 && a.top < b.bottom - 1 && b.top < a.bottom - 1) out.push(R[i][0] + '/' + R[j][0]); }
    return out; });
  ok('HUD panels do not overlap', clash.length === 0, clash.join(', '));

  // W3: timeline — date input, flyby markers, scrubber
  await p.fill('#jump', '2029-04-13'); await p.dispatchEvent('#jump', 'change'); await p.waitForTimeout(200);
  const jdJump = await p.evaluate(() => __neo.simJD);
  ok('date input jumps to 2029-04-13', Math.abs(jdJump - 2462239.5) < 1.5, `JD ${jdJump.toFixed(2)}`);
  const flyJson = JSON.parse(fs.readFileSync('/home/user/neo-risk/docs/data/flybys.json', 'utf8')).flybys;
  const nMarks = await p.$$eval('#timeline .mark[data-j]', m => m.length);   // flyby markers (the "today" tick is separate)
  ok('timeline has one marker per flyby', nMarks === flyJson.length, `${nMarks}`);
  // every flyby marker must be clickable (not covered by a neighbour)
  const covered = await p.$$eval('#timeline .mark[data-j]', ms => ms.filter(m => { const r = m.getBoundingClientRect();
    const top = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2); return top !== m; }).map(m => m.dataset.j));
  ok('no flyby marker is covered', covered.length === 0, `covered: ${covered}`);
  await p.click('#timeline .mark[data-j="0"]'); await p.waitForTimeout(400);
  const after = await p.evaluate(() => ({ jd: __neo.simJD, sel: __neo.selected }));
  ok('marker jumps to its closest approach', Math.abs(after.jd - flyJson[0].jd_ca) < 1.5, `${after.jd.toFixed(2)} vs ${flyJson[0].jd_ca.toFixed(2)}`);
  ok('marker selects its asteroid', after.sel === flyJson[0].row);
  await p.$eval('#tl', el => { el.value = el.min; el.dispatchEvent(new Event('input')); }); await p.waitForTimeout(150);
  const jdMin = await p.evaluate(() => __neo.simJD);
  ok('scrubber start = 1950', Math.abs(jdMin - 2433282.5) < 2, `${jdMin.toFixed(1)}`);
  await p.keyboard.press('Escape');

  // time controls
  if (!(await p.evaluate(() => __neo.playing))) await p.click('#play');      // known state: playing
  await p.click('#play'); ok('play button pauses', !(await p.evaluate(() => __neo.playing)));
  const t1 = await p.evaluate(() => __neo.simJD); await p.waitForTimeout(800);
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
  // below 1180 px the list sits behind the "Close encounters" button, as a user would see it
  if (await p.isVisible('#open-enc')) await p.click('#open-enc');
  await p.click('#enc-list .enc'); await p.waitForTimeout(400);
  ok('encounter list opens a flyby', await p.evaluate(() => __neo.mode) === 'flyby');
  await p.keyboard.press('Escape'); await p.waitForTimeout(200);
  ok('Escape leaves flyby', await p.evaluate(() => __neo.mode) === 'solar');
  ok('atlas: no page errors', p.errs.length === 0, p.errs.slice(0, 3).join(' | '));
  await p.close();

  // W4: shareable views — a crafted link restores time, pause, selection, filter, colour mode, camera
  const s = await page(b, { width: W, height: H });
  await s.goto('http://localhost:8765/index.html?t=2462240.41&p=0&obj=99942&f=1&m=1&cam=12.5,6,18,1,0,-2');
  await s.waitForFunction(() => window.__neo && __neo.ready && __neo.selected >= 0, null, { timeout: 60000 });
  await s.waitForTimeout(500);
  const st = await s.evaluate(() => ({ jd: __neo.simJD, playing: __neo.playing, sel: __neo.selected, apo: __neo.rowOf('99942'), view: __neo.view }));
  ok('link restores time', Math.abs(st.jd - 2462240.41) < 0.01, st.jd.toFixed(3));
  ok('link restores pause', st.playing === false);
  ok('link restores selection', st.sel === st.apo);
  ok('link restores filter and colour mode', st.view.f === 1 && st.view.m === 1 && /PHA/.test(await s.textContent('#filters [aria-pressed="true"]')));
  ok('link restores camera', Math.hypot(st.view.cam[0] - 12.5, st.view.cam[1] - 6, st.view.cam[2] - 18) < 0.05 && Math.hypot(st.view.cam[3] - 1, st.view.cam[4], st.view.cam[5] + 2) < 0.05, JSON.stringify(st.view.cam));
  const link = await s.evaluate(() => __neo.shareURL());
  const u = new URL(link);
  ok('share link round-trips', Math.abs(+u.searchParams.get('t') - st.jd) < 0.01 && u.searchParams.get('obj') === '99942' && u.searchParams.get('f') === '1' && u.searchParams.get('m') === '1' && u.searchParams.get('p') === '0');
  await s.click('#share'); await s.waitForTimeout(300);
  ok('Copy link gives feedback', /copied|Copied|select/i.test(await s.textContent('#share')));
  ok('share: no page errors', s.errs.length === 0, s.errs.slice(0, 3).join(' | '));
  await s.close();

  // W6: quality + accessibility (atlas)
  const q6 = await page(b, { width: W, height: H });
  await q6.goto('http://localhost:8765/index.html'); await q6.waitForFunction(() => window.__neo && __neo.ready, null, { timeout: 60000 });
  const unnamed = await q6.evaluate(() => [...document.querySelectorAll('button, input, select, a[href]')].filter(el => {
    if (!el.offsetWidth && !el.offsetHeight) return false;                       // hidden controls are exempt
    const lab = el.getAttribute('aria-label') || el.getAttribute('title') || (el.labels && el.labels[0] && el.labels[0].textContent) || el.textContent || el.placeholder;
    return !lab || !lab.trim(); }).map(el => el.outerHTML.slice(0, 80)));
  ok('atlas: every visible control has an accessible name', unnamed.length === 0, unnamed.slice(0, 2).join(' | '));
  await q6.keyboard.press('?'); await q6.waitForTimeout(200);
  ok('"?" opens the keyboard help', await q6.isVisible('#help'));
  await q6.keyboard.press('Escape'); await q6.waitForTimeout(150);
  ok('Escape closes the help', !(await q6.isVisible('#help')));
  const t0k = await q6.evaluate(() => __neo.simJD); await q6.keyboard.press('Space'); await q6.keyboard.press('ArrowRight'); await q6.waitForTimeout(100);
  ok('→ steps one day forward when paused', Math.abs(await q6.evaluate(() => __neo.simJD) - t0k - 1) < 0.2);
  await q6.keyboard.press('g'); ok('G toggles the grid', !(await q6.evaluate(() => __neo.gridVisible))); await q6.keyboard.press('g');
  await q6.keyboard.press('Tab'); await q6.keyboard.press('Tab');
  ok('keyboard focus is visible', await q6.evaluate(() => { const a = document.activeElement; return a && a !== document.body && getComputedStyle(a).outlineStyle !== 'none'; }));
  const q0 = await q6.evaluate(() => __neo.quality);
  await q6.evaluate(() => { __neo.adapt(12); __neo.adapt(12); __neo.adapt(12); });
  const q1 = await q6.evaluate(() => __neo.quality);
  ok('low frame rate steps quality down', q1.level > q0.level && (q1.pixelRatio < q0.pixelRatio || q1.bloom === false), JSON.stringify([q0, q1]));
  ok('a11y page: no errors', q6.errs.length === 0, q6.errs.slice(0, 2).join(' | '));
  await q6.close();

  // W7: guided tour — every step sets up what it describes
  const tr = await page(b, { width: W, height: H });
  await tr.goto('http://localhost:8765/index.html'); await tr.waitForFunction(() => window.__neo && __neo.ready, null, { timeout: 60000 });
  await tr.click('#tour-start'); await tr.waitForTimeout(300);
  const nSteps = await tr.evaluate(() => __neo.tour.steps);
  ok('tour has 5 steps', nSteps === 5, `${nSteps}`);
  const seen = [];
  for (let st = 0; st < nSteps; st++) {
    await tr.waitForTimeout(st === 3 ? 1500 : 900);
    const s = await tr.evaluate(() => ({ i: __neo.tour.i, title: document.querySelector('#tour h3').textContent, mode: __neo.mode,
      sel: __neo.selected, apo: __neo.rowOf('99942'), f: __neo.view.f, top: __neo.topOrbits.count }));
    seen.push(s);
    if (st < nSteps - 1) await tr.click('#tour-next'); else await tr.click('#tour-next');   // last click = Finish
  }
  ok('tour step 1: overview', seen[0].i === 0 && seen[0].mode === 'solar');
  ok('tour step 2: risk colour shows the top-50 orbits', seen[1].top === 50);
  ok('tour step 3: Apophis selected', seen[2].sel === seen[2].apo);
  ok('tour step 4: Apophis flyby open', seen[3].mode === 'flyby');
  ok('tour step 5: back in the solar system', seen[4].mode === 'solar');
  await tr.waitForTimeout(300);
  ok('tour closes on Finish', !(await tr.isVisible('#tour')));
  ok('tour: no page errors', tr.errs.length === 0, tr.errs.slice(0, 2).join(' | '));
  await tr.close();

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
  {
    const X = JSON.parse(fs.readFileSync('/home/user/neo-risk/docs/data/explain.json', 'utf8'));
    const O = JSON.parse(fs.readFileSync('/home/user/neo-risk/docs/data/objects.json', 'utf8'));
    const k = O.pdes.indexOf('101955');
    await e.waitForSelector('#detail .drv', { timeout: 30000 });
    const d = await e.$$eval('#detail .drv', els => els.map(x => +x.dataset.val));
    ok('explorer: drivers equal explain.json (Bennu)', d.length === 5 && d.every((v, j) => Math.abs(v - X.sentry.val[k * 5 + j]) < 1e-9));
  }
  for (const t of ['decide', 'results', 'leak', 'about', 'explore']) {
    await e.click(`nav.tabs [data-tab="${t}"]`); await e.waitForTimeout(200);
    ok(`explorer tab ${t} renders`, await e.isVisible(`#tab-${t}`));
  }
  // R2: research-gap audit tab
  await e.click('nav.tabs [data-tab="gaps"]'); await e.waitForTimeout(300);
  ok('gaps tab renders', await e.isVisible('#tab-gaps'));
  const gaps = await e.$$eval('#gap-rows tr', rows => rows.map(r => ({
    status: (r.querySelector('[data-status]') || {}).dataset?.status, evidence: (r.querySelector('.evidence') || {}).textContent || '' })));
  ok('gaps: at least 10 rows', gaps.length >= 10, `${gaps.length} rows`);
  ok('gaps: every row has a valid status', gaps.every(g => ['addressed', 'partial', 'open'].includes(g.status)));
  ok('gaps: every row has evidence', gaps.every(g => g.evidence.trim().length > 20));
  // every number tagged with data-src must equal the value in summary.json
  const summary = JSON.parse(fs.readFileSync('/home/user/neo-risk/docs/data/summary.json', 'utf8'));
  const tagged = await e.$$eval('#gap-rows [data-src]', els => els.map(x => [x.dataset.src, x.dataset.val]));
  const lookup = src => { const [tab, model, col] = src.split(':');
    if (tab === 'metrics') return summary[model].metrics[col.split('.')[0]][col.split('.')[1]];
    if (tab === 'stage3') return col ? summary.stage3[model][col] : summary.stage3[model];
    return summary.tables[tab].find(r => r.model === model)[col]; };
  const bad = tagged.filter(([src, v]) => Math.abs(lookup(src) - +v) > 1e-9);
  ok('gaps: numbers match summary.json', tagged.length >= 12 && bad.length === 0, `${tagged.length} numbers, ${bad.length} mismatched ${bad.slice(0, 2)}`);
  await e.click('nav.tabs [data-tab="explore"]'); await e.waitForTimeout(200);   // later steps use the search box
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
  // W5a: CSV export of the filtered table (all rows, not just the visible 50)
  {
    const O = JSON.parse(fs.readFileSync('/home/user/neo-risk/docs/data/objects.json', 'utf8'));
    await e.click('nav.tabs [data-tab="explore"]');
    await e.click('#f-short'); await e.waitForTimeout(300);                       // filter: short arcs only
    const nShown = +(await e.textContent('#count')).match(/^([\d,]+)/)[1].replace(/,/g, '');
    const [dl] = await Promise.all([e.waitForEvent('download'), e.click('#csv')]);
    const csv = fs.readFileSync(await dl.path(), 'utf8').trim().split('\n');
    const hdr = csv[0].split(',');
    ok('CSV: one row per filtered object', csv.length - 1 === nShown, `${csv.length - 1} rows vs ${nShown}`);
    ok('CSV: header has designation and both scores', ['designation', 'p_full', 'p_leakfree', 'label'].every(h => hdr.includes(h)));
    const row = csv[1].match(/("([^"]|"")*"|[^,]*)(,|$)/g).map(x => x.replace(/,$/, '').replace(/^"|"$/g, '').replace(/""/g, '"'));
    const k = O.pdes.indexOf(row[hdr.indexOf('designation')]);
    ok('CSV: values match objects.json', k >= 0 && Math.abs(+row[hdr.indexOf('p_full')] - O.p_pha_full[k]) < 1e-6 && O.short[k] === 1,
      `${row[hdr.indexOf('designation')]} ${row[hdr.indexOf('p_full')]} vs ${O.p_pha_full[k]}`);
    await e.click('#f-short');
    // W5b: calibration chart, recomputed independently (10 equal-count bins of p_full, Mode A)
    await e.click('nav.tabs [data-tab="results"]'); await e.waitForTimeout(400);
    const pts = await e.$$eval('#calib [data-bin]', els => els.map(x => [x.dataset.model, +x.dataset.bin, +x.dataset.pred, +x.dataset.obs]));
    const full = pts.filter(q => q[0] === 'full');
    const idx = O.p_pha_full.map((p, i) => [p, i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const nB = 10, n = idx.length, exp = [];
    for (let b2 = 0; b2 < nB; b2++) { const sl = idx.slice(Math.floor(b2 * n / nB), Math.floor((b2 + 1) * n / nB));
      exp.push([sl.reduce((s, q) => s + q[0], 0) / sl.length, sl.reduce((s, q) => s + O.pha[q[1]], 0) / sl.length]); }
    ok('calibration: 10 bins per model', full.length === 10 && pts.length === 20, `${pts.length}`);
    ok('calibration: bins match an independent recomputation', full.every(([, b2, pr, obs]) => Math.abs(pr - exp[b2][0]) < 1e-9 && Math.abs(obs - exp[b2][1]) < 1e-9));
    // W5c: compare pinned objects
    await e.click('nav.tabs [data-tab="explore"]');
    await e.fill('#q', '101955'); await e.dispatchEvent('#q', 'input'); await e.waitForTimeout(400);
    await e.click('#tbl tbody tr[data-i]'); await e.click('#pin');
    await e.fill('#q', '99942'); await e.dispatchEvent('#q', 'input'); await e.waitForTimeout(400);
    await e.click('#tbl tbody tr[data-i]'); await e.click('#pin'); await e.waitForTimeout(200);
    const cmp = await e.$$eval('#compare [data-pdes]', els => els.map(x => [x.dataset.pdes, +x.dataset.pfull]));
    ok('compare: two pinned objects', cmp.length === 2 && cmp.map(c => c[0]).sort().join() === '101955,99942', JSON.stringify(cmp));
    ok('compare: scores match objects.json', cmp.every(([d, p]) => Math.abs(p - O.p_pha_full[O.pdes.indexOf(d)]) < 1e-9));
    await e.fill('#q', ''); await e.dispatchEvent('#q', 'input');
  }
  await e.click('#theme'); await e.waitForTimeout(200);
  ok('theme toggles', await e.evaluate(() => document.documentElement.getAttribute('data-theme')) !== null);
  await e.click('a[href="index.html"]'); await e.waitForURL(/index\.html/);
  ok('explorer → atlas link', true);
  await e.goto('http://localhost:8765/explorer.html'); await e.waitForSelector('#loading', { state: 'detached', timeout: 30000 });
  for (const t of ['explore', 'decide', 'results', 'leak', 'gaps']) {
    await e.click(`nav.tabs [data-tab="${t}"]`); await e.waitForTimeout(150);
    const un = await e.evaluate(() => [...document.querySelectorAll('button, input, select, a[href]')].filter(el => {
      if (!el.offsetWidth && !el.offsetHeight) return false;
      const lab = el.getAttribute('aria-label') || el.getAttribute('title') || (el.labels && el.labels[0] && el.labels[0].textContent) || el.textContent || el.placeholder;
      return !lab || !lab.trim(); }).map(el => el.outerHTML.slice(0, 80)));
    ok(`explorer ${t}: every visible control has an accessible name`, un.length === 0, un.slice(0, 2).join(' | '));
  }
  ok('explorer: no page errors', e.errs.length === 0, e.errs.slice(0, 3).join(' | '));
  await e.close();

  await b.close();
  console.log(results.join('\n'));
  console.log(`\n${results.length - fails}/${results.length} checks passed`);
  process.exit(fails ? 1 : 0);
})().catch(err => { console.log(results.join('\n')); console.error('CRASH', err.message); process.exit(2); });
