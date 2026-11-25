// Simulated usability + device testing for the NEO site (Chromium only).
// node tests/web/usability.js <three-dir> <out-dir>   -> <out-dir>/ux_report.json + screenshots
// A. device lab: errors, overflow, HUD overlaps, unreachable controls, tap-target sizes
// B. slow-CPU (4x throttle): main-thread ms per frame, hover-pick, search, filter, card render
// C. personas: scripted tasks with step counts, success and friction notes
const { chromium, devices } = require('playwright');
const path = require('path'), fs = require('fs');
const [THREE_DIR, OUT, MODE] = process.argv.slice(2);
const ASSERT = MODE === '--assert';
fs.mkdirSync(OUT, { recursive: true });
const R = { devices: [], explorer: [], perf: {}, personas: [] };
const BASE = 'http://localhost:8765/';

async function open(b, ctxOpts, url) {
  const ctx = await b.newContext(ctxOpts); const p = await ctx.newPage(); p.errs = [];
  p.on('pageerror', e => p.errs.push(e.message));
  await p.route(/three@0\.160\.0\/(.*)$/, r => r.fulfill({ path: path.join(THREE_DIR, r.request().url().match(/three@0\.160\.0\/(.*)$/)[1]), contentType: 'application/javascript' }));
  await p.route(/fonts\./, r => r.abort());
  await p.goto(BASE + url);
  if (url.startsWith('index')) await p.waitForFunction(() => window.__neo && __neo.ready, null, { timeout: 90000 });
  else await p.waitForSelector('#loading', { state: 'detached', timeout: 60000 });
  return { ctx, p };
}

// layout audit run inside the page
const AUDIT = (touch) => {
  const vis = e => { if (!e) return false; const cs = getComputedStyle(e); if (cs.display === 'none' || cs.visibility === 'hidden') return false;
    const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0 && e.closest('[hidden]') === null; };
  const out = { overflowX: document.documentElement.scrollWidth > innerWidth + 1, overlaps: [], unreachable: [], small: [], offscreen: [] };
  const panels = ['#brand', '#clock', '#side', '#card', '#timebar', '#timeline', '#enc', '#tour', '#fly', '#flybar'].map(s => [s, document.querySelector(s)]).filter(([, e]) => vis(e));
  for (let i = 0; i < panels.length; i++) for (let j = i + 1; j < panels.length; j++) {
    const a = panels[i][1].getBoundingClientRect(), b = panels[j][1].getBoundingClientRect();
    if (a.left < b.right - 1 && b.left < a.right - 1 && a.top < b.bottom - 1 && b.top < a.bottom - 1) out.overlaps.push(panels[i][0] + '/' + panels[j][0]); }
  const pageScrolls = document.documentElement.scrollHeight > innerHeight + 1 && getComputedStyle(document.body).overflowY !== 'hidden';
  for (const el of document.querySelectorAll('button, a[href], input, select, summary')) {
    if (!vis(el)) continue;
    const r = el.getBoundingClientRect(), name = (el.getAttribute('aria-label') || el.textContent || el.id || el.type || '').trim().slice(0, 30);
    // visible area = viewport intersected with every clipping ancestor; note whether any of them scrolls
    let L = 0, T = 0, Rr = innerWidth, B = innerHeight, scrollable = pageScrolls;
    for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) {
      const cs = getComputedStyle(a);
      if (cs.overflowX !== 'visible' || cs.overflowY !== 'visible') {
        const ar = a.getBoundingClientRect(); L = Math.max(L, ar.left); T = Math.max(T, ar.top); Rr = Math.min(Rr, ar.right); B = Math.min(B, ar.bottom);
        if (/(auto|scroll)/.test(cs.overflowX + cs.overflowY)) scrollable = true; }
    }
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    const inClip = cx >= L && cx <= Rr && cy >= T && cy <= B;
    if (!inClip) {                                   // out of sight: fine only if you can scroll to it
      const horizOut = r.right > innerWidth + 1 || r.left < -1;
      if (!scrollable || (horizOut && !pageScrolls && !/(auto|scroll)/.test(getComputedStyle(el.parentElement).overflowX + (el.closest('nav.tabs,.tbl-wrap') ? 'auto' : ''))))
        out.offscreen.push(name);
      continue;
    }
    const top = document.elementFromPoint(cx, cy);
    if (top && top !== el && !el.contains(top) && !(el.labels && [...el.labels].some(l => l.contains(top))))
      out.unreachable.push(name + ' <- ' + (top.id || top.className || top.tagName).toString().slice(0, 30));
    const lab = el.labels && el.labels[0] ? el.labels[0].getBoundingClientRect() : r;
    const w = Math.max(r.width, lab.width), h = Math.max(r.height, lab.height);
    if (touch && (w < 24 || h < 24)) out.small.push(`${name} ${Math.round(w)}x${Math.round(h)}`);
  }
  return out;
};

(async () => {
  const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  // ---------------------------------------------------------------- A. device lab
  const DEV = ['iPhone SE', 'iPhone 13', 'Pixel 7', 'Galaxy S9+', 'iPad Mini', 'iPad Pro 11', 'iPhone 13 landscape'];
  const DESK = [[960, 600], [1024, 768], [1280, 720], [1366, 768], [1920, 1080], [2560, 1440]];   // incl. short narrow screens
  const targets = [...DEV.map(d => [d, devices[d]]), ...DESK.map(([w, h]) => [`Desktop ${w}x${h}`, { viewport: { width: w, height: h } }])];
  for (const [name, opts] of targets) {
    const { ctx, p } = await open(b, opts, 'index.html');
    const touch = !!opts.hasTouch;
    const a0 = await p.evaluate(AUDIT, touch);
    await p.screenshot({ path: path.join(OUT, `dev_${name.replace(/\W+/g, '_')}.png`) });
    await p.evaluate(() => __neo.select(__neo.rowOf('99942'))); await p.waitForTimeout(2200);
    const a1 = await p.evaluate(AUDIT, touch);
    await p.screenshot({ path: path.join(OUT, `dev_${name.replace(/\W+/g, '_')}_card.png`) });
    await p.evaluate(() => __neo.enterFlyby(0)); await p.waitForTimeout(800);
    const a2 = await p.evaluate(AUDIT, touch);
    R.devices.push({ name, touch, viewport: opts.viewport, errors: p.errs, idle: a0, card: a1, flyby: a2 });
    await ctx.close();
  }
  for (const [name, opts] of [['iPhone SE', devices['iPhone SE']], ['Desktop 1366x768', { viewport: { width: 1366, height: 768 } }]]) {
    const { ctx, p } = await open(b, opts, 'explorer.html');
    const res = {};
    for (const t of ['explore', 'decide', 'results', 'leak', 'gaps']) { await p.click(`nav.tabs [data-tab="${t}"]`); await p.waitForTimeout(250); res[t] = await p.evaluate(AUDIT, !!opts.hasTouch); }
    R.explorer.push({ name, errors: p.errs, tabs: res });
    await p.screenshot({ path: path.join(OUT, `explorer_${name.replace(/\W+/g, '_')}.png`), fullPage: true });
    await ctx.close();
  }

  // ---------------------------------------------------------------- B. slow CPU (4x)
  {
    const { ctx, p } = await open(b, { viewport: { width: 1366, height: 768 } }, 'index.html');
    const cdp = await ctx.newCDPSession(p);
    const measure = async (label) => {
      await p.waitForTimeout(3000);
      const js = await p.evaluate(() => __neo.perf.jsMs);
      const pickMs = await p.evaluate(() => { const t = performance.now(); for (let i = 0; i < 5; i++) __neo.pick(600 + i, 380); return (performance.now() - t) / 5; });
      const t0 = Date.now(); await p.fill('#q', 'apo'); await p.waitForSelector('#results button'); const searchMs = Date.now() - t0;
      await p.fill('#q', '');
      const t1 = Date.now(); await p.click('#filters [data-f="3"]'); await p.waitForTimeout(50); const filterMs = Date.now() - t1 - 50; await p.click('#filters [data-f="0"]');
      const t2 = Date.now(); await p.evaluate(() => __neo.select(__neo.rowOf('99942'))); await p.waitForSelector('#card .drv', { timeout: 30000 }); const cardMs = Date.now() - t2;
      await p.keyboard.press('Escape');
      R.perf[label] = { jsMsPerFrame: +js.toFixed(2), hoverPickMs: +pickMs.toFixed(1), searchMs, filterMs, cardMs };
    };
    await measure('normal');
    await cdp.send('Emulation.setCPUThrottlingRate', { rate: 4 });
    await measure('cpu_4x_slower');
    await ctx.close();
  }

  // ---------------------------------------------------------------- C. personas
  const persona = async (who, device, goal, fn) => {
    const { ctx, p } = await open(b, device, fn.url || 'index.html');
    const log = { who, goal, steps: 0, success: false, friction: [], errors: [] };
    const step = async (desc, f) => { log.steps++; try { await f(); } catch (e) { log.friction.push(`step ${log.steps} (${desc}) failed: ${e.message.split('\n')[0].slice(0, 120)}`); throw e; } };
    try { await fn(p, step, log); } catch (e) { /* recorded */ }
    log.errors = p.errs;
    await p.screenshot({ path: path.join(OUT, `persona_${who.replace(/\W+/g, '_')}.png`) });
    R.personas.push(log); await ctx.close();
  };
  await persona('Public visitor (phone)', devices['iPhone 13'], 'Find out whether any asteroid will hit Earth soon', async (p, step, log) => {
    const words = (await p.textContent('body')).toLowerCase();
    if (!/impact|hit earth|no known/i.test(await p.innerText('#brand'))) log.friction.push('the opening panel does not say (visibly) what the risk to Earth actually is');
    await step('tap "Close encounters"', () => p.tap('#open-enc'));
    await step('tap "Next 24 months"', () => p.tap('#enc-tab-next'));
    const txt = await p.textContent('#enc');
    log.success = /lunar|LD/.test(txt);
    if (!/no (known )?impact|none .* impact|will not hit|miss/i.test(txt)) log.friction.push('the list never says plainly that none of these approaches is an impact');
    if (!/LD/.test(txt) || !/lunar distance/i.test(txt)) log.friction.push('"LD" is used without explaining it in the list itself');
  });
  await persona('Student (laptop)', { viewport: { width: 1366, height: 768 } }, 'Find Apophis and watch its 2029 flyby', async (p, step, log) => {
    await step('type "apophis" in search', () => p.fill('#q', 'apophis'));
    await step('press Enter', () => p.keyboard.press('Enter'));
    await p.waitForTimeout(2000);
    await step('click "Watch 2029 flyby"', () => p.click('#c-fly'));
    await p.waitForTimeout(500);
    await step('click "Jump to closest approach"', () => p.click('#f-ca'));
    await p.waitForTimeout(1500);
    log.success = await p.evaluate(() => __neo.mode === 'flyby');
    if (!(await p.isVisible('#fly'))) log.friction.push('flyby readouts not visible');
  });
  await persona('Astronomer (desktop)', { viewport: { width: 1920, height: 1080 }, acceptDownloads: true }, 'Download scores for short-arc objects on the Sentry list and check calibration', Object.assign(async (p, step, log) => {
    await step('tick "on the target list only"', () => p.click('#f-pos'));
    await step('tick "short arcs only"', () => p.click('#f-short'));
    const [dl] = await Promise.all([p.waitForEvent('download'), step('click "Download CSV"', () => p.click('#csv'))]);
    const rows = fs.readFileSync(await dl.path(), 'utf8').trim().split('\n').length - 1;
    await step('open the Results tab', () => p.click('nav.tabs [data-tab="results"]'));
    log.success = rows > 0 && await p.isVisible('#calib svg');
    log.notes = `${rows} rows downloaded`;
    if (!/version|generated|snapshot/i.test(fs.readFileSync(await dl.path(), 'utf8').split('\n')[0] + (await p.textContent('footer'))))
      log.friction.push('the download and the page do not state the data snapshot date or version (needed to cite it)');
  }, { url: 'explorer.html' }));
  await persona('Keyboard-only user', { viewport: { width: 1366, height: 768 } }, 'Search, select and read an asteroid without a mouse', async (p, step, log) => {
    let tabs = 0;
    for (; tabs < 40; tabs++) { await p.keyboard.press('Tab'); if (await p.evaluate(() => document.activeElement && document.activeElement.id === 'q')) break; }
    log.notes = `${tabs + 1} Tab presses to reach the search box`;
    if (tabs + 1 > 8) log.friction.push(`${tabs + 1} Tab presses before the search box (no skip link; "/" works but is only discoverable via "?")`);
    await step('type "bennu"', () => p.keyboard.type('bennu'));
    await step('press Enter', () => p.keyboard.press('Enter'));
    await p.waitForTimeout(1500);
    log.success = await p.evaluate(() => __neo.selected === __neo.rowOf('101955'));
    const focusInCard = await p.evaluate(() => document.querySelector('#card').contains(document.activeElement));
    if (!focusInCard) log.friction.push('after selecting, keyboard focus does not move to the card, so a screen-reader user is not taken to the result');
  });
  await persona('Sceptical reviewer (desktop)', { viewport: { width: 1366, height: 768 } }, 'Check whether the model leaks MOID/H and what is not solved', Object.assign(async (p, step, log) => {
    await step('open Leakage audit', () => p.click('nav.tabs [data-tab="leak"]'));
    const leak = await p.textContent('#tab-leak');
    await step('open Research gaps', () => p.click('nav.tabs [data-tab="gaps"]'));
    const n = await p.$$eval('#gap-rows [data-status="open"]', x => x.length);
    log.success = /0\.99/.test(leak) && n > 0;
    log.notes = `${n} open gaps listed`;
    if (!/github|source code|repository/i.test(await p.textContent('body'))) log.friction.push('no link to the source code or method write-up, so claims cannot be inspected');
  }, { url: 'explorer.html' }));

  await b.close();
  fs.writeFileSync(path.join(OUT, 'ux_report.json'), JSON.stringify(R, null, 1));
  console.log('wrote', path.join(OUT, 'ux_report.json'));
  if (ASSERT) {           // strict mode for loop/verify.sh: any finding fails
    const bad = [];
    const chkA = (where, a) => {
      if (a.overflowX) bad.push(`${where}: horizontal overflow`);
      if (a.overlaps.length) bad.push(`${where}: overlaps ${a.overlaps}`);
      if (a.unreachable.length) bad.push(`${where}: unreachable ${a.unreachable.slice(0, 3).join('; ')}`);
      if (a.offscreen.length) bad.push(`${where}: off-screen ${a.offscreen.slice(0, 3).join('; ')}`);
      if (a.small.length) bad.push(`${where}: small targets ${a.small.slice(0, 3).join('; ')}`); };
    for (const d of R.devices) { for (const st of ['idle', 'card', 'flyby']) chkA(`${d.name}/${st}`, d[st]); if (d.errors.length) bad.push(`${d.name}: errors ${d.errors[0]}`); }
    for (const e of R.explorer) { for (const [t, a] of Object.entries(e.tabs)) chkA(`explorer ${e.name}/${t}`, a); if (e.errors.length) bad.push(`explorer ${e.name}: errors`); }
    const slow = R.perf.cpu_4x_slower || {};
    if (!(slow.hoverPickMs < 30)) bad.push(`hover pick ${slow.hoverPickMs} ms at 4x CPU (limit 30)`);
    if (!(slow.jsMsPerFrame < 8)) bad.push(`main thread ${slow.jsMsPerFrame} ms/frame at 4x CPU (limit 8)`);
    for (const p of R.personas) { if (!p.success) bad.push(`persona ${p.who}: task failed`); for (const f of p.friction) bad.push(`persona ${p.who}: ${f}`); if (p.errors.length) bad.push(`persona ${p.who}: errors`); }
    console.log(bad.length ? 'UX FAIL\n  ' + bad.join('\n  ') : 'UX PASS');
    process.exit(bad.length ? 1 : 0);
  }
})().catch(e => { console.error('CRASH', e); fs.writeFileSync(path.join(OUT, 'ux_report.json'), JSON.stringify(R, null, 1)); process.exit(2); });
