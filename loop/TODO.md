# Loop task list (one item per iteration, top to bottom)

- [x] R1 **Physical priors cited.** Set the TAXON_PRIORS values from Thomas et al. 2011 (NEO albedo) and Carry 2012 (density), with citations in the code. Verify with a test that pins the cited values, and a check that the model's predictions don't change (the albedo order of the types is preserved).
- [x] R2 **Research-gap audit, in the explorer.** A "Research gaps" tab with one row per gap (the four from the original brief, plus the gaps found since). Each row gives the evidence, read live from summary.json; a status (addressed, partial or open); and what remains. Verify: e2e checks that the tab renders, that every row has a status and evidence, and that its numbers match summary.json.
- [x] W1 **"Why this score" explanations.** Per-object LightGBM SHAP-style contributions (pred_contrib) for the full model in both modes, with the top 5 drivers shown in the atlas card and the explorer detail. Verify: a Python check that each object's contributions sum to its logit; e2e checks that the drivers render.
- [x] W2 **Atlas depth and scale.** An ecliptic grid with 1/2/3 au rings and labels, a "Top-risk orbits" overlay toggle (orbit lines of the top 50), and Sun corona and lens glow polish. Verify: e2e checks for the toggles, line counts and no errors.
- [x] W3 **Timeline.** A scrubber spanning 1950–2100, with flyby event markers you can click to jump to, plus a date input. Verify: e2e checks that a date input sets the simulation time and that a marker opens its date.
- [x] W4 **Shareable views.** The URL encodes the time, selection, filter, colour mode and camera, with a "Copy link" button. Verify: e2e reloads a crafted URL and checks the restored state.
- [x] W5 **Explorer professional tools.** CSV export of the filtered table, a calibration (reliability) chart, and a side-by-side comparison of two objects. Verify: e2e checks the download content, the chart and the comparison.
- [x] W6 **Quality and accessibility.** Adaptive resolution when the frame rate drops, a keyboard shortcut overlay (?), an accessible name on every control, and visible focus outlines. Verify: e2e checks for unnamed buttons (none allowed), that "?" opens help, and the resolution step-down logic.
- [x] W7 **Guided tour.** Five steps (Sun and NEOs → risk colour → Apophis → flyby → explorer). Verify: e2e walks every step and checks for no errors.

## Round 2 (branch loop/site-v3)
- [x] V1 **Discovery history.** A toggle that shows only the asteroids known on the simulation date (first observation on or before it), with a live counter. Verify: the counter for 2000-01-01 equals a count computed from objects.json.
- [x] V2 **Uncertainty cloud.** Each selected high-risk asteroid shows its ~32 covariance clones, anchored to the displayed position. Verify in data: each cloud's centre equals that object's elements in orbits.bin. Verify in e2e: the clones are present, and their spread grows over 20 years.
- [x] V3 **Upcoming close approaches** (JPL cad.api, next 24 months, <= 0.05 au). Shown as a list next to the featured flybys; clicking an entry jumps to its date and selects the asteroid. Verify: the list is sorted and within 0.05 au, and a click sets the time and the selection.
- [x] V4 **Earth-centred view.** A frame switch that keeps Earth at the origin. Verify: in the Earth frame the scene origin equals Earth's position, and switching back restores the Sun-centred frame.

## Round 3: simulated usability (branch loop/site-v4); verifier adds tests/web/usability.js --assert
- [x] U1 **Layout on every device.** Phones get a compact time bar (no off-screen controls; the date and Copy link move to the menu sheet). The card fits under the brand panel. Short screens get a compact brand panel. Tablets up to 900 px wide use the phone layout. The encounters panel clears the time bar on 1280–1366 px laptops. Touch targets are at least 24 px.
- [x] U2 **GPU picking.** Hover and click use an ID render of a 15×15 px area instead of 43k CPU Kepler solves. Target: < 30 ms at 4× CPU throttle.
- [ ] U3 **Content found by the personas.** A plain-language risk statement (the Torino and Palermo maxima, read live from the data); "near misses, not impacts", with LD spelled out; the data snapshot date on both pages and in the CSV; a source-code link; keyboard focus moves to the card after a search.
- [ ] U4 **Faster first card on phones.** Split explain.json into 1,000-object chunks.
