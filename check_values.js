// Sanity checks for the values the page shows (analysis bar, analysis list). Runs in well under a minute:
//   node check_values.js
// 1. perspective: value(0) + value(1) = 1 for one position, at depth 1 and 2
// 2. the analysis depth is what the bar shows: a position without dice is searched like every analysed segment
//    (the raw net alone, depth 1, is far off in ordinary openings; see index.html rollValue)
// 3. early game: depth-2 values of the opening positions of self-play games stay near 50%
// 4. the bar series along a game (roll position and best value of each segment, all depth 2) has no big jumps
// 5. wasm core and the JS twin (analysis.js evalStateJS) agree at depth 1 and 2
// 6. a windowed search agrees with the full-window value: exact inside the window, a bound outside
const BG = require('./engine.js'), L = require('./bot.js'), A = require('./analysis.js'), C = require('./core_wasm.js');
const net = new L.Net(require('./weights_mixed.json'));
let fails = 0;
const check = (ok, msg) => { if (!ok) { fails++; console.log('FAIL', msg); } };
if (!C.ready()) { console.log('FAIL bgcore.wasm not loaded'); process.exit(1); }

let seed = 99;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const rd = () => [1 + Math.floor(rnd() * 6), 1 + Math.floor(rnd() * 6)];
const mixed = () => { const s = BG.newGame('mixed'); s.log = { push() {} }; return s; };

// self-play games at depth 1 (fast); records the position at every roll and every segment start
const games = [];
for (let g = 0; g < 8; g++) {
  const s = mixed(), rolls = [], segs = [];
  for (let n = 0; n < 60 && s.phase !== 'over'; n++) {
    if (s.phase === 'roll') { rolls.push({ i: segs.length, s: L.cloneState(s) }); BG.roll(s, rd()); continue; }
    if (n < 24) segs.push(L.cloneState(s));
    const r = C.bestPlan('mixed', net, s, 1);
    for (const m of r.path) { if (s.phase !== 'move' && s.phase !== 'walk') break; BG.applyMove(s, m); }
    if (s.phase === 'move' || s.phase === 'walk') BG.applyMove(s, L.bestPlan(net, s).path[0]);
  }
  games.push({ rolls, segs });
}

// 1. perspective
for (const G of games) for (const r of G.rolls.slice(0, 6)) for (const d of [1, 2]) {
  const a = C.evalState('mixed', net, r.s, 0, d), b = C.evalState('mixed', net, r.s, 1, d);
  check(Math.abs(a + b - 1) < 1e-4, `perspective sum ${a} + ${b} at depth ${d}`);
}

// 2 + 3. early game: depth-2 value of the first 4 roll positions of every game within 30..70%
let worstRaw = 0;
for (const G of games) for (const r of G.rolls.slice(0, 4)) {
  const v2 = C.evalState('mixed', net, r.s, 0, 2), raw = L.valueFor(net, r.s, 0);
  worstRaw = Math.max(worstRaw, Math.abs(raw - v2));
  check(v2 > 0.30 && v2 < 0.70, `early position at depth 2 is ${(v2 * 100).toFixed(1)}%: ${L.key(r.s)}`);
}
console.log(`largest |raw net - depth 2| over the early positions: ${(worstRaw * 100).toFixed(1)} points`);

// 4. the series the bar shows for a human who plays the analysed best move: no jumps over 25 points
for (const G of games.slice(0, 3)) {
  const s = mixed(); let prev = null, worst = 0;
  const push = (v) => { if (prev !== null) worst = Math.max(worst, Math.abs(v - prev)); prev = v; };
  for (let n = 0; n < 30 && s.phase !== 'over'; n++) {
    if (s.phase === 'roll') { push(C.evalState('mixed', net, s, 0, 2)); BG.roll(s, rd()); continue; }
    if (s.turn === 0) { // an analysed segment of White: the bar shows its best value
      const res = C.bestPlan('mixed', net, s, 2); // = the best value of the analysis, found with cutoffs
      push(res.value);
      for (const m of res.path) { if (s.phase !== 'move' && s.phase !== 'walk') break; BG.applyMove(s, m); }
    } else {
      const r = C.bestPlan('mixed', net, s, 1);
      for (const m of r.path) { if (s.phase !== 'move' && s.phase !== 'walk') break; BG.applyMove(s, m); }
    }
  }
  check(worst < 0.25, `bar series jumps by ${(worst * 100).toFixed(1)} points`);
}

// 5. wasm vs JS
let maxd = 0;
for (const G of games.slice(0, 2)) for (const r of G.rolls.slice(0, 2)) for (const d of [1, 2]) {
  maxd = Math.max(maxd, Math.abs(A.evalStateJS(net, L.cloneState(r.s), 0, d) - C.evalState('mixed', net, r.s, 0, d)));
}
for (const G of games.slice(0, 1)) for (const st of G.segs.slice(0, 2)) {
  const p = st.turn;
  for (const e of L.endStates(L.cloneState(st)).slice(0, 2)) {
    maxd = Math.max(maxd, Math.abs(A.evalStateJS(net, L.cloneState(e.s), p, 2) - C.evalState('mixed', net, e.s, p, 2)));
  }
}
console.log(`largest JS/wasm difference: ${maxd.toExponential(2)}`);
check(maxd < 1e-4, `JS and wasm differ by ${maxd}`);

// 6. windows
let nIn = 0, nLo = 0, nHi = 0;
for (const G of games.slice(0, 4)) for (const r of G.rolls.slice(1, 4)) {
  const full = C.evalState('mixed', net, r.s, 0, 2);
  for (const [al, be] of [[full - 0.1, full + 0.1], [full + 0.05, full + 0.3], [full - 0.3, full - 0.05], [-1, 2]]) {
    const w = C.evalWindow('mixed', net, r.s, 0, 2, al, be);
    if (w > al + 1e-6 && w < be - 1e-6) { nIn++; check(Math.abs(w - full) < 1e-4, `window value ${w} inside (${al}, ${be}) but full is ${full}`); }
    else if (w <= al + 1e-6) { nLo++; check(full <= w + 1e-4, `window upper bound ${w} below full ${full}`); }
    else { nHi++; check(full >= w - 1e-4, `window lower bound ${w} above full ${full}`); }
  }
}
console.log(`windows: ${nIn} exact, ${nLo} upper bounds, ${nHi} lower bounds`);

console.log(fails ? `${fails} FAILURES` : 'check_values: all ok');
process.exit(fails ? 1 : 0);
