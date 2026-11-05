// Checks the C++ core (as bgcore.wasm, built from cpp/) against engine.js/bot.js/analysis.js:
// on positions from random-ish games of every rule, the end states of each segment and their values,
// the game flow after each move, the net value of each position, TD(lambda) steps, the multi-ply
// search values, and that the alpha-beta cutoffs change neither values nor best plans.
//   node check_cpp.js [games per rule]
// The core computes in float (JS in double): values are compared to 2e-5.
const fs = require('fs');
const BG = require('./engine.js');
const L = require('./bot.js');
const A = require('./analysis.js');
const C = require('./core_wasm.js');

if (!C.ready()) { console.log('bgcore.wasm missing: run make in cpp/'); process.exit(1); }
const ex = C.exports();
const GAMES = +process.argv[2] || 30;
const TOL = 2e-5;
const RULE_ID = { paul: 0, rene: 1, mixed: 2 };
const phaseName = ['roll', 'move', 'walk', 'over'];
const io = () => new Int32Array(ex.memory.buffer, ex.ioPtr(), 64);
const HIDDEN = { mixed: 256, paul: 64, rene: 128 }; // random nets: every compiled hidden size once
const boardKey = (b, bar, off) => Array.from(b).join(',') + '|' + Array.from(bar) + '|' + Array.from(off);
let fails = 0, leaves = 0, positions = 0, searches = 0, tdSteps = 0;
const fail = (msg) => { if (fails++ < 10) console.log('FAIL', msg); };

function sameState(s, where) {
  const o = io(), dice = [];
  for (let d = 1; d <= 6; d++) for (let c = 0; c < o[32 + d]; c++) dice.push(d);
  const a = [boardKey(o.slice(3, 27), [o[27], o[28]], [o[29], o[30]]), o[31], phaseName[o[32]], dice.join(''), o[32] === 3 ? o[43] : -1, o[2]].join(' ');
  const b = [boardKey(s.board.slice(1), s.bar, s.off), s.turn, s.phase, s.dice.slice().sort().join(''), s.phase === 'over' ? (s.winner === null ? -1 : s.winner) : -1, s.turnCount].join(' ');
  if (a !== b) fail(`${where}: core ${a}\n   engine ${b}`);
}

// ---------------------------------------------------------------- rules, end states, 1-ply values
const samples = { paul: [], rene: [], mixed: [] }; // positions to roll or to move, for the search checks
for (const rule of ['mixed', 'paul', 'rene']) {
  const net = L.newNet(rule, HIDDEN[rule]);
  for (let i = 0; i < net.W1.length; i++) net.W1[i] *= 5; // sharper random net: more varied values
  C.useNet(rule, net);
  ex.tdInit(RULE_ID[rule], 0.7);
  for (let gi = 0; gi < GAMES; gi++) {
    const limit = rule === 'mixed' ? (gi % 3 === 0 ? 25 : 200) : 0;
    const s = BG.newGame(rule, { plyLimit: limit });
    s.log = { push() {} };
    for (let ply = 0; ply < 3000 && s.phase !== 'over'; ply++) {
      positions++;
      C.load(s);
      const vf = ex.tdForward(), ve = net.forward(L.features(s));
      if (Math.abs(vf - ve) > TOL) fail(`${rule} forward ${vf} vs ${ve}`);
      if (Math.random() < 0.02) samples[rule].push(L.cloneState(s));
      if (s.phase === 'roll') {
        const t = [1 + Math.floor(Math.random() * 6), 1 + Math.floor(Math.random() * 6)];
        BG.roll(s, t); ex.roll(t[0], t[1]);
        sameState(s, rule + ' roll');
        continue;
      }
      const ref = new Map();
      const ends = L.endStates(s);
      for (const e of ends) ref.set(boardKey(e.s.board.slice(1), e.s.bar, e.s.off), L.walkValue(net, e.s, s.turn));
      const n = ex.ends(), out = new Float32Array(ex.memory.buffer, ex.foutPtr(), n * 29), got = new Map();
      for (let i = 0; i < n; i++) {
        const o = out.subarray(i * 29, i * 29 + 29);
        got.set(boardKey(o.slice(0, 24), [o[24], o[25]], [o[26], o[27]]), o[28]);
      }
      leaves += ref.size;
      if (got.size !== ref.size) fail(`${rule} ${got.size} ends vs ${ref.size}`);
      for (const [k, v] of ref) {
        if (!got.has(k)) fail(`${rule} missing end ${k}`);
        else if (Math.abs(got.get(k) - v) > TOL) fail(`${rule} end value ${got.get(k)} vs ${v} at ${k}`);
      }
      const path = ends[Math.floor(Math.random() * ends.length)].path;
      for (const m of path) {
        BG.applyMove(s, m); ex.applyMove(C.encode(m));
        sameState(s, rule + ' move');
      }
    }
  }
}

// ---------------------------------------------------------------- TD(lambda) vs plain trace arithmetic
for (const rule of ['mixed', 'paul', 'rene']) {
  const net = L.newNet(rule, HIDDEN[rule]), KEYS = ['W1', 'b1', 'W2', 'b2'];
  C.useNet(rule, net);
  ex.tdInit(RULE_ID[rule], 0.7);
  const E = {}; for (const k of KEYS) E[k] = new Float64Array(net[k].length);
  const s = BG.newGame(rule); s.log = { push() {} };
  for (let ply = 0; ply < 400 && s.phase !== 'over'; ply++) {
    if (s.phase === 'roll' || s.phase === 'walk') {
      const sign = s.turn === 0 ? 1 : -1, d = (Math.random() - 0.5) * 0.1;
      const f = L.features(s); net.forward(f);
      for (const k of KEYS) for (let i = 0; i < E[k].length; i++) E[k][i] *= 0.7;
      net.addGrad(f, net.y[0], sign, E);
      for (const k of KEYS) for (let i = 0; i < E[k].length; i++) net[k][i] += d * E[k][i];
      C.load(s); ex.tdStep(sign, d);
      tdSteps++;
    }
    if (s.phase === 'roll') BG.roll(s);
    else { const e = L.endStates(s); for (const m of e[Math.floor(Math.random() * e.length)].path) BG.applyMove(s, m); }
  }
  const r = RULE_ID[rule], H = net.H, got = {
    W1: new Float32Array(ex.memory.buffer, ex.w1Ptr(r), net.W1.length), b1: new Float32Array(ex.memory.buffer, ex.b1Ptr(r), H),
    W2: new Float32Array(ex.memory.buffer, ex.w2Ptr(r), H), b2: new Float32Array(ex.memory.buffer, ex.b2Ptr(r), 1),
  };
  let md = 0;
  for (const k of KEYS) for (let i = 0; i < net[k].length; i++) md = Math.max(md, Math.abs(net[k][i] - got[k][i]));
  if (md > TOL) fail(`${rule} TD weights differ by ${md}`);
}

// ---------------------------------------------------------------- multi-ply search on the trained nets
const DEPTHS = { paul: [1, 2, 3], rene: [1, 2], mixed: [1, 2] };
for (const rule of ['paul', 'rene', 'mixed']) {
  const file = __dirname + '/weights_' + rule + '.json';
  if (!fs.existsSync(file)) continue;
  const net = new L.Net(JSON.parse(fs.readFileSync(file)));
  for (const s0 of samples[rule].slice(0, rule === 'mixed' ? 4 : 6)) {
    const s = L.cloneState(s0);
    if (s.phase === 'roll') BG.roll(s);
    if (s.phase !== 'move' && s.phase !== 'walk') continue;
    const ends = L.endStates(s), p = s.turn;
    for (const depth of DEPTHS[rule]) {
      if (depth === 3 && ends.length > 30) continue; // analysis.js is slow there
      // a few end states against analysis.js, with and without cutoffs
      for (const e of ends.slice(0, depth === 1 ? 20 : 2)) {
        const ref = A.evalStateJS(net, e.s, p, depth);
        C.useNet(rule, net); C.load(e.s);
        const v1 = ex.evalNode(p, depth, 1), v0 = ex.evalNode(p, depth, 0);
        searches++;
        if (Math.abs(v1 - ref) > TOL || Math.abs(v0 - ref) > TOL) fail(`${rule} depth ${depth}: core ${v1} / ${v0} vs analysis ${ref}`);
      }
      // best plan with root pruning = best plan without
      C.load(s);
      const vp = ex.bestPlan(depth, 1), pp = io().slice(48, 53).join();
      C.load(s);
      const vn = ex.bestPlan(depth, 0), pn = io().slice(48, 53).join();
      if (Math.abs(vp - vn) > TOL || (pp !== pn && Math.abs(vp - vn) > 0)) fail(`${rule} depth ${depth} best plan: pruned ${vp} ${pp} vs full ${vn} ${pn}`);
      if (depth > 1) { // and = the maximum of the core's exact values of all ends
        let mx = -1;
        for (const e of ends) { C.load(e.s); mx = Math.max(mx, ex.evalNode(p, depth, 1)); }
        if (Math.abs(mx - vp) > TOL) fail(`${rule} depth ${depth} best plan ${vp} vs max over ends ${mx}`);
      }
    }
  }
}
// ---------------------------------------------------------------- deep net (residual layers): values and searches
{
  const file = __dirname + '/weights_mixed_big.json';
  if (fs.existsSync(file)) {
    const net = new L.Net(JSON.parse(fs.readFileSync(file)));
    if (!net.deep) fail('weights_mixed_big.json has no residual layers');
    for (const s0 of samples.mixed.slice(0, 6)) {
      const s = L.cloneState(s0);
      C.useNet('mixed', net); ex.tdInit(RULE_ID.mixed, 0.7); C.load(s);
      const vf = ex.tdForward(), ve = net.forward(L.features(s));
      positions++;
      if (Math.abs(vf - ve) > TOL) fail(`deep forward ${vf} vs ${ve}`);
      if (s.phase === 'roll') BG.roll(s);
      if (s.phase !== 'move' && s.phase !== 'walk') continue;
      for (const e of L.endStates(s).slice(0, 3)) for (const depth of [1, 2]) {
        const ref = A.evalStateJS(net, e.s, s.turn, depth);
        C.useNet('mixed', net); C.load(e.s);
        const v = ex.evalNode(s.turn, depth, 1);
        searches++;
        if (Math.abs(v - ref) > TOL) fail(`deep depth ${depth}: core ${v} vs analysis ${ref}`);
      }
    }
  }
}
// ---------------------------------------------------------------- pool.js (inline here: node has no Web Workers)
// its split root search = the core's best plan; its value of one end = the core's exact value
(async () => {
  const pool = require('./pool.js').create(1);
  for (const rule of ['paul', 'rene', 'mixed']) {
    const file = __dirname + '/weights_' + rule + '.json';
    if (!fs.existsSync(file)) continue;
    const net = new L.Net(JSON.parse(fs.readFileSync(file)));
    for (const s0 of samples[rule].slice(0, 4)) {
      const s = L.cloneState(s0);
      if (s.phase === 'roll') BG.roll(s);
      if (s.phase !== 'move' && s.phase !== 'walk') continue;
      for (const depth of [1, 2]) {
        const ref = C.bestPlan(rule, net, s, depth), got = await pool.best(rule, net, s, depth, 0).promise;
        const e = L.cloneState(s);
        for (const m of got.path) BG.applyMove(e, m);
        searches++;
        if (Math.abs(ref.value - got.value) > TOL || L.key(e) !== got.key) fail(`${rule} depth ${depth} pool best ${got.value} vs core ${ref.value}`);
        const v = await pool.value(rule, net, e, s.turn, depth, 0).promise, vr = depth > 1 ? C.evalState(rule, net, e, s.turn, depth) : got.value;
        if (Math.abs(v - vr) > TOL) fail(`${rule} depth ${depth} pool value ${v} vs core ${vr}`);
      }
    }
  }
  // rollout bot: rollRange is deterministic and in [0, 1]; rolloutBest plays one of the segment's ends
  {
    const net = new L.Net(JSON.parse(fs.readFileSync(__dirname + '/weights_mixed.json')));
    for (const s0 of samples.mixed.slice(0, 3)) {
      const s = L.cloneState(s0);
      if (s.phase === 'roll') BG.roll(s);
      if (s.phase !== 'move' && s.phase !== 'walk') continue;
      const ends = L.endStates(s).slice(0, 2).map((e) => require('./pool.js').pack(e.s));
      const r1 = C.rollRange('mixed', net, ends, s.turn, 5, 6, 1, 2, 8, 800), r2 = C.rollRange('mixed', net, ends, s.turn, 5, 6, 1, 2, 8, 800);
      if (JSON.stringify(r1) !== JSON.stringify(r2) || r1.flat().some((x) => !(x >= 0 && x <= 1))) fail(`rollRange ${JSON.stringify(r1)} vs ${JSON.stringify(r2)}`);
      // stratified: games 0..39 (the last 4 reuse the play up to the second throw), cold, then warm
      const s1 = C.rollRange('mixed', net, ends, s.turn, 0, 40, 3, 4, 8, 800, true), s2 = C.rollRange('mixed', net, ends, s.turn, 0, 40, 3, 4, 8, 800, true);
      if (JSON.stringify(s1) !== JSON.stringify(s2) || s1.flat().some((x) => !(x >= 0 && x <= 1))) fail(`rollRange strat ${JSON.stringify(s1)} vs ${JSON.stringify(s2)}`);
      const keys = L.endStates(s).map((e) => L.key(e.s));
      for (const opt of [{}, { replies: true }]) {
        const got = await pool.rolloutBest('mixed', net, s, opt.replies ? 1500 : 300, 0, null, opt).promise;
        searches++;
        if (!got || !keys.includes(got.key) || !(got.value >= 0 && got.value <= 1)) fail(`rolloutBest ${JSON.stringify(opt)} ${got && got.key} value ${got && got.value}`);
      }
    }
  }
  console.log(`${positions} positions, ${leaves} end states, ${tdSteps} TD steps, ${searches} search values compared, ${fails} failures`);
  process.exit(fails ? 1 : 0);
})();
