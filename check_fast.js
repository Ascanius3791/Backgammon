// Checks fast.js against engine.js/bot.js: on positions from random-ish games of every rule, the set of
// end states and their values, the game flow after each move, and the net value of each position.
//   node check_fast.js [games per rule]
const BG = require('./engine.js');
const L = require('./bot.js');
const FA = require('./fast.js');

const GAMES = +process.argv[2] || 40;
const toEnc = (m) => (m.from === 'bar' ? FA.BAR : m.from) | ((m.to === 'off' ? FA.OFF : m.to) << 5) | (m.die << 10) | ((m.back ? 1 : 0) << 13);
const phaseName = ['roll', 'move', 'walk', 'over'];
const boardKey = (b, bar, off) => Array.from(b).slice(1).join(',') + '|' + Array.from(bar) + '|' + Array.from(off);
let fails = 0, leaves = 0, positions = 0;
const fail = (msg) => { if (fails++ < 10) console.log('FAIL', msg); };

function sameState(g, s, where) {
  const F = g.F, dice = [];
  for (let d = 1; d <= 6; d++) for (let c = 0; c < F.cnt[d]; c++) dice.push(d);
  const a = [boardKey(F.b, F.bar, F.off), g.turn, phaseName[g.phase], dice.join(''), g.phase === FA.OVER ? g.winner : -1, g.turnCount].join(' ');
  const b = [boardKey(s.board, s.bar, s.off), s.turn, s.phase, s.dice.slice().sort().join(''), s.phase === 'over' ? s.winner : -1, s.turnCount].join(' ');
  if (a !== b) fail(`${where}: fast ${a}\n   engine ${b}`);
}

for (const rule of ['mixed', 'paul', 'rene']) {
  const net = L.newNet(rule);
  for (let i = 0; i < net.W1.length; i++) net.W1[i] *= 5; // sharper random net: more varied values
  const W = FA.makeWeights(net);
  const td = new FA.TD(W, 0.7);
  for (let gi = 0; gi < GAMES; gi++) {
    const limit = rule === 'mixed' ? (gi % 3 === 0 ? 25 : 200) : 0;
    const s = BG.newGame(rule, { plyLimit: limit });
    s.log = { push() {} };
    const g = new FA.Game(rule, W, { plyLimit: limit });
    for (let ply = 0; ply < 3000 && s.phase !== 'over'; ply++) {
      positions++;
      g.fromEngine(s);
      // value of the position to roll / to walk
      const vf = td.forward(g), ve = net.forward(L.features(s));
      if (Math.abs(vf - ve) > 1e-9) fail(`${rule} forward ${vf} vs ${ve}`);
      if (s.phase === 'roll') {
        const t = [1 + Math.floor(Math.random() * 6), 1 + Math.floor(Math.random() * 6)];
        BG.roll(s, t); g.roll(t[0], t[1]);
        sameState(g, s, rule + ' roll');
        continue;
      }
      // end states and values
      const ref = new Map();
      const ends = L.candidates(net, s);
      for (const e of ends) ref.set(boardKey(e.s.board, e.s.bar, e.s.off), e.value);
      const got = new Map();
      g.onLeaf = (C, v) => got.set(boardKey(C.b, C.bar, C.off), v);
      g.search();
      g.onLeaf = null;
      leaves += ref.size;
      if (got.size !== ref.size) fail(`${rule} ${got.size} leaves vs ${ref.size}`);
      for (const [k, v] of ref) {
        if (!got.has(k)) fail(`${rule} missing leaf ${k}`);
        else if (Math.abs(got.get(k) - v) > 1e-9) fail(`${rule} leaf value ${got.get(k)} vs ${v} at ${k}`);
      }
      // play a random end state in both engines, move by move
      const path = ends[Math.floor(Math.random() * ends.length)].path;
      for (const m of path) {
        BG.applyMove(s, m); g.applyMove(toEnc(m));
        sameState(g, s, rule + ' move');
      }
    }
  }
}
// TD(lambda): train.js's scaled trace + one dense update vs train_old.js's plain trace arithmetic
let tdSteps = 0;
for (const rule of ['mixed', 'paul']) {
  const net = L.newNet(rule), W = FA.makeWeights(net), td = new FA.TD(W, 0.7);
  const KEYS = ['W1', 'b1', 'W2', 'b2'];
  const E = {}; for (const k of KEYS) E[k] = new Float64Array(net[k].length);
  const s = BG.newGame(rule); s.log = { push() {} };
  const g = new FA.Game(rule, W);
  for (let ply = 0; ply < 400 && s.phase !== 'over'; ply++) {
    if (s.phase === 'roll' || s.phase === 'walk') {
      g.fromEngine(s);
      const sign = s.turn === 0 ? 1 : -1;
      const f = L.features(s); net.forward(f);
      for (const k of KEYS) for (let i = 0; i < E[k].length; i++) E[k][i] *= 0.7;
      net.addGrad(f, net.y[0], sign, E);
      td.forward(g); td.trace(g, sign);
      const d = (Math.random() - 0.5) * 0.1;
      for (const k of KEYS) for (let i = 0; i < E[k].length; i++) net[k][i] += d * E[k][i];
      td.update(d);
      tdSteps++;
    }
    if (s.phase === 'roll') BG.roll(s);
    else { const e = L.endStates(s); for (const m of e[Math.floor(Math.random() * e.length)].path) BG.applyMove(s, m); }
  }
  let md = 0;
  for (const k of KEYS) for (let i = 0; i < net[k].length; i++) md = Math.max(md, Math.abs(net[k][i] - W[k][i]));
  if (md > 1e-9) fail(`${rule} TD weights differ by ${md}`);
}
console.log(`${positions} positions, ${leaves} end states, ${tdSteps} TD steps compared, ${fails} failures`);
process.exit(fails ? 1 : 0);
