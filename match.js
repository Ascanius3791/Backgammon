// Head-to-head of two Mixed nets from two checkouts, on shared dice: every pair of games uses one dice
// sequence and swaps the colours. engine.js is the arbiter; each side picks its moves with its own fast.js.
//   node match.js [pairs] [workers]
//   NEW=weights_mixed.json  OLD_DIR=../Bagammon  OLD=weights_mixed.json  LIMIT=200
// OLD_DIR is a checkout whose fast.js/bot.js/weights match the old net (e.g. main with the 253-input Mixed net).
const fs = require('fs');
const path = require('path');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const BG = require('./engine.js');

const toMove = (m) => {
  const f = m & 31, t = (m >> 5) & 31;
  return { from: f === 25 ? 'bar' : f, to: t === 0 ? 'off' : t, die: (m >> 10) & 7, back: !!((m >> 13) & 1) };
};

// small seeded generator (mulberry32) so a pair's dice depend only on its number
function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

function play(throws, newIsWhite, gN, gO, limit) {
  const s = BG.newGame('mixed', { plyLimit: limit });
  s.log = { push() {} };
  let r = 0;
  for (let steps = 0; s.phase !== 'over' && steps < 20000; steps++) {
    if (s.phase === 'roll') { BG.roll(s, throws[r++ % throws.length]); continue; }
    const g = (s.turn === 0) === newIsWhite ? gN : gO;
    g.fromEngine(s);
    g.search();
    for (let k = 0; k < g.bestLen && (s.phase === 'move' || s.phase === 'walk'); k++) BG.applyMove(s, toMove(g.bestPath[k]));
  }
  return { unfinished: s.phase !== 'over', newWon: s.phase === 'over' && (s.winner === 0) === newIsWhite, limitHit: !!s.limitHit };
}

function setup(o) {
  const FN = require('./fast.js'), FO = require(path.resolve(o.oldDir, 'fast.js'));
  const wN = FN.makeWeights(JSON.parse(fs.readFileSync(o.newFile)));
  const wO = FO.makeWeights(JSON.parse(fs.readFileSync(o.oldFile)));
  return { gN: new FN.Game('mixed', wN, { plyLimit: o.limit }), gO: new FO.Game('mixed', wO, { plyLimit: o.limit }) };
}

if (!isMainThread) {
  const o = workerData, { gN, gO } = setup(o);
  for (let p = o.from; p < o.to; p++) {
    const R = rng(p * 7919 + 1), throws = [];
    for (let i = 0; i < 600; i++) throws.push([1 + Math.floor(R() * 6), 1 + Math.floor(R() * 6)]);
    const a = play(throws, true, gN, gO, o.limit), b = play(throws, false, gN, gO, o.limit);
    parentPort.postMessage({ res: [a, b] });
  }
  parentPort.postMessage({ done: true });
} else {
  const PAIRS = +process.argv[2] || 200, WORKERS = +process.argv[3] || 7;
  const o = {
    newFile: path.resolve(process.env.NEW || 'weights_mixed.json'),
    oldDir: process.env.OLD_DIR || '../Bagammon',
    limit: process.env.LIMIT !== undefined ? +process.env.LIMIT : 200,
  };
  o.oldFile = path.resolve(o.oldDir, process.env.OLD || 'weights_mixed.json');
  let games = 0, wins = 0, unfinished = 0, limitHit = 0, pairWins = 0, pairLoss = 0, finished = 0;
  const t0 = Date.now();
  const report = () => {
    const n = games - unfinished, p = wins / Math.max(n, 1), se = Math.sqrt(p * (1 - p) / Math.max(n, 1));
    const elo = (x) => 400 * Math.log10(Math.min(Math.max(x, 1e-6), 1 - 1e-6) / (1 - Math.min(Math.max(x, 1e-6), 1 - 1e-6)));
    console.log(`new vs old: ${wins}/${n} = ${(100 * p).toFixed(1)}% ±${(196 * se).toFixed(1)}  Elo ${elo(p).toFixed(0)} (${elo(p - 1.96 * se).toFixed(0)}..${elo(p + 1.96 * se).toFixed(0)})` +
      `  unfinished ${unfinished}, turn limit ${limitHit}, pairs won ${pairWins} lost ${pairLoss}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  };
  for (let w = 0; w < WORKERS; w++) {
    const from = Math.floor(PAIRS * w / WORKERS), to = Math.floor(PAIRS * (w + 1) / WORKERS);
    const wk = new Worker(__filename, { workerData: Object.assign({ from, to }, o) });
    wk.on('message', (m) => {
      if (m.done) { if (++finished === WORKERS) report(); return; }
      let pw = 0;
      for (const g of m.res) { games++; if (g.unfinished) unfinished++; else if (g.newWon) { wins++; pw++; } if (g.limitHit) limitHit++; }
      if (pw === 2) pairWins++; else if (pw === 0) pairLoss++;
      if (games % 100 === 0) report();
    });
    wk.on('error', (e) => { console.error(e); process.exitCode = 1; });
  }
}
