// TD(lambda) self-play training on the fast core (fast.js) with parallel self-play threads.
//   RULE=paul|rene|mixed [PRE=n] [WORKERS=n] node train.js [games] [alpha] [lambda] [evalEvery]
// Resumes from weights_<rule>.json. Writes weights_<rule>.json/.js and metrics_<rule>.js (for plot.html).
// PRE: the first PRE games are played by the greedy baseline instead of the net, so the net first
// learns values from games that actually finish (needed for Mixed, where aimless play never ends).
// WORKERS (default 4): self-play threads. They share one set of weights and update it without locks
// (Hogwild-style), each with its own eligibility trace; WORKERS=1 is the plain sequential algorithm.
// (Training on private copies merged after each game diverged with 4+ workers at alpha 0.05: the
// merged per-game steps are computed on stale weights and overshoot.)
// The previous single-threaded trainer on engine.js/bot.js is kept as train_old.js.
// cpp/ has the same trainer in C++ (./bgtrain, same arguments and files), about 5x faster than this one.
const fs = require('fs');
const path = require('path');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const L = require('./bot.js');
const FA = require('./fast.js');

// returns {plies, sqErr, steps, capped}
// greedySide: 2 = the greedy baseline plays both sides, 0/1 = it plays that side, -1 = pure self-play
function selfPlayGame(g, td, greedySide, o) {
  g.reset();
  td.clear();
  let plies = 0, sqErr = 0, steps = 0, cur = 0;
  while (g.phase !== FA.OVER && plies++ < o.maxPlies) {
    // gradient of White's winning chance in the current state goes into the eligibility trace
    const y = td.forward(g);
    cur = g.turn === 0 ? y : 1 - y;
    td.trace(g, g.turn === 0 ? 1 : -1);

    // one step: the roll, the mover's segment and, in Mixed, the opponent's forced walk of that throw.
    // The net only ever sees positions with nobody holding dice (no walk inputs; the walk is searched).
    if (g.phase === FA.ROLL) g.roll();
    while (g.phase === FA.MOVE || g.phase === FA.WALK) {
      if (greedySide === 2 || g.turn === greedySide) g.playGreedy(); else g.playBest();
    }

    let nw;
    if (g.phase === FA.OVER) nw = g.winner < 0 ? 0.5 : g.winner === 0 ? 1 : 0; // -1: drawn at the turn limit
    else { const y2 = td.forward(g); nw = g.turn === 0 ? y2 : 1 - y2; }
    sqErr += (nw - cur) * (nw - cur); steps++;
    td.update(o.alpha * (nw - cur));
  }
  const capped = g.phase !== FA.OVER;
  if (capped) td.update(o.alpha * (0.5 - cur)); // never finished: a draw
  return { plies, sqErr, steps, capped: capped || g.limitHit };
}

// ---------------------------------------------------------------- worker side
if (!isMainThread) {
  const o = workerData;
  const net = FA.sharedViews(new Float64Array(o.weights), o.nin);
  const counter = new Int32Array(o.counter);
  const g = new FA.Game(o.rule, net, { plyLimit: o.rule === 'mixed' ? o.turnLimit : 0 });
  const td = new FA.TD(net, o.lambda);
  for (;;) {
    const gi = Atomics.add(counter, 0, 1) + 1; // global game number decides who plays
    if (gi > o.last) break;
    // After the warm-up, half the Mixed games are against the greedy baseline: aimless net-vs-net games
    // never end, and a net that sees no finished games learns that nothing matters.
    const mix = o.rule === 'mixed' && gi > o.pre && gi % 2 === 0;
    parentPort.postMessage(selfPlayGame(g, td, gi <= o.pre ? 2 : mix ? (gi % 4 === 0 ? 0 : 1) : -1, o));
  }
  parentPort.postMessage({ done: true });
  return;
}

// ---------------------------------------------------------------- main side
const RULE = process.env.RULE || 'paul';
const PRE = +process.env.PRE || 0;
const WORKERS = +process.env.WORKERS || 4;
const GAMES = +process.argv[2] || 5000;
const ALPHA = +process.argv[3] || 0.05;
const LAMBDA = +process.argv[4] || 0.7;
// Mixed: games longer than this many turns are decided by pip count (LIMIT=0 turns it off)
const TURN_LIMIT = process.env.LIMIT !== undefined ? +process.env.LIMIT : 200;
const MAX_PLIES = RULE === 'mixed' ? 1500 : 3000;
const EVAL_EVERY = +process.argv[5] || 500;
const LOG_EVERY = +process.env.LOG || Math.max(1, Math.min(100, EVAL_EVERY));
const FILE = path.join(__dirname, 'weights_' + RULE + '.json');
const METRICS = path.join(__dirname, 'metrics_' + RULE + '.js');

const net = FA.makeWeights(fs.existsSync(FILE) ? JSON.parse(fs.readFileSync(FILE)) : L.newNet(RULE), true);
let metrics = [];
let startGame = 0;
if (fs.existsSync(METRICS)) {
  metrics = JSON.parse(fs.readFileSync(METRICS, 'utf8').replace(/^[^=]*=/, '').replace(/;\s*$/, ''));
  if (metrics.length) startGame = metrics[metrics.length - 1].game;
}

function save() {
  const w = JSON.stringify(FA.toJSON(net));
  fs.writeFileSync(FILE, w);
  fs.writeFileSync(path.join(__dirname, 'weights_' + RULE + '.js'), 'window.BOT_WEIGHTS_' + RULE.toUpperCase() + '=' + w + ';\n');
  fs.writeFileSync(METRICS, 'window.METRICS_' + RULE.toUpperCase() + '=' + JSON.stringify(metrics) + ';\n');
}

// Mixed games never end between aimless players, so there the yardstick is the greedy baseline.
// Plays on a snapshot of the weights; the workers keep training meanwhile.
function evalMatch(n) {
  const g = new FA.Game(RULE, FA.sharedViews(Float64Array.from(net.all), net.nin));
  const cap = RULE === 'mixed' ? 800 : 5000;
  let wins = 0, played = 0;
  for (let i = 0; i < n; i++) {
    const bot = i % 2 === 0 ? 0 : 1;
    g.reset();
    let rolls = 0;
    while (g.phase !== FA.OVER && rolls + g.moves < cap) {
      if (g.phase === FA.ROLL) { g.roll(); rolls++; }
      else if (g.turn === bot) g.playBest();
      else if (RULE === 'mixed') g.playGreedy();
      else g.playRandom();
    }
    if (g.phase === FA.OVER) { played++; if (g.winner === bot) wins++; }
  }
  return { wins, played, n };
}

const t0 = Date.now();
let acc = { sq: 0, steps: 0, plies: 0, games: 0, capped: 0 };
let done = 0, finished = 0;
const opts = {
  rule: RULE, nin: net.nin, weights: net.all.buffer, counter: new SharedArrayBuffer(4), last: startGame + GAMES,
  pre: PRE, alpha: ALPHA, lambda: LAMBDA, turnLimit: TURN_LIMIT, maxPlies: MAX_PLIES,
};
new Int32Array(opts.counter)[0] = startGame;

function onResult(r) {
  if (r.done) { if (++finished === WORKERS) save(); return; }
  const i = ++done, g = startGame + i;
  acc.sq += r.sqErr; acc.steps += r.steps; acc.plies += r.plies; acc.games++; acc.capped += r.capped ? 1 : 0;
  if (i % LOG_EVERY === 0 || i === GAMES) {
    const row = { game: g, tdErr: acc.sq / Math.max(acc.steps, 1), plies: acc.plies / acc.games, capped: acc.capped / acc.games };
    if (i % EVAL_EVERY === 0 || i === GAMES) {
      const m = evalMatch(RULE === 'mixed' ? 20 : 100);
      row.winRate = m.played ? m.wins / m.played : null;
      row.unfinished = m.n - m.played;
    }
    metrics.push(row);
    acc = { sq: 0, steps: 0, plies: 0, games: 0, capped: 0 };
    save();
    console.log(`game ${g}  ${((Date.now() - t0) / 1000).toFixed(0)}s  tdErr ${row.tdErr.toFixed(4)}  plies ${row.plies.toFixed(0)}  capped ${(row.capped * 100).toFixed(0)}%` +
      (row.winRate !== undefined ? `  vs ${RULE === 'mixed' ? 'greedy' : 'random'}: ${row.winRate} (${row.unfinished} unfinished)` : ''));
  }
}

for (let k = 0; k < WORKERS; k++) {
  const w = new Worker(__filename, { workerData: opts });
  w.on('message', onResult);
  w.on('error', (e) => { console.error(e); process.exitCode = 1; });
}
