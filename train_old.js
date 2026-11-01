// TD(lambda) self-play training.
//   RULE=paul|rene|mixed [PRE=n] node train.js [games] [alpha] [lambda] [evalEvery]
// Resumes from weights_<rule>.json. Writes weights_<rule>.json/.js and metrics_<rule>.js (for plot.html).
// PRE: the first PRE games are played by the greedy baseline instead of the net, so the net first
// learns values from games that actually finish (needed for Mixed, where aimless play never ends).
const fs = require('fs');
const path = require('path');
const BG = require('./engine.js');
const L = require('./bot.js');

const RULE = process.env.RULE || 'paul';
const PRE = +process.env.PRE || 0;
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

const net = fs.existsSync(FILE) ? new L.Net(JSON.parse(fs.readFileSync(FILE))) : L.newNet(RULE);
const zeros = () => ({ W1: new Float64Array(net.W1.length), b1: new Float64Array(L.H), W2: new Float64Array(L.H), b2: new Float64Array(1),
  W2b: new Float64Array(L.H), b2b: new Float64Array(1) });
const E = zeros();     // eligibility trace (single head), or for White's winning chance (two heads)
const EB = zeros();    // two heads only: trace for Black's winning chance
const KEYS = net.heads === 2 ? ['W1', 'b1', 'W2', 'b2', 'W2b', 'b2b'] : ['W1', 'b1', 'W2', 'b2'];
let metrics = [];
let startGame = 0;
if (fs.existsSync(METRICS)) {
  metrics = JSON.parse(fs.readFileSync(METRICS, 'utf8').replace(/^[^=]*=/, '').replace(/;\s*$/, ''));
  if (metrics.length) startGame = metrics[metrics.length - 1].game;
}

function save() {
  const w = JSON.stringify(net.toJSON());
  fs.writeFileSync(FILE, w);
  fs.writeFileSync(path.join(__dirname, 'weights_' + RULE + '.js'), 'window.BOT_WEIGHTS_' + RULE.toUpperCase() + '=' + w + ';\n');
  fs.writeFileSync(METRICS, 'window.METRICS_' + RULE.toUpperCase() + '=' + JSON.stringify(metrics) + ';\n');
}

function update(d, dB) {
  for (const k of KEYS) {
    const w = net[k], e = E[k], eb = EB[k];
    if (dB === undefined) for (let i = 0; i < w.length; i++) w[i] += d * e[i];
    else for (let i = 0; i < w.length; i++) w[i] += d * e[i] + dB * eb[i];
  }
}

// White's and Black's winning chance for the state about to be rolled
function chances(s) {
  const f = L.features(s);
  const y = net.forward(f);
  if (net.heads === 2) return { f, w: s.turn === 0 ? net.y[0] : net.y[1], b: s.turn === 0 ? net.y[1] : net.y[0] };
  return { f, w: s.turn === 0 ? y : 1 - y, b: s.turn === 0 ? 1 - y : y };
}

// returns {plies, sqErr, steps, capped}
// greedySide: 2 = the greedy baseline plays both sides, 0/1 = it plays that side, -1 = pure self-play
function selfPlayGame(greedySide) {
  const s = BG.newGame(RULE, { plyLimit: RULE === 'mixed' ? TURN_LIMIT : 0 });
  s.log = { push() {} };
  const two = net.heads === 2;
  for (const k in E) { E[k].fill(0); EB[k].fill(0); }
  let plies = 0, sqErr = 0, steps = 0, cur = null;
  while (s.phase !== 'over' && plies++ < MAX_PLIES) {
    // gradient of the current state's winning chances goes into the eligibility traces
    cur = chances(s);
    for (const k of KEYS) {
      const a = E[k], ab = EB[k];
      for (let i = 0; i < a.length; i++) { a[i] *= LAMBDA; if (two) ab[i] *= LAMBDA; }
    }
    if (two) {
      net.addGradHead(cur.f, s.turn === 0 ? 0 : 1, 1, E);   // White's chance
      net.addGradHead(cur.f, s.turn === 0 ? 1 : 0, 1, EB);  // Black's chance
    } else {
      net.addGrad(cur.f, net.y[0], s.turn === 0 ? 1 : -1, E);
    }

    if (s.phase === 'roll') BG.roll(s);
    if (greedySide === 2 || s.turn === greedySide) {
      const p = s.turn;
      while (s.phase === 'move' || s.phase === 'walk') { BG.applyMove(s, L.greedyBot.move(s)); if (s.turn !== p) break; }
    } else {
      for (const m of L.bestPlan(net, s).path) BG.applyMove(s, m);
    }

    let nw, nb;
    if (s.phase === 'over') { nw = s.winner === 0 ? 1 : 0; nb = 1 - nw; }
    else { const n = chances(s); nw = n.w; nb = n.b; }
    sqErr += (nw - cur.w) * (nw - cur.w); steps++;
    if (two) update(ALPHA * (nw - cur.w), ALPHA * (nb - cur.b));
    else update(ALPHA * (nw - cur.w));
  }
  const capped = s.phase !== 'over';
  if (capped) { // never finished: a loss for both (two heads), a draw otherwise
    if (two) update(ALPHA * (0 - cur.w), ALPHA * (0 - cur.b));
    else update(ALPHA * (0.5 - cur.w));
  }
  return { plies, sqErr, steps, capped: capped || s.limitHit };
}

// Mixed games never end between aimless players, so there the yardstick is the greedy baseline.
function evalMatch(n) {
  const bot = L.netBot(net);
  const opp = RULE === 'mixed' ? L.greedyBot : L.randomBot;
  const cap = RULE === 'mixed' ? 800 : 5000;
  let wins = 0, played = 0;
  for (let g = 0; g < n; g++) {
    const botWhite = g % 2 === 0;
    const w = L.playGame(botWhite ? [bot, opp] : [opp, bot], RULE, cap);
    if (w >= 0) { played++; if ((w === 0) === botWhite) wins++; }
  }
  return { wins, played, n };
}

const t0 = Date.now();
let acc = { sq: 0, steps: 0, plies: 0, games: 0, capped: 0 };
for (let i = 1; i <= GAMES; i++) {
  const g = startGame + i;
  // After the warm-up, half the games are against the greedy baseline: aimless net-vs-net games
  // never end, and a net that sees no finished games learns that nothing matters.
  const mix = RULE === 'mixed' && g > PRE && g % 2 === 0;
  const r = selfPlayGame(g <= PRE ? 2 : mix ? (g % 4 === 0 ? 0 : 1) : -1);
  acc.sq += r.sqErr; acc.steps += r.steps; acc.plies += r.plies; acc.games++; acc.capped += r.capped ? 1 : 0;
  if (i % LOG_EVERY === 0 || i === GAMES) {
    const row = { game: g, tdErr: acc.sq / Math.max(acc.steps, 1), plies: acc.plies / acc.games, capped: acc.capped / acc.games };
    if (i % EVAL_EVERY === 0 || i === GAMES) {
      const m = evalMatch(RULE === "mixed" ? 20 : 100);
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
