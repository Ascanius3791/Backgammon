// Search distillation for Paul's rules (fitted value iteration):
//   repeat: self-play with the current net (epsilon-random plans for variety), label every position
//   before a roll with the depth-2 search value (average over the 21 rolls of the best move scored by
//   the net), fit the net to  (1-BETA)*search + BETA*real outcome  with cross-entropy, then gate the
//   new net against the old one in a match.
//   node distill.js [rounds]     env: GAMES (per round) EPOCHS LR BETA EPS ARENA WORKERS
const fs = require('fs');
const path = require('path');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const BG = require('./engine.js');
const L = require('./bot.js');
const A = require('./analysis.js');

const RULE = 'paul';
const dir = __dirname;
const toNet = (w) => new L.Net(w);
const raw = (net) => ({ W1: net.W1, b1: net.b1, W2: net.W2, b2: net.b2[0] });

// ---------------------------------------------------------------- worker side
function genGames(net, n, eps) {
  const out = { len: [], idx: [], val: [], T: [], z: [], game: [] };
  for (let g = 0; g < n; g++) {
    const s = BG.newGame(RULE);
    s.log = { push() {} };
    const mine = [];
    for (let plies = 0; s.phase !== 'over' && plies < 3000; plies++) {
      if (s.phase === 'roll') {
        const f = L.features(s);
        mine.push({ f, T: A.evalState(net, s, s.turn, 2), q: s.turn });
        BG.roll(s);
        continue;
      }
      const plan = Math.random() < eps
        ? (() => { const e = L.endStates(s); return e[Math.floor(Math.random() * e.length)].path; })()
        : L.bestPlan(net, s).path;
      for (const m of plan) BG.applyMove(s, m);
    }
    if (s.phase !== 'over') continue;
    for (const x of mine) {
      out.len.push(x.f.idx.length);
      for (let i = 0; i < x.f.idx.length; i++) { out.idx.push(x.f.idx[i]); out.val.push(x.f.val[i]); }
      out.T.push(x.T);
      out.z.push(s.winner === x.q ? 1 : 0);
      out.game.push(g);
    }
  }
  return out;
}

function match(a, b, n) { // wins of a over b, alternating colours, both 1-ply
  const ba = L.netBot(a), bb = b ? L.netBot(b) : L.randomBot;
  let w = 0, done = 0;
  for (let g = 0; g < n; g++) {
    const aw = g % 2 === 0;
    const r = L.playGame(aw ? [ba, bb] : [bb, ba], RULE);
    if (r >= 0) { done++; if ((r === 0) === aw) w++; }
  }
  return { w, done };
}

if (!isMainThread) {
  const d = workerData;
  if (d.task === 'gen') parentPort.postMessage(genGames(toNet(d.a), d.n, d.eps));
  else parentPort.postMessage(match(toNet(d.a), d.b === 'random' ? null : toNet(d.b), d.n));
  return;
}

// ---------------------------------------------------------------- main side
const ROUNDS = +process.argv[2] || 3;
const GAMES = +process.env.GAMES || 800;
const EPOCHS = +process.env.EPOCHS || 4;
const LR = +process.env.LR || 0.004;
const BETA = process.env.BETA !== undefined ? +process.env.BETA : 0.2;
const EPS = process.env.EPS !== undefined ? +process.env.EPS : 0.1;
const ARENA = +process.env.ARENA || 400;
const WORKERS = +process.env.WORKERS || 8;

const runWorkers = (task, extra, total) => Promise.all(Array.from({ length: WORKERS }, (_, i) => new Promise((res, rej) => {
  const n = Math.floor(total / WORKERS) + (i < total % WORKERS ? 1 : 0);
  const w = new Worker(__filename, { workerData: Object.assign({ task, n }, extra) });
  w.once('message', res); w.once('error', rej);
})));

function sgd(net, D, ids, lr) { // cross-entropy on the logit: dL/dz = target - y
  const H = L.H, h = net.h;
  let sq = 0;
  for (const i of ids) {
    const f = { idx: D.idx.subarray(D.off[i], D.off[i + 1]), val: D.val.subarray(D.off[i], D.off[i + 1]), bias: 0 };
    const y = net.forward(f);
    const tgt = (1 - BETA) * D.T[i] + BETA * D.z[i];
    const dz = lr * (tgt - y);
    sq += (tgt - y) * (tgt - y);
    net.b2[0] += dz;
    for (let j = 0; j < H; j++) {
      const dh = dz * net.W2[j] * (1 - h[j] * h[j]);
      net.W2[j] += dz * h[j];
      net.b1[j] += dh;
      for (let t = 0; t < f.idx.length; t++) net.W1[f.idx[t] * H + j] += f.val[t] * dh;
    }
  }
  return sq / ids.length;
}

function pack(parts) {
  const n = parts.reduce((a, p) => a + p.T.length, 0), nf = parts.reduce((a, p) => a + p.idx.length, 0);
  const D = { n, off: new Int32Array(n + 1), idx: new Int32Array(nf), val: new Float64Array(nf), T: new Float64Array(n), z: new Float64Array(n), game: new Int32Array(n) };
  let i = 0, o = 0, gbase = 0;
  for (const p of parts) {
    let po = 0;
    for (let k = 0; k < p.T.length; k++, i++) {
      D.off[i] = o;
      for (let t = 0; t < p.len[k]; t++, o++, po++) { D.idx[o] = p.idx[po]; D.val[o] = p.val[po]; }
      D.T[i] = p.T[k]; D.z[i] = p.z[k]; D.game[i] = gbase + p.game[k];
    }
    gbase += 100000;
  }
  D.off[n] = o;
  return D;
}

function diag(net, D, ids) { // on held-out positions
  let vT = 0, vz = 0, mean = 0, m2 = 0;
  for (const i of ids) {
    const y = net.forward({ idx: D.idx.subarray(D.off[i], D.off[i + 1]), val: D.val.subarray(D.off[i], D.off[i + 1]), bias: 0 });
    vT += (y - D.T[i]) ** 2; vz += (y - D.z[i]) ** 2; mean += y; m2 += y * y;
  }
  const n = ids.length;
  return { mseSearch: vT / n, brier: vz / n, std: Math.sqrt(m2 / n - (mean / n) ** 2) };
}

const elo = (w, n) => { const p = Math.min(Math.max(w / n, 0.001), 0.999); return 400 * Math.log10(p / (1 - p)); };
const sum = (rs) => rs.reduce((a, r) => ({ w: a.w + r.w, done: a.done + r.done }), { w: 0, done: 0 });
const save = (net, file) => fs.writeFileSync(path.join(dir, file), JSON.stringify(net.toJSON()));

(async () => {
  const wfile = path.join(dir, 'weights_paul.json');
  const gen0 = path.join(dir, 'weights_paul_gen0.json');
  if (!fs.existsSync(gen0)) fs.copyFileSync(wfile, gen0);
  let net = toNet(JSON.parse(fs.readFileSync(wfile)));
  const first = toNet(JSON.parse(fs.readFileSync(gen0)));
  const t0 = Date.now(), T = () => ((Date.now() - t0) / 1000).toFixed(0) + 's';
  const baseRandom = sum(await runWorkers('match', { a: raw(net), b: 'random' }, 100));
  console.log(`start: vs random ${baseRandom.w}/${baseRandom.done}`);
  for (let r = 1; r <= ROUNDS; r++) {
    const parts = await runWorkers('gen', { a: raw(net), eps: EPS }, GAMES);
    const D = pack(parts);
    const hold = [], train = [];
    for (let i = 0; i < D.n; i++) (D.game[i] % 10 === 0 ? hold : train).push(i);
    const before = diag(net, D, hold);
    console.log(`round ${r} [${T()}] ${D.n} positions. held-out before: |V-search|^2 ${before.mseSearch.toFixed(5)}  brier ${before.brier.toFixed(4)}  std(V) ${before.std.toFixed(3)}`);
    const cand = toNet(raw(net));
    for (let e = 0; e < EPOCHS; e++) {
      for (let i = train.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); const t = train[i]; train[i] = train[j]; train[j] = t; }
      const loss = sgd(cand, D, train, LR * (1 - e / EPOCHS));
      console.log(`  epoch ${e + 1} train loss ${loss.toFixed(5)}`);
    }
    const after = diag(cand, D, hold);
    console.log(`  held-out after:  |V-search|^2 ${after.mseSearch.toFixed(5)}  brier ${after.brier.toFixed(4)}  std(V) ${after.std.toFixed(3)}`);
    const m = sum(await runWorkers('match', { a: raw(cand), b: raw(net) }, ARENA));
    console.log(`  new vs old: ${m.w}/${m.done} (${(100 * m.w / m.done).toFixed(1)}%, ${elo(m.w, m.done).toFixed(0)} Elo)  [${T()}]`);
    if (m.w * 2 > m.done) {
      net = cand;
      save(net, `weights_paul_gen${r}.json`);
      save(net, 'weights_paul.json');
      fs.writeFileSync(path.join(dir, 'weights_paul.js'), 'window.BOT_WEIGHTS_PAUL=' + JSON.stringify(net.toJSON()) + ';\n');
      console.log('  accepted');
    } else console.log('  rejected, keeping the old net');
  }
  const f = sum(await runWorkers('match', { a: raw(net), b: raw(first) }, ARENA * 2));
  const rn = sum(await runWorkers('match', { a: raw(net), b: 'random' }, 200));
  console.log(`final vs generation 0: ${f.w}/${f.done} (${elo(f.w, f.done).toFixed(0)} Elo)   vs random ${rn.w}/${rn.done}   [${T()}]`);
})();
