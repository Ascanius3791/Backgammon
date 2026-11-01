// Fast core for training: the same rules as engine.js and the same net as bot.js, rebuilt for speed.
//   - flat typed-array position, copy-on-descend frames instead of cloned state objects
//   - Zobrist hashes instead of string keys for de-duplicating move sequences
//   - incremental hidden layer: the 240 stone inputs live in an accumulator per perspective that a move
//     updates by a few rows of W1; only the 6 scalar inputs (and Mixed walk inputs) are added per leaf
// check_fast.js verifies that end states, values and game flow match engine.js/bot.js exactly.
'use strict';
const BG = require('./engine.js');
const L = require('./bot.js');

const H = L.H; // fixed: nets of other sizes are trained by ./bgtrain only
const ROLL = 0, MOVE = 1, WALK = 2, OVER = 3;
const BAR = 25, OFF = 0; // relative 'from' of a bar entry, relative 'to' of a bear-off
const STONES = BG.STONES, MAX_STACK = BG.MAX_STACK;
const MAXD = 4; // dice per segment at most (Rene's doubles)
// Math.tanh (and Math.exp) are slow out-of-line calls in older V8 (Math.tanh was ~40% of the search on
// node 12). Cubic Hermite interpolation in a table instead: max error 4e-9, about 3x faster.
const TSC = 64, TLIM = 10, TN = 2 * TLIM * TSC + 2, TT = new Float64Array(TN), TDV = new Float64Array(TN);
for (let i = 0; i < TN; i++) { TT[i] = Math.tanh(i / TSC - TLIM); TDV[i] = (1 - TT[i] * TT[i]) / TSC; }
function tanh(x) {
  const u = (x + TLIM) * TSC;
  if (u <= 0) return -1;
  if (u >= TN - 1) return 1;
  const i = u | 0, t = u - i, t2 = t * t, t3 = t2 * t;
  return TT[i] * (2 * t3 - 3 * t2 + 1) + TDV[i] * (t3 - 2 * t2 + t) + TT[i + 1] * (3 * t2 - 2 * t3) + TDV[i + 1] * (t3 - t2);
}

// move encoding: from (rel 1..24, 25 = bar) | to (rel 1..24, 0 = off) << 5 | die << 10 | back << 13
const enc = (f, t, d, b) => f | (t << 5) | (d << 10) | (b << 13);
const mFrom = (m) => m & 31, mTo = (m) => (m >> 5) & 31, mDie = (m) => (m >> 10) & 7, mBack = (m) => (m >> 13) & 1;

// Zobrist tables: two independent 32-bit halves, combined into one 53-bit number key
const rnd32 = () => (Math.random() * 4294967296) >>> 0;
const ZP1 = new Int32Array(25 * 11), ZP2 = new Int32Array(25 * 11); // [point][count + 5]
const ZS1 = new Int32Array(4 * 16), ZS2 = new Int32Array(4 * 16);   // [bar0, bar1, off0, off1][n]
for (let i = 0; i < ZP1.length; i++) { ZP1[i] = rnd32(); ZP2[i] = rnd32(); }
for (let i = 0; i < ZS1.length; i++) { ZS1[i] = rnd32(); ZS2[i] = rnd32(); }

class Frame {
  constructor() {
    this.b = new Int8Array(25);       // absolute board, + White, - Black (like engine.js)
    this.bar = new Int32Array(2);
    this.off = new Int32Array(2);
    this.pip = new Int32Array(2);
    this.acc = new Float64Array(2 * H); // b1 + stone inputs, from White's (0..H-1) and Black's view
    this.h1 = 0; this.h2 = 0;
    this.cnt = new Int8Array(7);      // remaining dice by face
  }
  copyFrom(o) {
    this.b.set(o.b); this.bar.set(o.bar); this.off.set(o.off); this.pip.set(o.pip); this.acc.set(o.acc);
    this.h1 = o.h1; this.h2 = o.h2; this.cnt.set(o.cnt);
  }
}

// Weights as typed arrays; w can be a bot.js Net or parsed JSON. shared: in a SharedArrayBuffer (for workers).
function makeWeights(w, shared) {
  if (w.all) return w;
  if (w.heads === 2 || (w.W2b && !w.heads)) throw new Error('fast.js supports single-output nets only');
  if (w.b1.length !== H) throw new Error('fast.js supports ' + H + ' hidden units only (use ./bgtrain for bigger nets)');
  const n = w.W1.length / H, size = weightsSize(n);
  const buf = new Float64Array(shared ? new SharedArrayBuffer(size * 8) : new ArrayBuffer(size * 8));
  const o = sharedViews(buf, n);
  o.W1.set(w.W1); o.b1.set(w.b1); o.W2.set(w.W2); o.b2[0] = typeof w.b2 === 'number' ? w.b2 : w.b2[0];
  return o;
}
// W1 | b1 | W2 | b2 laid out in one Float64Array (so one SharedArrayBuffer can hold a whole net)
function sharedViews(buf, nin) {
  const B = buf.buffer, o = buf.byteOffset;
  return {
    W1: new Float64Array(B, o, nin * H),
    b1: new Float64Array(B, o + nin * H * 8, H),
    W2: new Float64Array(B, o + (nin * H + H) * 8, H),
    b2: new Float64Array(B, o + (nin * H + 2 * H) * 8, 1),
    nin, all: buf,
  };
}
const weightsSize = (nin) => nin * H + 2 * H + 1;
const toJSON = (w) => {
  const r = (a) => Array.from(a, (x) => Math.round(x * 1e5) / 1e5);
  return { W1: r(w.W1), b1: r(w.b1), W2: r(w.W2), b2: w.b2[0] };
};

class Game {
  constructor(rule, net, opts) {
    const R = BG.RULES[rule];
    this.rule = rule;
    this.mixed = R.mixed; this.dm = R.doubleMoves; this.rollAgain = R.rollAgain; this.nonExact = R.nonExact;
    this.net = net; // from makeWeights
    this.plyLimit = (opts && opts.plyLimit) || 0;
    this.F = new Frame();
    this.frames = Array.from({ length: MAXD + 2 }, () => new Frame());
    this.mbuf = Array.from({ length: MAXD + 1 }, () => new Int32Array(256));
    this.path = new Int32Array(MAXD); this.bestPath = new Int32Array(MAXD); this.bestLen = 0; this.best = -1;
    this.seen = new Set(); this.leaves = new Set(); this.onLeaf = null;
    this.hbuf = new Float64Array(H);
    this.idx = new Int32Array(64); this.val = new Float64Array(64); this.nf = 0; this.bias = 0;
    this.reset();
  }

  reset() {
    const F = this.F;
    F.b.fill(0); F.bar.fill(0); F.off.fill(0);
    for (const [r, n] of [[24, 2], [13, 5], [8, 3], [6, 5]]) { F.b[r] += n; F.b[25 - r] -= n; }
    this.turn = 0; this.phase = ROLL; this.t0 = 0; this.t1 = 0; this.pending = false; this.rolledDouble = false;
    this.winner = -1; this.turnCount = 0; this.limitHit = false; this.moves = 0;
    F.cnt.fill(0);
    this.refresh(F);
  }

  // load an engine.js state (for checks and for the UI)
  fromEngine(s) {
    const F = this.F;
    for (let a = 1; a <= 24; a++) F.b[a] = s.board[a];
    F.bar[0] = s.bar[0]; F.bar[1] = s.bar[1]; F.off[0] = s.off[0]; F.off[1] = s.off[1];
    this.turn = s.turn; this.phase = { roll: ROLL, move: MOVE, walk: WALK, over: OVER }[s.phase];
    this.t0 = s.throw ? s.throw[0] : 0; this.t1 = s.throw ? s.throw[1] : 0;
    this.pending = !!s.pending; this.rolledDouble = s.rolledDouble;
    this.winner = s.winner === null ? -1 : s.winner; this.turnCount = s.turnCount || 0; this.limitHit = s.limitHit;
    F.cnt.fill(0);
    for (const d of s.dice) F.cnt[d]++;
    this.refresh(F);
  }

  // recompute pips, hash and accumulators of a frame from scratch
  refresh(F) {
    const { W1, b1 } = this.net, acc = F.acc, b = F.b;
    for (let j = 0; j < H; j++) { acc[j] = b1[j]; acc[H + j] = b1[j]; }
    F.pip[0] = 25 * F.bar[0]; F.pip[1] = 25 * F.bar[1];
    let h1 = 0, h2 = 0;
    for (let a = 1; a <= 24; a++) {
      const v = b[a];
      h1 ^= ZP1[a * 11 + v + 5]; h2 ^= ZP2[a * 11 + v + 5];
      if (v > 0) { F.pip[0] += a * v; for (let k = 0; k < v; k++) this.stone(acc, 0, a, k, 1); }
      if (v < 0) { F.pip[1] += (25 - a) * -v; for (let k = 0; k < -v; k++) this.stone(acc, 1, a, k, 1); }
    }
    for (let i = 0; i < 4; i++) {
      const n = i < 2 ? F.bar[i] : F.off[i - 2];
      h1 ^= ZS1[i * 16 + n]; h2 ^= ZS2[i * 16 + n];
    }
    F.h1 = h1; F.h2 = h2;
  }

  // thermometer unit k of player x's stack on absolute point a switched on (sign 1) or off (-1), both views.
  // As in bot.js, a stack sits at its owner's relative point in either view; only own/opponent block differs.
  stone(acc, x, a, k, sign) {
    const W1 = this.net.W1, r = (x === 0 ? a : 25 - a) - 1;
    const o0 = ((x === 0 ? 0 : 120) + r * 5 + k) * H;              // White's view
    const o1 = ((x === 1 ? 0 : 120) + r * 5 + k) * H;              // Black's view
    if (sign > 0) for (let j = 0; j < H; j++) { acc[j] += W1[o0 + j]; acc[H + j] += W1[o1 + j]; }
    else for (let j = 0; j < H; j++) { acc[j] -= W1[o0 + j]; acc[H + j] -= W1[o1 + j]; }
  }

  // set absolute point a to value nv, keeping the hash up to date
  setPt(F, a, nv) {
    const v = F.b[a];
    F.h1 ^= ZP1[a * 11 + v + 5] ^ ZP1[a * 11 + nv + 5];
    F.h2 ^= ZP2[a * 11 + v + 5] ^ ZP2[a * 11 + nv + 5];
    F.b[a] = nv;
  }
  setSide(F, i, arr, j, nv) { // i: 0/1 bar, 2/3 off
    const v = arr[j];
    F.h1 ^= ZS1[i * 16 + v] ^ ZS1[i * 16 + nv];
    F.h2 ^= ZS2[i * 16 + v] ^ ZS2[i * 16 + nv];
    arr[j] = nv;
  }

  // apply move m of player p to frame F (no dice bookkeeping, no turn logic)
  make(F, p, m) {
    const f = mFrom(m), t = mTo(m), sg = p === 0 ? 1 : -1, q = 1 - p, acc = F.acc;
    if (f === BAR) {
      this.setSide(F, p, F.bar, p, F.bar[p] - 1);
      F.pip[p] -= 25;
    } else {
      const a = p === 0 ? f : 25 - f, c = F.b[a] * sg;
      this.setPt(F, a, F.b[a] - sg);
      this.stone(acc, p, a, c - 1, -1);
      F.pip[p] -= f;
    }
    if (t === OFF) {
      this.setSide(F, 2 + p, F.off, p, F.off[p] + 1);
    } else {
      const a = p === 0 ? t : 25 - t;
      if (F.b[a] * sg === -1) {
        this.setPt(F, a, 0);
        this.stone(acc, q, a, 0, -1);
        this.setSide(F, q, F.bar, q, F.bar[q] + 1);
        F.pip[q] += 25 - (q === 0 ? a : 25 - a);
      }
      const c = F.b[a] * sg;
      this.setPt(F, a, F.b[a] + sg);
      this.stone(acc, p, a, c, 1);
      F.pip[p] += t;
    }
  }

  // ---------------------------------------------------------------- rules (mirrors engine.js)
  own(F, p, r) { const v = F.b[p === 0 ? r : 25 - r]; return p === 0 ? (v > 0 ? v : 0) : (v < 0 ? -v : 0); }
  canLand(F, p, r) { const v = F.b[p === 0 ? r : 25 - r] * (p === 0 ? 1 : -1); return v > -2 && v < MAX_STACK; }
  highest(F, p) { for (let r = 24; r >= 1; r--) if (this.own(F, p, r)) return r; return 0; }
  entryBlocked(F, p) { for (let d = 1; d <= 6; d++) if (this.canLand(F, p, 25 - d)) return false; return true; }

  // legal single-die moves of p into out[n..]; returns the new n
  gen(F, p, d, out, n) {
    if (F.bar[p] > 0) {
      if (this.canLand(F, p, 25 - d)) out[n++] = enc(BAR, 25 - d, d, 0);
      return n;
    }
    const top = this.highest(F, p), home = top <= 6;
    if (this.mixed && home && this.own(F, p, d)) { out[n++] = enc(d, OFF, d, 0); return n; }
    const canBack = this.mixed && F.off[0] === 0 && F.off[1] === 0 && F.bar[1 - p] === 0;
    for (let r = 1; r <= top; r++) {
      if (!this.own(F, p, r)) continue;
      const t = r - d;
      if (t >= 1) { if (this.canLand(F, p, t)) out[n++] = enc(r, t, d, 0); }
      else if (home && (t === 0 || (this.nonExact && r === top))) out[n++] = enc(r, OFF, d, 0);
      if (canBack && r > 6 && r + d <= 24 && this.canLand(F, p, r + d)) out[n++] = enc(r, r + d, d, 1);
    }
    return n;
  }
  genAll(F, p, out) {
    let n = 0;
    for (let d = 1; d <= 6; d++) if (F.cnt[d]) n = this.gen(F, p, d, out, n);
    return n;
  }
  anyMove(F, p) {
    const out = this.mbuf[MAXD];
    for (let d = 1; d <= 6; d++) if (F.cnt[d] && this.gen(F, p, d, out, 0)) return true;
    return false;
  }
  setDice(F, a, b) {
    F.cnt.fill(0);
    if (a === b) F.cnt[a] = this.dm; else { F.cnt[a]++; F.cnt[b]++; }
  }

  // ---------------------------------------------------------------- game flow on this.F
  roll(a, b) {
    if (a === undefined) { a = 1 + Math.floor(Math.random() * 6); b = 1 + Math.floor(Math.random() * 6); }
    this.t0 = a; this.t1 = b; this.rolledDouble = a === b;
    if (this.mixed) this.pending = true;
    this.setDice(this.F, a, b);
    this.phase = MOVE;
    this.settle();
  }
  applyMove(m) {
    this.moves++;
    this.F.cnt[mDie(m)]--;
    this.make(this.F, this.turn, m);
    this.settle();
  }
  settle() {
    const p = this.turn, F = this.F;
    if (F.off[p] === STONES) { this.winner = p; this.phase = OVER; F.cnt.fill(0); return; }
    if (!this.anyMove(F, p)) { F.cnt.fill(0); this.finishSegment(); }
  }
  finishSegment() {
    if (this.phase === WALK) { this.phase = ROLL; return; }
    if (this.rollAgain && this.rolledDouble) { this.phase = ROLL; return; }
    this.nextTurn();
  }
  nextTurn() {
    const q = 1 - this.turn, F = this.F;
    if (this.plyLimit && ++this.turnCount >= this.plyLimit) {
      const a = F.pip[0], b = F.pip[1];
      this.winner = a < b ? 0 : a > b ? 1 : this.turn;
      this.phase = OVER; F.cnt.fill(0); this.limitHit = true;
      return;
    }
    this.turn = q; this.rolledDouble = false; F.cnt.fill(0); this.phase = ROLL;
    if (!this.mixed) return;
    if (F.bar[q] > 0 && this.entryBlocked(F, q)) { this.turn = 1 - q; this.pending = false; return; }
    if (this.pending) {
      this.setDice(F, this.t0, this.t1);
      this.pending = false; this.phase = WALK;
      this.settle();
    }
  }

  // ---------------------------------------------------------------- evaluation
  // net output for frame F seen by side P; walk = Mixed 'must walk' inputs on (dice from F.cnt)
  evalFrame(F, P, walk) {
    const { W1, W2, b2 } = this.net, Q = 1 - P, acc = F.acc, o = P * H;
    const e0 = F.bar[P] / 5, e1 = F.bar[Q] / 5, e2 = F.off[P] / 15, e3 = F.off[Q] / 15, e4 = F.pip[P] / 200, e5 = F.pip[Q] / 200;
    const B = 240 * H;
    let z = b2[0] + (this.mixed ? (F.pip[Q] - F.pip[P]) / 60 : 0);
    if (!walk) {
      for (let j = 0; j < H; j++) {
        const x = acc[o + j] + e0 * W1[B + j] + e1 * W1[B + H + j] + e2 * W1[B + 2 * H + j] + e3 * W1[B + 3 * H + j]
          + e4 * W1[B + 4 * H + j] + e5 * W1[B + 5 * H + j];
        z += tanh(x) * W2[j];
      }
    } else {
      const hb = this.hbuf, c = F.cnt;
      for (let j = 0; j < H; j++) {
        hb[j] = acc[o + j] + e0 * W1[B + j] + e1 * W1[B + H + j] + e2 * W1[B + 2 * H + j] + e3 * W1[B + 3 * H + j]
          + e4 * W1[B + 4 * H + j] + e5 * W1[B + 5 * H + j] + W1[246 * H + j];
      }
      for (let d = 1; d <= 6; d++) if (c[d]) { const v = 0.5 * c[d], r = (246 + d) * H; for (let j = 0; j < H; j++) hb[j] += v * W1[r + j]; }
      for (let j = 0; j < H; j++) z += tanh(hb[j]) * W2[j];
    }
    return 1 / (1 + Math.exp(-z));
  }

  // value for mover p of frame F, a finished segment (phase MOVE or WALK) of the current turn:
  // runs the segment-end transition of engine.js virtually, then scores the resulting position
  leafValue(F, p) {
    if (F.off[p] === STONES) return 1;
    if (this.phase === WALK || (this.rollAgain && this.rolledDouble)) return this.evalFrame(F, p, false);
    if (this.plyLimit && this.turnCount + 1 >= this.plyLimit) {
      const a = F.pip[0], b = F.pip[1];
      return (a < b ? 0 : a > b ? 1 : p) === p ? 1 : 0;
    }
    const q = 1 - p;
    if (!this.mixed) return 1 - this.evalFrame(F, q, false);
    if (F.bar[q] > 0 && this.entryBlocked(F, q)) return this.evalFrame(F, p, false);
    // q walks the same throw (pending is always set during a Mixed move segment)
    const save0 = F.cnt;
    const c = this.frames[MAXD + 1].cnt; // scratch dice
    F.cnt = c; this.setDice(F, this.t0, this.t1);
    const walk = this.anyMove(F, q);
    const v = this.evalFrame(F, q, walk);
    F.cnt = save0;
    return 1 - v;
  }

  // best end state of the current MOVE/WALK segment for the side to play: fills bestPath/bestLen/best
  search() {
    const p = this.turn, f0 = this.frames[0];
    // the incrementally kept accumulators of this.F are stale once training has changed the weights
    this.refresh(this.F);
    f0.copyFrom(this.F);
    this.seen.clear(); this.leaves.clear();
    this.best = -1; this.bestLen = 0;
    this.dfs(0, p);
    return this.best;
  }
  dfs(d, p) {
    const F = this.frames[d], out = this.mbuf[d];
    const n = this.genAll(F, p, out);
    if (n === 0) return false;
    const C = this.frames[d + 1];
    for (let i = 0; i < n; i++) {
      const m = out[i];
      C.copyFrom(F);
      C.cnt[mDie(m)]--;
      this.make(C, p, m);
      this.path[d] = m;
      let leaf = C.off[p] === STONES;
      if (!leaf) {
        let left = 0;
        for (let k = 1; k <= 6; k++) left += C.cnt[k];
        if (left === 0) leaf = true;
        else {
          let code = 0;
          for (let k = 1; k <= 6; k++) code = code * 5 + C.cnt[k];
          const key = (C.h1 >>> 0) * 2097152 + (((C.h2 ^ Math.imul(code, -1640531535)) >>> 0) & 2097151);
          if (this.seen.has(key)) continue;
          this.seen.add(key);
          if (this.dfs(d + 1, p)) continue;
          leaf = true; // no remaining die can be played
        }
      }
      const key = (C.h1 >>> 0) * 2097152 + ((C.h2 >>> 0) & 2097151);
      if (this.leaves.has(key)) continue;
      this.leaves.add(key);
      const v = this.leafValue(C, p);
      if (this.onLeaf) this.onLeaf(C, v);
      if (v > this.best) { this.best = v; this.bestLen = d + 1; for (let k = 0; k <= d; k++) this.bestPath[k] = this.path[k]; }
    }
    return true;
  }
  playBest() {
    this.search();
    const n = this.bestLen, path = this.bestPath.slice(0, n);
    for (let k = 0; k < n && (this.phase === MOVE || this.phase === WALK); k++) this.applyMove(path[k]);
  }

  // Greedy baseline of bot.js (hit bonus 0): plays single moves until the turn passes or the segment ends
  playGreedy() {
    const p = this.turn, out = this.mbuf[0];
    while (this.phase === MOVE || this.phase === WALK) {
      const n = this.genAll(this.F, p, out);
      let best = 0, bv = -1e9;
      for (let i = 0; i < n; i++) {
        const m = out[i];
        const v = (mBack(m) ? -mDie(m) : mDie(m)) + (mTo(m) === OFF ? 6 : 0) + Math.random() * 0.1;
        if (v > bv) { bv = v; best = m; }
      }
      this.applyMove(best);
      if (this.turn !== p) break;
    }
  }

  // random single moves until the turn passes or the segment ends (bot.js randomBot)
  playRandom() {
    const p = this.turn, out = this.mbuf[0];
    while (this.phase === MOVE || this.phase === WALK) {
      this.applyMove(out[Math.floor(Math.random() * this.genAll(this.F, p, out))]);
      if (this.turn !== p) break;
    }
  }

  // full sparse inputs of the current position seen by the side to roll (like bot.js features)
  features() {
    const P = this.turn, Q = 1 - P, F = this.F, idx = this.idx, val = this.val;
    let n = 0;
    for (let r = 1; r <= 24; r++) {
      const a = this.own(F, P, r), b = this.own(F, Q, r);
      for (let k = 0; k < a; k++) { idx[n] = (r - 1) * 5 + k; val[n++] = 1; }
      for (let k = 0; k < b; k++) { idx[n] = 120 + (r - 1) * 5 + k; val[n++] = 1; }
    }
    const extra = [F.bar[P] / 5, F.bar[Q] / 5, F.off[P] / 15, F.off[Q] / 15, F.pip[P] / 200, F.pip[Q] / 200];
    for (let k = 0; k < 6; k++) { idx[n] = 240 + k; val[n++] = extra[k]; }
    if (this.mixed && this.phase === WALK) {
      idx[n] = 246; val[n++] = 1;
      for (let d = 1; d <= 6; d++) for (let c = 0; c < F.cnt[d]; c++) { idx[n] = 246 + d; val[n++] = 0.5; }
    }
    this.nf = n;
    this.bias = this.mixed ? (F.pip[Q] - F.pip[P]) / 60 : 0;
  }
}

// One-net trainer state: TD(lambda) with a scaled eligibility trace (E_true = c * E), so the per-step
// decay is O(1). The update only visits the W1 rows whose input has been active in this game (all
// other rows have a zero trace), plus b1, W2 and b2.
class TD {
  constructor(net, lambda) {
    this.net = net; this.lambda = lambda;
    this.E = new Float64Array(net.all.length);
    this.touched = new Uint8Array(net.nin);
    this.rows = new Int32Array(net.nin); this.nrows = 0;
    this.c = 1;
    this.h = new Float64Array(H);
    this.dh = new Float64Array(H);
    this.y = 0.5;
  }
  clear() {
    const E = this.E, rows = this.rows;
    for (let r = 0; r < this.nrows; r++) { this.touched[rows[r]] = 0; E.fill(0, rows[r] * H, rows[r] * H + H); }
    E.fill(0, this.net.nin * H);
    this.nrows = 0; this.c = 1;
  }
  // P(side to roll wins) for g's current position; leaves features in g and hidden activations in this.h
  forward(g) {
    g.features();
    const { W1, b1, W2, b2 } = this.net, h = this.h, idx = g.idx, val = g.val, n = g.nf;
    for (let j = 0; j < H; j++) h[j] = b1[j];
    for (let t = 0; t < n; t++) { const o = idx[t] * H, v = val[t]; for (let j = 0; j < H; j++) h[j] += v * W1[o + j]; }
    let z = b2[0] + g.bias;
    for (let j = 0; j < H; j++) { h[j] = tanh(h[j]); z += h[j] * W2[j]; }
    return (this.y = 1 / (1 + Math.exp(-z)));
  }
  // E = lambda * E + sign * grad(output), right after forward(g)
  trace(g, sign) {
    const lam = this.lambda;
    if (lam === 0) this.clear(); else this.c *= lam;
    if (this.c < 1e-150) { const E = this.E, c = this.c; for (let i = 0; i < E.length; i++) E[i] *= c; this.c = 1; }
    for (let t = 0; t < g.nf; t++) { const r = g.idx[t]; if (!this.touched[r]) { this.touched[r] = 1; this.rows[this.nrows++] = r; } }
    const net = this.net, E = this.E, h = this.h, nin = net.nin;
    const s = sign * this.y * (1 - this.y) / this.c;
    const oB1 = nin * H, oW2 = oB1 + H, oB2 = oW2 + H;
    const idx = g.idx, val = g.val, n = g.nf, W2 = net.W2, dh = this.dh;
    E[oB2] += s;
    for (let j = 0; j < H; j++) {
      E[oW2 + j] += s * h[j];
      dh[j] = s * W2[j] * (1 - h[j] * h[j]);
      E[oB1 + j] += dh[j];
    }
    for (let t = 0; t < n; t++) { const o = idx[t] * H, v = val[t]; for (let j = 0; j < H; j++) E[o + j] += v * dh[j]; }
  }
  update(delta) {
    const w = this.net.all, E = this.E, k = delta * this.c, rows = this.rows;
    for (let r = 0; r < this.nrows; r++) { const o = rows[r] * H; for (let j = o; j < o + H; j++) w[j] += k * E[j]; }
    for (let i = this.net.nin * H; i < w.length; i++) w[i] += k * E[i];
  }
}

module.exports = { Game, TD, Frame, makeWeights, sharedViews, weightsSize, toJSON, ROLL, MOVE, WALK, OVER, mFrom, mTo, mDie, mBack, BAR, OFF };
