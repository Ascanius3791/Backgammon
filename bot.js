// Bot for Paul's, Rene's and Mixed rules: small neural net V(position, side to roll) = P(side to roll wins),
// trained by TD(lambda) self-play (see train.js). Move choice is a 1-ply afterstate search:
// every legal sequence for the rolled dice is enumerated and the resulting position is scored.
// Works in the browser (global BotLib) and in node (require).
(function (root) {
  'use strict';
  const BG = typeof require !== 'undefined' ? require('./engine.js') : root.BG;

  const H = 64; // hidden units of a new net; a saved net has its own count (b1.length)
  const NIN = 246; // 24 points * 5 thermometer units * 2 sides, then 6 scalar inputs
  const ninFor = () => NIN; // every rule sees the same inputs; Mixed searches the opponent's walk instead

  // Position from the point of view of the side to roll (s.turn).
  function features(s) {
    const p = s.turn, q = 1 - p;
    const idx = [], val = [];
    let pipOwn = 25 * s.bar[p], pipOpp = 25 * s.bar[q];
    for (let r = 1; r <= 24; r++) {
      const a = BG.ownAt(s, p, r), b = BG.ownAt(s, q, r);
      for (let k = 0; k < a; k++) { idx.push((r - 1) * 5 + k); val.push(1); }
      for (let k = 0; k < b; k++) { idx.push(120 + (r - 1) * 5 + k); val.push(1); }
      pipOwn += r * a; pipOpp += r * b;
    }
    const extra = [s.bar[p] / 5, s.bar[q] / 5, s.off[p] / 15, s.off[q] / 15, pipOwn / 200, pipOpp / 200];
    for (let k = 0; k < 6; k++) { idx.push(240 + k); val.push(extra[k]); }
    // Mixed: fixed pip-count lead as a baseline for the output. Without it, self-play on a net that
    // knows nothing collapses into aimless walking (backward moves) and games never end.
    const bias = s.rule === 'mixed' ? (pipOpp - pipOwn) / 60 : 0;
    return { idx, val, bias };
  }

  // One hidden layer of this.H units, then this.deep residual layers h += tanh(Wd h + bd) (Wd: H rows of H
  // numbers per layer, input i's weights to all units; trained by cpp/train.cpp DEEP only). heads = 1: output = P(side to roll wins) (Paul, Rene).
  // heads = 2 (Mixed): output 0 = P(side to roll wins), output 1 = P(the other side wins); a game that
  // is never finished counts as a loss for both, so stalling is never better than playing on.
  class Net {
    constructor(w, nin, heads, hidden) {
      const nh = this.H = w ? w.b1.length : hidden || H;
      this.W1 = new Float64Array(w ? w.W1.length : (nin || NIN) * nh);
      this.b1 = new Float64Array(nh);
      this.W2 = new Float64Array(nh);
      this.b2 = new Float64Array(1);
      this.heads = w ? (w.W2b ? 2 : 1) : (heads || 1);
      this.W2b = new Float64Array(nh);
      this.b2b = new Float64Array(1);
      this.h = new Float64Array(nh);
      this.y = [0.5, 0.5];
      this.deep = w && w.bd ? w.bd.length / nh : 0;
      this.Wd = new Float64Array(this.deep * nh * nh);
      this.bd = new Float64Array(this.deep * nh);
      this.z = new Float64Array(nh);
      if (this.deep) { this.Wd.set(w.Wd); this.bd.set(w.bd); }
      if (w) {
        this.W1.set(w.W1); this.b1.set(w.b1); this.W2.set(w.W2); this.b2[0] = w.b2;
        if (w.W2b) { this.W2b.set(w.W2b); this.b2b[0] = w.b2b; }
      } else {
        for (let i = 0; i < this.W1.length; i++) this.W1[i] = (Math.random() - 0.5) * 0.2;
        for (let j = 0; j < nh; j++) { this.W2[j] = (Math.random() - 0.5) * 0.2; this.W2b[j] = (Math.random() - 0.5) * 0.2; }
      }
    }
    forward(f) {
      const h = this.h, W1 = this.W1, H = this.H;
      for (let j = 0; j < H; j++) h[j] = this.b1[j];
      for (let t = 0; t < f.idx.length; t++) {
        const o = f.idx[t] * H, v = f.val[t];
        for (let j = 0; j < H; j++) h[j] += v * W1[o + j];
      }
      let z = this.b2[0] + (f.bias || 0), z2 = this.b2b[0] - (f.bias || 0);
      if (this.deep) {
        for (let j = 0; j < H; j++) h[j] = Math.tanh(h[j]);
        const zz = this.z;
        for (let k = 0; k < this.deep; k++) {
          const W = this.Wd, o = k * H * H;
          for (let j = 0; j < H; j++) zz[j] = this.bd[k * H + j];
          for (let i = 0; i < H; i++) { const v = h[i], oi = o + i * H; for (let j = 0; j < H; j++) zz[j] += v * W[oi + j]; }
          for (let j = 0; j < H; j++) h[j] += Math.tanh(zz[j]);
        }
        for (let j = 0; j < H; j++) { z += h[j] * this.W2[j]; z2 += h[j] * this.W2b[j]; }
      } else for (let j = 0; j < H; j++) { h[j] = Math.tanh(h[j]); z += h[j] * this.W2[j]; z2 += h[j] * this.W2b[j]; }
      this.y[0] = 1 / (1 + Math.exp(-z));
      this.y[1] = this.heads === 2 ? 1 / (1 + Math.exp(-z2)) : 1 - this.y[0];
      return this.y[0];
    }
    // adds sign * d(output k)/dw to the trace E; call right after forward(f)
    addGradHead(f, k, sign, E) {
      if (this.deep) throw new Error('deep nets are trained by cpp/train.cpp only');
      const y = this.y[k], dz = sign * y * (1 - y), h = this.h, H = this.H;
      const W2 = k ? this.W2b : this.W2, EW2 = k ? E.W2b : E.W2, Eb2 = k ? E.b2b : E.b2;
      Eb2[0] += dz;
      for (let j = 0; j < H; j++) {
        EW2[j] += dz * h[j];
        const dh = dz * W2[j] * (1 - h[j] * h[j]);
        E.b1[j] += dh;
        for (let t = 0; t < f.idx.length; t++) E.W1[f.idx[t] * H + j] += f.val[t] * dh;
      }
    }
    addGrad(f, y, sign, E) { this.addGradHead(f, 0, sign, E); }
    toJSON() {
      const r = (a) => Array.from(a, (x) => Math.round(x * 1e5) / 1e5);
      const o = { W1: r(this.W1), b1: r(this.b1), W2: r(this.W2), b2: this.b2[0] };
      if (this.heads === 2) { o.W2b = r(this.W2b); o.b2b = this.b2b[0]; }
      if (this.deep) { o.Wd = r(this.Wd); o.bd = r(this.bd); }
      return o;
    }
  }

  // single output for every rule (the two-headed variant is kept but unused: it did not stop stalling)
  const newNet = (rule, hidden) => new Net(null, ninFor(rule), 1, hidden);

  const NOLOG = { push() {} };
  const cloneState = (s) => ({
    rule: s.rule, board: s.board.slice(), bar: s.bar.slice(), off: s.off.slice(), turn: s.turn,
    phase: s.phase, dice: s.dice.slice(), throw: s.throw, pending: s.pending ? s.pending.slice() : null,
    rolledDouble: s.rolledDouble, winner: s.winner, log: NOLOG,
    plyLimit: s.plyLimit, limitDraw: s.limitDraw, turnCount: s.turnCount, limitHit: s.limitHit, rolls: s.rolls,
  });

  const key = (s) => s.board.join(',') + '|' + s.bar + '|' + s.off + '|' + s.dice.join('') + '|' + s.phase + s.turn;

  // P(player p wins) for a position where the roller is s.turn
  function valueFor(net, s, p) {
    if (s.phase === 'over') return s.winner === null ? 0.5 : s.winner === p ? 1 : 0; // null: drawn at the turn limit
    net.forward(features(s));
    if (net.heads === 2) { // own winning chance minus the other side's, scaled to 0..1
      const me = s.turn === p ? net.y[0] : net.y[1], other = s.turn === p ? net.y[1] : net.y[0];
      return (me - other + 1) / 2;
    }
    return s.turn === p ? net.y[0] : 1 - net.y[0];
  }

  // All distinct end results of the current move or walk segment: [{s, path}].
  function endStates(s0) {
    const p = s0.turn;
    const seen = new Set(), done = new Map();
    const stack = [{ s: s0, path: [] }];
    while (stack.length) {
      const { s, path } = stack.pop();
      if ((s.phase !== 'move' && s.phase !== 'walk') || s.turn !== p) { // segment over (turn may have passed)
        const k = key(s);
        if (!done.has(k)) done.set(k, { s, path });
        continue;
      }
      const k = key(s);
      if (seen.has(k)) continue;
      seen.add(k);
      for (const m of BG.allMoves(s)) {
        const c = cloneState(s);
        BG.applyMove(c, m);
        stack.push({ s: c, path: path.concat([m]) });
      }
    }
    return Array.from(done.values());
  }

  // Mixed, after the mover's own throw is played: the opponent must now walk exactly that throw. The dice
  // are known, so there is no chance node: they take the end state that is best for them, i.e. worst for p.
  // A skipped or dice-less opponent (no legal walk, benched) leaves a single end state.
  function walkValue(net, s, p) {
    if (s.phase !== 'walk') return valueFor(net, s, p);
    // the walker takes the end state that is best for them: worst for p if they are the opponent, best if it is p
    const best = s.turn === p ? Math.max : Math.min;
    let v = s.turn === p ? 0 : 1;
    for (const e of endStates(s)) v = best(v, valueFor(net, e.s, p));
    return v;
  }

  // Same, scored for the mover: [{s, path, value}]. The net only ever scores positions with nobody holding
  // dice; Mixed throws (phase 'move') are looked one walk ahead, Mixed walks are scored directly.
  function candidates(net, s0) {
    const p = s0.turn, ahead = s0.rule === 'mixed' && s0.phase === 'move';
    return endStates(s0).map(({ s, path }) => ({ s, path, value: ahead ? walkValue(net, s, p) : valueFor(net, s, p) }));
  }

  function bestPlan(net, s) {
    let best = null;
    for (const c of candidates(net, s)) if (!best || c.value > best.value) best = c;
    return best;
  }

  // Bot interface used by the UI and by the tournament code.
  function netBot(net) {
    return {
      move(s) { return bestPlan(net, s).path[0]; },
    };
  }

  const randomBot = {
    move(s) { const m = BG.allMoves(s); return m[Math.floor(Math.random() * m.length)]; },
  };


  // Simple baseline: largest die, bear off and hit when possible, avoid walking backwards.
  const greedyBot = {
    hit: 0,
    move(s) {
      const p = s.turn;
      let best = null, bv = -1e9;
      for (const m of BG.allMoves(s)) {
        let v = (m.back ? -m.die : m.die) + (m.to === 'off' ? 6 : 0);
        if (m.to !== 'off' && s.board[BG.rel(p, m.to)] * BG.sgn(p) === -1) v += this.hit;
        v += Math.random() * 0.1;
        if (v > bv) { bv = v; best = m; }
      }
      return best;
    },
  };

  // Plays one full game; bots[0] is White. Returns the winner (0/1) or -1 at the ply limit.
  function playGame(bots, rule, maxPlies) {
    const s = BG.newGame(rule || 'paul');
    s.log = { push() {} };
    for (let n = 0; n < (maxPlies || 5000) && s.phase !== 'over'; n++) {
      const b = bots[s.turn];
      if (s.phase === 'roll') BG.roll(s);
      else BG.applyMove(s, b.move(s));
    }
    return s.phase === 'over' ? s.winner : -1;
  }

  const api = { H, NIN, ninFor, Net, newNet, features, endStates, key, candidates, bestPlan, netBot, randomBot, greedyBot, playGame, cloneState, valueFor, walkValue };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.BotLib = api;
})(typeof window !== 'undefined' ? window : globalThis);
