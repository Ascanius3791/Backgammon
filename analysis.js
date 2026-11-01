// Move analysis on top of the bot's net: expectimax over the dice with a settable depth.
// depth 1 = the net's value of the position after your move (what the bot itself uses);
// depth 2 = also average over the opponent's 21 rolls and his best reply; and so on.
// Values are P(player p wins), which is also the expected score of a game (win 1, loss 0).
// Works in the browser (global AnalysisLib) and in node (require).
(function (root) {
  'use strict';
  const L = typeof require !== 'undefined' ? require('./bot.js') : root.BotLib;
  const BG = typeof require !== 'undefined' ? require('./engine.js') : root.BG;
  // the C++ core as WebAssembly (core_wasm.js, cpp/): same values, about 12x faster; optional
  const C = typeof require !== 'undefined' ? (() => { try { return require('./core_wasm.js'); } catch (e) { return null; } })() : root.CoreWasm;

  const ROLLS = [];
  for (let a = 1; a <= 6; a++) for (let b = a; b <= 6; b++) ROLLS.push({ dice: [a, b], w: a === b ? 1 / 36 : 2 / 36 });

  const KEEP = 4; // at depth 3+, only the best KEEP replies by 1-ply value are searched further

  function evalStateJS(net, s, p, d) {
    if (s.phase === 'over') return s.winner === p ? 1 : 0;
    if (d <= 1 && s.phase !== 'move') return L.walkValue(net, s, p); // 'move' = just rolled: still needs its best reply
    if (s.phase === 'roll') {
      let v = 0;
      for (const r of ROLLS) {
        const c = L.cloneState(s);
        BG.roll(c, r.dice);
        v += r.w * evalStateJS(net, c, p, d - 1);
      }
      return v;
    }
    // decision node: the side to move picks the end state that is best for itself
    let ends = L.endStates(s);
    const mine = s.turn === p;
    if (d >= 2 && ends.length > KEEP) {
      ends.forEach((e) => { e.q = L.walkValue(net, e.s, p); });
      ends.sort((x, y) => (mine ? y.q - x.q : x.q - y.q));
      ends = ends.slice(0, KEEP);
    }
    let best = mine ? -1 : 2;
    for (const e of ends) {
      const v = evalStateJS(net, e.s, p, d);
      if (mine ? v > best : v < best) best = v;
    }
    return best;
  }

  function evalState(net, s, p, d) {
    return C && C.ready() ? C.evalState(s.rule, net, s, p, d) : evalStateJS(net, s, p, d);
  }

  // Generator: scores every end result of the mover's segment, yielding after each one so a
  // caller can time-slice. Returns [{path, key, value}] sorted best first.
  function* analyze(net, s0, depth) {
    const p = s0.turn;
    const ends = L.endStates(s0);
    const out = [];
    for (let i = 0; i < ends.length; i++) {
      const e = ends[i];
      out.push({ path: e.path, key: L.key(e.s), value: evalState(net, e.s, p, depth) });
      yield (i + 1) / ends.length;
    }
    out.sort((a, b) => b.value - a.value);
    return out;
  }

  function analyzeSync(net, s0, depth) {
    const g = analyze(net, s0, depth);
    for (;;) { const r = g.next(); if (r.done) return r.value; }
  }

  const api = { analyze, analyzeSync, evalState, evalStateJS };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.AnalysisLib = api;
})(typeof window !== 'undefined' ? window : globalThis);
