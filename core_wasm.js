// JS side of the WebAssembly core (cpp/core.h, built by cpp/Makefile): loads bgcore.wasm and exchanges
// engine.js states with it. Browser: global CoreWasm, module bytes from bgcore_wasm.js (instantiated
// asynchronously, so ready() is false for a moment). Node: require, loaded synchronously from bgcore.wasm.
// The search workers of pool.js run the same code (coreApi) on their own instance of the module.
(function (root) {
  'use strict';

  // Everything that talks to one module instance. Self-contained: its source is also put into the workers.
  function coreApi(ex) {
    const RULE_ID = { paul: 0, rene: 1, mixed: 2 };
    const PHASE_ID = { roll: 0, move: 1, walk: 2, over: 3 };
    const CAND_W = 6;
    const loaded = [null, null, null]; // net object whose weights are in the module, per rule
    ex.init();

    const f32 = (ptr, n) => new Float32Array(ex.memory.buffer, ptr, n);
    const i32 = (ptr, n) => new Int32Array(ex.memory.buffer, ptr, n);

    // weights of a bot.js Net (or the parsed JSON) into the module, once per net object
    function useNet(rule, net) {
      const r = RULE_ID[rule];
      if (loaded[r] === net) return;
      const b2 = typeof net.b2 === 'number' ? net.b2 : net.b2[0], H = net.b1.length;
      if (!ex.setShape(r, net.W1.length / H, H)) throw new Error('bgcore.wasm has no build for ' + H + ' hidden units');
      f32(ex.w1Ptr(r), net.W1.length).set(net.W1);
      f32(ex.b1Ptr(r), H).set(net.b1);
      f32(ex.w2Ptr(r), H).set(net.W2);
      f32(ex.b2Ptr(r), 1)[0] = b2;
      loaded[r] = net;
    }

    // engine.js state -> module position (io layout: see cpp/wasm.cpp load/store)
    function load(s) {
      const io = i32(ex.ioPtr(), 64);
      io[0] = RULE_ID[s.rule]; io[1] = s.plyLimit || 0; io[2] = s.turnCount || 0;
      for (let a = 1; a <= 24; a++) io[2 + a] = s.board[a];
      io[27] = s.bar[0]; io[28] = s.bar[1]; io[29] = s.off[0]; io[30] = s.off[1];
      io[31] = s.turn; io[32] = PHASE_ID[s.phase];
      for (let d = 1; d <= 6; d++) io[32 + d] = 0;
      for (const d of s.dice) io[32 + d]++;
      io[39] = s.throw ? s.throw[0] : 0; io[40] = s.throw ? s.throw[1] : 0;
      io[41] = s.pending ? 1 : 0; io[42] = s.rolledDouble ? 1 : 0;
      io[43] = s.winner === null || s.winner === undefined ? -1 : s.winner; io[44] = s.limitHit ? 1 : 0;
      ex.load();
    }

    const decode = (m) => {
      const f = m & 31, t = (m >> 5) & 31, o = { from: f === 25 ? 'bar' : f, to: t === 0 ? 'off' : t, die: (m >> 10) & 7 };
      if ((m >> 13) & 1) o.back = true;
      return o;
    };
    const encode = (m) => (m.from === 'bar' ? 25 : m.from) | ((m.to === 'off' ? 0 : m.to) << 5) | (m.die << 10) | ((m.back ? 1 : 0) << 13);

    return {
      useNet, load, decode, encode,
      exports: () => ex,
      // analysis.js evalState: P(player p wins) in s after searching depth more rolls
      evalState(rule, net, s, p, depth) { useNet(rule, net); load(s); return ex.evalNode(p, depth, 1); },
      // the same with the window (alpha, beta), fail-soft (a result <= alpha or >= beta is only a bound)
      evalWindow(rule, net, s, p, depth, alpha, beta) { useNet(rule, net); load(s); return ex.evalNodeW(p, depth, alpha, beta); },
      // best plan for the side to move: { path: [engine.js moves], value }
      bestPlan(rule, net, s, depth) {
        useNet(rule, net); load(s);
        const value = ex.bestPlan(depth, 1), io = i32(ex.ioPtr(), 64), path = [];
        for (let k = 0; k < io[48]; k++) path.push(decode(io[49 + k]));
        return { path, value };
      },
      // every way to finish the segment of s, best first by 1-ply value: [{ path: [engine.js moves], q }]
      cands(rule, net, s) {
        useNet(rule, net); load(s);
        const n = ex.cands(), o = f32(ex.foutPtr(), n * CAND_W), out = [];
        for (let i = 0; i < n; i++) {
          const path = [];
          for (let k = 0; k < o[i * CAND_W]; k++) path.push(decode(o[i * CAND_W + 1 + k]));
          out.push({ path, q: o[i * CAND_W + 5] });
        }
        return out;
      },
    };
  }

  let core = null, wasmBytes = null, readyCbs = [];
  function setup(instance) {
    core = coreApi(instance.exports);
    for (const f of readyCbs) f();
    readyCbs = null;
  }
  if (typeof module !== 'undefined' && module.exports) {
    try {
      wasmBytes = require('fs').readFileSync(require('path').join(__dirname, 'bgcore.wasm'));
      setup(new WebAssembly.Instance(new WebAssembly.Module(wasmBytes)));
    } catch (e) { core = null; }
  } else if (root.BG_WASM_B64 && root.WebAssembly) {
    const bin = atob(root.BG_WASM_B64), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    wasmBytes = bytes;
    WebAssembly.instantiate(bytes).then((r) => setup(r.instance)).catch(() => { core = null; });
  }

  const call = (name) => (...a) => core[name](...a);
  const api = {
    ready: () => core !== null,
    onReady(f) { if (core) f(); else if (readyCbs) readyCbs.push(f); },
    exports: () => core && core.exports(),
    useNet: call('useNet'), load: call('load'), decode: call('decode'), encode: call('encode'),
    evalState: call('evalState'), evalWindow: call('evalWindow'), bestPlan: call('bestPlan'), cands: call('cands'),
    wasmBytes: () => wasmBytes,
    coreApiSource: coreApi.toString(),
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.CoreWasm = api;
})(typeof window !== 'undefined' ? window : globalThis);
