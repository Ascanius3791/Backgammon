// Deep searches on background threads: depth 3 and 4 take seconds to minutes, so they must not run on the
// page itself. Web Workers made from a blob (allowed from file://), each with its own instance of
// bgcore.wasm (core_wasm.js coreApi). A job is one call of the core's search: P(player p wins) in s after
// depth more rolls, with the window (alpha, beta).
// best() splits the root over the workers: the move with the best 1-ply value is searched first with the
// full window, then all others in parallel, each with the best value known when it starts as alpha (so the
// core cuts them off as soon as they cannot be better). The result is the exact best move and value.
// Jobs start by priority (lower first). A job may stop the running job with the worst priority, if that is
// worse than its own; the stopped job is restarted later.
// Without workers (node, or a browser that refuses them) the jobs run one by one on the page's own module.
// Browser: global SearchPool. Node: require.
(function (root) {
  'use strict';
  const isNode = typeof module !== 'undefined' && module.exports;
  const C = isNode ? require('./core_wasm.js') : root.CoreWasm;
  const L = isNode ? require('./bot.js') : root.BotLib;
  const BG = isNode ? require('./engine.js') : root.BG;

  // the worker: coreApi (from core_wasm.js) on its own module instance, one job per message
  function workerMain() {
    let core = null;
    const nets = {};
    onmessage = (e) => {
      const m = e.data;
      if (m.wasm) { core = coreApi(new WebAssembly.Instance(new WebAssembly.Module(m.wasm)).exports); return; }
      if (m.net) nets[m.rule] = m.net;
      postMessage(core.evalWindow(m.rule, nets[m.rule], m.s, m.p, m.depth, m.alpha, m.beta));
    };
  }

  // the fields of an engine.js state the core reads, as plain data (states carry a log that cannot be sent)
  const pack = (s) => ({
    rule: s.rule, plyLimit: s.plyLimit, turnCount: s.turnCount, limitHit: s.limitHit, board: s.board.slice(), bar: s.bar.slice(),
    off: s.off.slice(), turn: s.turn, phase: s.phase, dice: s.dice.slice(), throw: s.throw, pending: s.pending, rolledDouble: s.rolledDouble, winner: s.winner,
  });

  class WorkerSlot {
    constructor(pool) { this.pool = pool; this.job = null; this.spawn(); }
    spawn() {
      this.nets = {}; // net objects whose weights this worker has
      this.w = new Worker(this.pool.url);
      this.w.onmessage = (e) => this.pool.done(this, e.data);
      this.w.onerror = () => this.pool.fallback();
      this.w.postMessage({ wasm: C.wasmBytes() });
    }
    start(job) {
      this.job = job;
      const m = { rule: job.rule, s: job.s, p: job.p, depth: job.depth, alpha: job.alpha(), beta: job.beta };
      if (this.nets[job.rule] !== job.net) {
        const n = job.net;
        m.net = { W1: n.W1, b1: n.b1, W2: n.W2, b2: n.b2 };
        this.nets[job.rule] = n;
      }
      this.w.postMessage(m);
    }
    kill() { this.w.terminate(); this.job = null; this.spawn(); }
    close() { this.w.terminate(); }
  }

  // runs the job on the page's module, in a timer callback (a running call cannot be stopped)
  class InlineSlot {
    constructor(pool) { this.pool = pool; this.job = null; }
    start(job) {
      this.job = job;
      setTimeout(() => {
        if (this.job !== job) return;
        this.pool.done(this, C.evalWindow(job.rule, job.net, job.s, job.p, job.depth, job.alpha(), job.beta));
      }, 0);
    }
    kill() { this.job = null; }
    close() {}
  }

  class Pool {
    constructor(n) {
      this.queue = [];
      this.seq = 0;
      this.slots = [];
      try {
        if (isNode || typeof Worker === 'undefined' || !C.wasmBytes()) throw new Error('no workers');
        const src = `${C.coreApiSource}\n(${workerMain})();`;
        this.url = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
        for (let i = 0; i < n; i++) this.slots.push(new WorkerSlot(this));
      } catch (e) {
        this.fallback();
      }
    }
    get size() { return this.slots.length; }
    // workers failed: drop them, run everything (including what was running) inline
    fallback() {
      if (this.inline) return;
      this.inline = true;
      for (const sl of this.slots) {
        if (sl.job) this.queue.push(sl.job);
        sl.close();
      }
      this.slots = [new InlineSlot(this)];
      this.schedule();
    }
    submit(job) {
      if (job.group.cancelled) return;
      job.seq = this.seq++;
      this.queue.push(job);
      this.schedule();
    }
    schedule() {
      const worse = (a, b) => a.pri > b.pri || (a.pri === b.pri && a.seq > b.seq);
      this.queue.sort((a, b) => a.pri - b.pri || a.seq - b.seq);
      while (this.queue.length) {
        let sl = this.slots.find((x) => !x.job);
        if (!sl) { // preempt the worst running job, if it has a worse priority than the next one
          let w = null;
          for (const x of this.slots) if (!w || worse(x.job, w.job)) w = x;
          if (w.job.pri <= this.queue[0].pri) return;
          const j = w.job;
          w.kill();
          this.queue.push(j);
          this.queue.sort((a, b) => a.pri - b.pri || a.seq - b.seq);
          sl = w;
        }
        sl.start(this.queue.shift());
      }
    }
    done(slot, v) {
      const job = slot.job;
      slot.job = null;
      if (job && !job.group.cancelled) job.resolve(v);
      this.schedule();
    }
    cancel(group) {
      group.cancelled = true;
      this.queue = this.queue.filter((j) => j.group !== group);
      for (const sl of this.slots) if (sl.job && sl.job.group === group) sl.kill();
      this.schedule();
    }
    cancelAll() {
      this.queue = [];
      for (const sl of this.slots) if (sl.job) { sl.job.group.cancelled = true; sl.kill(); }
    }
    busy() { return this.queue.length > 0 || this.slots.some((x) => x.job); }

    // exact value of one end state s for player p (a move already played): { promise, cancel }
    value(rule, net, s, p, depth, pri) {
      const group = {};
      const promise = new Promise((resolve) => {
        this.submit({ group, pri, rule, net, s: pack(s), p, depth, alpha: () => -1, beta: 2, resolve });
      });
      return { promise, cancel: () => this.cancel(group) };
    }

    // best way to finish the segment of s at the given depth: { promise, cancel }. The promise gives
    // { path, value, key, n } (key: BotLib.key of the end state, n: number of candidates), or null if
    // s has no move. onProgress(done, n) after each finished candidate.
    best(rule, net, s0, depth, pri, onProgress) {
      const group = {};
      const promise = new Promise((resolve) => {
        const p = s0.turn, cs = C.cands(rule, net, s0);
        if (!cs.length) return resolve(null);
        for (const c of cs) {
          const e = L.cloneState(s0);
          for (const m of c.path) BG.applyMove(e, m);
          c.end = pack(e);
          c.key = L.key(e);
        }
        const finish = (i, v) => resolve({ path: cs[i].path, value: v, key: cs[i].key, n: cs.length });
        if (depth <= 1) return finish(0, cs[0].q);
        let best = -1, bi = 0, done = 0;
        const job = (i, alpha) => this.submit({
          group, pri, rule, net, s: cs[i].end, p, depth, alpha, beta: 2,
          resolve: (v) => {
            if (v > best) { best = v; bi = i; } // v > the alpha it was searched with: exact
            done++;
            if (onProgress) onProgress(done, cs.length);
            if (done === cs.length) finish(bi, best);
            else if (i === 0) for (let k = 1; k < cs.length; k++) job(k, () => best);
          },
        });
        job(0, () => -1);
      });
      return { promise, cancel: () => this.cancel(group) };
    }
  }

  const defaultSize = () => Math.max(1, Math.min(8, ((typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4) - 1));
  const api = { create: (n) => new Pool(n || defaultSize()), pack };
  if (isNode) module.exports = api;
  else root.SearchPool = api;
})(typeof window !== 'undefined' ? window : globalThis);
