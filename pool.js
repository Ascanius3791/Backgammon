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
      if (m.rr) postMessage(core.rollRange(m.rule, nets[m.rule], m.ends, m.p, m.i0, m.n, m.seedHi, m.seedLo, m.trunc, m.cap));
      else if (m.roll) postMessage(core.rollouts(m.rule, nets[m.rule], m.a, m.s, m.p, m.n, m.seed, m.cap));
      else postMessage(core.evalWindow(m.rule, nets[m.rule], m.s, m.p, m.depth, m.alpha, m.beta));
    };
  }

  // the fields of an engine.js state the core reads, as plain data (states carry a log that cannot be sent)
  const pack = (s) => ({
    rule: s.rule, plyLimit: s.plyLimit, turnCount: s.turnCount, limitHit: s.limitHit, board: s.board.slice(), bar: s.bar.slice(),
    off: s.off.slice(), turn: s.turn, phase: s.phase, dice: s.dice.slice(), throw: s.throw, pending: s.pending, rolledDouble: s.rolledDouble, winner: s.winner,
  });

  // what a worker needs for a job: a search (window), a batch of rollouts (job.roll) or of the rollout bot's
  // truncated rollouts (job.rr)
  const jobMsg = (job) => job.rr
    ? { rule: job.rule, rr: true, ends: job.ends, p: job.p, i0: job.i0, n: job.n, seedHi: job.seedHi, seedLo: job.seedLo, trunc: job.trunc, cap: job.cap }
    : job.roll
    ? { rule: job.rule, roll: true, a: job.a, s: job.s, p: job.p, n: job.n, seed: job.seed, cap: job.cap }
    : { rule: job.rule, s: job.s, p: job.p, depth: job.depth, alpha: job.alpha(), beta: job.beta };
  const runInline = (job) => job.rr
    ? C.rollRange(job.rule, job.net, job.ends, job.p, job.i0, job.n, job.seedHi, job.seedLo, job.trunc, job.cap)
    : job.roll
    ? C.rollouts(job.rule, job.net, job.a, job.s, job.p, job.n, job.seed, job.cap)
    : C.evalWindow(job.rule, job.net, job.s, job.p, job.depth, job.alpha(), job.beta);

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
      const m = jobMsg(job);
      if (this.nets[job.rule] !== job.net) {
        const n = job.net;
        m.net = { W1: n.W1, b1: n.b1, W2: n.W2, b2: n.b2 };
        if (n.deep) { m.net.Wd = n.Wd; m.net.bd = n.bd; }
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
        this.pool.done(this, runInline(job));
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

    // Rollouts for player p for about ms milliseconds on all workers: games played out by the 1-ply bot from
    // b and, if a is given, from a with the same dice (pairs). { promise, cancel }; the promise and
    // onProgress(res) give res = { n (pairs), a, b (win rates), sa, sb (their 1-sigma errors), d (b - a),
    // sd (its error, from the paired differences), ms }. Batches are sized to take ~0.2 s from the measured
    // speed, so the budget is kept to about that.
    rollout(rule, net, a, b, p, ms, pri, onProgress) {
      const group = {}, now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
      const pa = a ? pack(a) : null, pb = pack(b), cap = rule === 'mixed' ? 800 : 5000;
      const t0 = now(), end = t0 + ms, sum = [0, 0, 0, 0, 0];
      let n = 0, inflight = 0, busy = 0; // busy: worker-milliseconds of the finished batches
      const promise = new Promise((resolve) => {
        const stats = () => {
          const ma = sum[0] / n, mb = sum[1] / n, f = n > 1 ? n / (n - 1) : 1;
          const se = (v) => Math.sqrt(Math.max(0, v) * f / n);
          return { n, a: pa ? ma : null, b: mb, sa: pa ? se(sum[2] / n - ma * ma) : null, sb: se(sum[3] / n - mb * mb),
            d: pa ? mb - ma : null, sd: pa ? se((sum[3] - 2 * sum[4] + sum[2]) / n - (mb - ma) ** 2) : null, ms: now() - t0 };
        };
        const launch = () => {
          const left = end - now(), per = n ? busy / n : 0; // worker-ms per pair
          const k = !per ? 2 : Math.min(Math.round(200 / per), Math.floor(left / per));
          if (left <= 0 || k < 1) return false;
          const st = now();
          inflight++;
          this.submit({
            group, pri, rule, net, roll: true, a: pa, s: pb, p, n: k, cap, seed: (Math.random() * 2 ** 32) >>> 0,
            resolve: (r) => {
              inflight--;
              for (let i = 0; i < 5; i++) sum[i] += r[i];
              n += k; busy += now() - st;
              if (onProgress) onProgress(stats());
              while (inflight < this.size && launch());
              if (!inflight) resolve(stats());
            },
          });
          return true;
        };
        for (let i = 0; i < this.size; i++) launch();
        if (!inflight) resolve(stats());
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

    // The rollout bot: the best way to finish the segment of s0 by truncated rollouts, in about ms milliseconds.
    // 1. every candidate end gets its depth-2 value (prefilter; 1-ply values of moves the bot never plays are
    //    unreliable); the best KEEP within MARGIN of the best go on.
    // 2. Racing: batches of games run on all workers, game i with the same throws from every candidate (paired);
    //    each game is played by the 1-ply bot for trunc throws, then the net's value counts. After each batch a
    //    candidate is dropped once the leader is ahead by more than Z standard errors of the paired difference
    //    (from MINN games on). Ends when one candidate is left or the time is up; the leader by mean is played.
    // { promise, cancel }; the promise gives { path, value, key, n, games, ms, cands: [{ path, key, d2, mean, se, n, alive }] }
    // (se: error of the difference to the chosen move; mean null if it played no game), or null without a move. onProgress(info) with the same fields.
    rolloutBest(rule, net, s0, ms, pri, onProgress, opt = {}) {
      const trunc = opt.trunc || 8, KEEP = opt.keep || 8, MARGIN = opt.margin || 0.25, Z = opt.z || 2, MINN = opt.minn || 72;
      const group = {}, now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
      const t0 = now(), p = s0.turn;
      const promise = new Promise((resolve) => {
        const cs = C.cands(rule, net, s0);
        if (!cs.length) return resolve(null);
        for (const c of cs) {
          const e = L.cloneState(s0);
          for (const m of c.path) BG.applyMove(e, m);
          c.end = pack(e);
          c.key = L.key(e);
        }
        const out = (i, extra) => Object.assign({ path: cs[i].path, key: cs[i].key, n: cs.length, ms: now() - t0 }, extra);
        if (cs.length === 1) return resolve(out(0, { value: cs[0].q, games: 0, cands: [] }));
        // 1. depth-2 values of all ends (full window: exact)
        let left = cs.length;
        for (let i = 0; i < cs.length; i++) this.submit({
          group, pri, rule, net, s: cs[i].end, p, depth: 2, alpha: () => -1, beta: 2,
          resolve: (v) => { cs[i].d2 = v; if (--left === 0) race(); },
        });
        // 2. racing
        const race = () => {
          const order = cs.map((c, i) => i).sort((a, b) => cs[b].d2 - cs[a].d2);
          const top = cs[order[0]].d2;
          const R = order.slice(0, KEEP).filter((i) => cs[i].d2 >= top - MARGIN)
            .map((i) => ({ i, res: [], alive: true, sum: 0, n: 0 })); // res[g]: result of game g (undefined: not played)
          const seedHi = (Math.random() * 2 ** 32) >>> 0, seedLo = (Math.random() * 2 ** 32) >>> 0;
          const cap = rule === 'mixed' ? 800 : 5000, end = t0 + ms;
          let next = 0, inflight = 0, busy = 0, played = 0, finished = false; // busy: worker-ms per game and candidate, summed
          const alive = () => R.filter((r) => r.alive);
          const mean = (r) => (r.n ? r.sum / r.n : -1);
          const leader = () => alive().reduce((b, r) => (mean(r) > mean(b) ? r : b));
          // mean and standard error of b - r over the games both have
          const diff = (b, r) => {
            let n = 0, s = 0, ss = 0;
            const m = Math.min(b.res.length, r.res.length);
            for (let g = 0; g < m; g++) {
              const x = b.res[g], y = r.res[g];
              if (x === undefined || y === undefined) continue;
              const d = x - y;
              n++; s += d; ss += d * d;
            }
            if (n < 2) return { n, d: 0, se: Infinity };
            const d = s / n;
            return { n, d, se: Math.sqrt(Math.max(ss / n - d * d, 1e-8) / (n - 1)) };
          };
          const info = () => {
            const b = leader();
            const list = R.map((r) => {
              const df = r === b ? { se: 0 } : diff(b, r);
              return { path: cs[r.i].path, key: cs[r.i].key, d2: cs[r.i].d2, mean: r.n ? mean(r) : null, se: df.se, n: r.n, alive: r.alive };
            }).sort((x, y) => (y.n ? y.mean : -1) - (x.n ? x.mean : -1));
            // no game in time (the prefilter took it all): the best move by depth 2 and its value
            return out(b.i, { value: b.n ? mean(b) : cs[b.i].d2, games: played, cands: list });
          };
          const finish = () => { if (!finished) { finished = true; resolve(info()); } };
          const prune = () => {
            const b = leader();
            for (const r of alive()) {
              if (r === b) continue;
              const df = diff(b, r);
              if (df.n >= MINN && df.d > Z * df.se) r.alive = false;
            }
          };
          if (R.length === 1) { R[0].n = 1; R[0].sum = cs[R[0].i].d2; return finish(); }
          const launch = () => {
            const rs = alive(), rem = end - now();
            if (rs.length < 2 || rem <= 0) return false;
            const per = played ? busy / played : 0; // worker-ms per game and candidate
            const k = !per ? 4 : Math.max(1, Math.min(Math.round(150 / (per * rs.length)), Math.floor(rem / (per * rs.length))));
            if (per && rem < per * rs.length) return false;
            const i0 = next, st = now();
            next += k; inflight++;
            this.submit({
              group, pri, rule, net, rr: true, ends: rs.map((r) => cs[r.i].end), p, i0, n: k, seedHi, seedLo, trunc, cap,
              resolve: (res) => {
                inflight--;
                busy += now() - st; played += k * rs.length;
                rs.forEach((r, j) => {
                  for (let g = 0; g < k; g++) { const x = res[j][g]; r.res[i0 + g] = x; r.sum += x; r.n++; }
                });
                prune();
                if (onProgress) onProgress(info());
                while (inflight < this.size && launch());
                if (!inflight) finish();
              },
            });
            return true;
          };
          for (let w = 0; w < this.size; w++) launch();
          if (!inflight) finish();
        };
      });
      return { promise, cancel: () => this.cancel(group) };
    }
  }

  const defaultSize = () => Math.max(1, Math.min(8, ((typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4) - 1));
  const api = { create: (n) => new Pool(n || defaultSize()), pack };
  if (isNode) module.exports = api;
  else root.SearchPool = api;
})(typeof window !== 'undefined' ? window : globalThis);
