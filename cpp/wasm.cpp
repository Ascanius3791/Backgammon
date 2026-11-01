// WebAssembly build of core.h for the browser (and for check_cpp.js in node). No C/C++ library:
// the few memory functions the compiler needs are defined here. JS side: core_wasm.js.
// Positions are exchanged through the int array io (layout in core_wasm.js, toIo/fromIo).
#include "core.h"

extern "C" {
void* memcpy(void* d, const void* s, unsigned long n) { char* a = (char*)d; const char* b = (const char*)s; while (n--) *a++ = *b++; return d; }
void* memmove(void* d, const void* s, unsigned long n) {
  char* a = (char*)d; const char* b = (const char*)s;
  if (a < b) while (n--) *a++ = *b++;
  else { a += n; b += n; while (n--) *--a = *--b; }
  return d;
}
void* memset(void* d, int c, unsigned long n) { char* a = (char*)d; while (n--) *a++ = (char)c; return d; }
void __wasm_call_ctors();
}

#define EXPORT(name) extern "C" __attribute__((export_name(#name)))
void* operator new(unsigned long, void* p) noexcept { return p; }

using namespace bg;

// One engine, TD state and current position per compiled hidden size; calls go to the one that fits the
// net of the current rule (set by load and tdInit). Engine and TD have nonzero member initializers: as
// plain statics they would be stored in the module (MBs of mostly zeros). Constructed in init() instead,
// in zeroed memory.
template <int H> struct Ctx { Engine<H> eng; TD<H> td; Pos<H> cur, alt; }; // alt: second position of rollouts
#define BG_MEM(n) alignas(64) static unsigned char mem##n[sizeof(Ctx<n>)];
BG_SIZES(BG_MEM)
static int curH = 64;
template <class F> static auto with(F&& f) {
  switch (curH) {
#define BG_CASE(n) case n: return f(*reinterpret_cast<Ctx<n>*>(mem##n));
    BG_SIZES(BG_CASE)
  }
  return f(*reinterpret_cast<Ctx<64>*>(mem64)); // not reached: setShape only accepts compiled sizes
}
static Net nets[3];
static int io[64];
static constexpr int ENDS_MAX = 4096, END_W = 29; // per end: board 1..24, bar0, bar1, off0, off1, 1-ply value
static float fout[ENDS_MAX * END_W];

EXPORT(init) void init() {
  __wasm_call_ctors(); initTables();
#define BG_NEW(n) new (mem##n) Ctx<n>;
  BG_SIZES(BG_NEW)
  for (auto& n : nets) n.h = 64;
}
EXPORT(ioPtr) int* ioPtr() { return io; }
EXPORT(foutPtr) float* foutPtr() { return fout; }
EXPORT(w1Ptr) float* w1Ptr(int r) { return nets[r].W1; }
EXPORT(b1Ptr) float* b1Ptr(int r) { return nets[r].b1; }
EXPORT(w2Ptr) float* w2Ptr(int r) { return nets[r].W2; }
EXPORT(b2Ptr) float* b2Ptr(int r) { return &nets[r].b2; }
// shape of nets[r]: inputs and hidden units (0 if h is not a compiled size); weights are laid out by it
EXPORT(setShape) int setShape(int r, int nin, int h) {
  if (!hiddenOk(h)) return 0;
  nets[r].nin = nin; nets[r].h = h;
  return 1;
}
EXPORT(evalCount) double evalCount() {
  double sum = 0;
#define BG_COUNT(k) sum += (double)reinterpret_cast<Ctx<k>*>(mem##k)->eng.nEval;
  BG_SIZES(BG_COUNT)
  return sum;
}

// io -> cur (and the engine's rule)
EXPORT(load) void load() {
  curH = nets[io[0]].h;
  with([](auto& c) {
    auto& cur = c.cur;
    c.eng.setRule(io[0], io[1], &nets[io[0]]);
    cur.turnCount = (int16_t)io[2];
    cur.b[0] = 0;
    for (int a = 1; a <= 24; a++) cur.b[a] = (int8_t)io[2 + a];
    cur.bar[0] = (uint8_t)io[27]; cur.bar[1] = (uint8_t)io[28]; cur.off[0] = (uint8_t)io[29]; cur.off[1] = (uint8_t)io[30];
    cur.turn = (uint8_t)io[31]; cur.phase = (uint8_t)io[32];
    cur.cnt[0] = 0;
    for (int d = 1; d <= 6; d++) cur.cnt[d] = (uint8_t)io[32 + d];
    cur.t0 = (uint8_t)io[39]; cur.t1 = (uint8_t)io[40]; cur.pending = (uint8_t)io[41]; cur.rolledDouble = (uint8_t)io[42];
    cur.winner = (int8_t)io[43]; cur.limitHit = (uint8_t)io[44];
    c.eng.refresh(cur);
  });
}
// cur -> io
EXPORT(store) void store() {
  with([](auto& c) {
    const auto& cur = c.cur;
    io[2] = cur.turnCount;
    for (int a = 1; a <= 24; a++) io[2 + a] = cur.b[a];
    io[27] = cur.bar[0]; io[28] = cur.bar[1]; io[29] = cur.off[0]; io[30] = cur.off[1];
    io[31] = cur.turn; io[32] = cur.phase;
    for (int d = 1; d <= 6; d++) io[32 + d] = cur.cnt[d];
    io[39] = cur.t0; io[40] = cur.t1; io[41] = cur.pending; io[42] = cur.rolledDouble; io[43] = cur.winner; io[44] = cur.limitHit;
  });
}

EXPORT(roll) void roll(int a, int b) { with([&](auto& c) { c.eng.roll(c.cur, a, b); }); store(); }
EXPORT(applyMove) void applyMove(int m) { with([&](auto& c) { c.eng.applyMove(c.cur, m); }); store(); }

// P(player p wins) in cur after searching depth more rolls (analysis.js evalState).
// ab = 0 switches off the alpha-beta cutoffs inside the tree (for comparisons).
EXPORT(evalNode) float evalNode(int p, int depth, int ab) {
  return with([&](auto& c) {
    c.eng.ab = ab != 0;
    return c.eng.node(c.cur, p, depth, -1.f, 2.f);
  });
}

// the same with the window (alpha, beta), fail-soft: <= alpha an upper bound, >= beta a lower bound
EXPORT(evalNodeW) float evalNodeW(int p, int depth, float alpha, float beta) {
  return with([&](auto& c) {
    c.eng.ab = true;
    return c.eng.node(c.cur, p, depth, alpha, beta);
  });
}

// all ends of cur's segment, best first by 1-ply value for the mover, into fout: per end CAND_W numbers
// (path length, the moves path[0..3], value); returns the count
static constexpr int CAND_W = 6;
EXPORT(cands) int cands() {
  return with([](auto& c) {
    int n = c.eng.rootCands(c.cur);
    if (n > ENDS_MAX * END_W / CAND_W) n = ENDS_MAX * END_W / CAND_W;
    for (int i = 0; i < n; i++) {
      float* o = fout + i * CAND_W;
      const Cand& k = c.eng.cands[i];
      o[0] = (float)k.len;
      for (int j = 0; j < MAXD; j++) o[1 + j] = j < k.len ? (float)k.path[j] : 0.f;
      o[5] = k.q;
    }
    return n;
  });
}

// best plan of the side to move in cur at the given depth: value returned, plan in io[48] (length), io[49..]
static int g_survive = 0; // 0: score every end of an own throw by its walk search; n: only the n best by cheap value
EXPORT(setSurvive) void setSurvive(int n) { g_survive = n; }
EXPORT(bestPlan) float bestPlan(int depth, int prune) {
  return with([&](auto& c) {
    auto& eng = c.eng;
    eng.ab = true;
    eng.survive = g_survive;
    float v = eng.bestPlan(c.cur, depth, prune != 0);
    io[48] = eng.bestLen;
    for (int k = 0; k < eng.bestLen; k++) io[49 + k] = eng.bestPath[k];
    return v;
  });
}

// Rollouts: games played out by the bot (1-ply, both sides), P(player p wins) estimated from their results.
// keepAlt copies cur to alt; rollouts(p, n, seed, both, cap) then plays n games from cur and, if both, n from
// alt with the same dice (game i of both from seed + i: common random numbers, so the difference of the two
// win rates has a much smaller error than each one). A game cut after cap rolls and moves counts 1/2.
// fout[0..4] = sums of a, b, a*a, b*b, a*b over the games (a from alt, b from cur).
EXPORT(keepAlt) void keepAlt() { with([](auto& c) { c.alt = c.cur; }); }
template <int H> static float playOut(Engine<H>& eng, Pos<H> s, int p, uint64_t seed, int cap) {
  Rng r;
  r.s = seed;
  int steps = 0;
  while (s.phase != OVER && steps < cap) {
    if (s.phase == ROLL) { eng.roll(s, r.die(), r.die()); steps++; }
    else eng.playBest(s, &steps);
  }
  return s.phase == OVER ? (s.winner == p ? 1.f : 0.f) : 0.5f;
}
EXPORT(rollouts) void rollouts(int p, int n, unsigned seed, int both, int cap) {
  with([&](auto& c) {
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    for (int i = 0; i < n; i++) {
      uint64_t z = ((uint64_t)seed << 32) + (uint64_t)i + 0x9E3779B97F4A7C15ull; // splitmix64
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= z >> 31;
      z |= 1;
      float b = playOut(c.eng, c.cur, p, z, cap), a = both ? playOut(c.eng, c.alt, p, z, cap) : b;
      sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
    }
    fout[0] = (float)sa; fout[1] = (float)sb; fout[2] = (float)saa; fout[3] = (float)sbb; fout[4] = (float)sab;
  });
}

// all ends of cur's segment with their 1-ply value for the mover, into fout; returns the count
EXPORT(ends) int ends() {
  return with([](auto& c) {
    auto& eng = c.eng;
    int n = 0, p = c.cur.turn;
    eng.enumerate(c.cur, [&](auto& C, int) {
      if (n == ENDS_MAX) return false;
      float* o = fout + n++ * END_W;
      for (int a = 1; a <= 24; a++) o[a - 1] = C.b[a];
      o[24] = C.bar[0]; o[25] = C.bar[1]; o[26] = C.off[0]; o[27] = C.off[1];
      o[28] = eng.leafValue(C, p);
      return true;
    });
    return n;
  });
}

// TD(lambda) on nets[rule]: forward on cur, trace with sign, update by delta. Returns the forward value.
// tdInit selects the rule's hidden size; load positions of the same rule after it.
EXPORT(tdInit) void tdInit(int r, float lambda) { curH = nets[r].h; with([&](auto& c) { c.td.init(&nets[r], lambda); }); }
EXPORT(tdForward) float tdForward() { return with([](auto& c) { return (float)c.td.forward(c.eng, c.cur); }); }
EXPORT(tdStep) float tdStep(int sign, float delta) {
  return with([&](auto& c) {
    float y = (float)c.td.forward(c.eng, c.cur);
    c.td.trace(sign);
    c.td.update(delta);
    return y;
  });
}
