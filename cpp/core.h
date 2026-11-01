// Bagammon core in C++: the rules of engine.js, the net of bot.js and the searches of bot.js/analysis.js,
// rebuilt for speed (the design of fast.js). Shared by the native trainer (train.cpp) and the browser
// module (wasm.cpp, compiled without any C/C++ library), so it uses no library either.
// check_cpp.js verifies end states, values, game flow, TD steps and searches against the JS reference.
#pragma once
#include <stdint.h>

namespace bg {

constexpr int HMAX = 256;            // hidden units at most; the code is compiled for each size in BG_SIZES
#define BG_SIZES(X) X(64) X(128) X(256)
#define BG_SIZE_OK(n) || h == n
inline bool hiddenOk(int h) { return false BG_SIZES(BG_SIZE_OK); }
#undef BG_SIZE_OK
constexpr int NIN_MAX = 246;          // 24 points * 5 thermometer units * 2 sides, then 6 scalars (every rule)
constexpr int STONES = 15, MAX_STACK = 5;
constexpr int MAXD = 4;               // dice per segment at most (Rene's doubles)
constexpr int BAR = 25, OFF = 0;      // relative 'from' of a bar entry, relative 'to' of a bear-off
constexpr int KEEP = 4;               // analysis.js: at depth 2+, only the best KEEP replies are searched further
enum Phase : uint8_t { ROLL, MOVE, WALK, OVER };
enum Rule : uint8_t { PAUL, RENE, MIXED };

// move encoding: from (rel 1..24, 25 = bar) | to (rel 1..24, 0 = off) << 5 | die << 10 | back << 13
inline int enc(int f, int t, int d, int b) { return f | (t << 5) | (d << 10) | (b << 13); }
inline int mFrom(int m) { return m & 31; }
inline int mTo(int m) { return (m >> 5) & 31; }
inline int mDie(int m) { return (m >> 10) & 7; }
inline int mBack(int m) { return (m >> 13) & 1; }

// ---------------------------------------------------------------- tables
// tanh by cubic Hermite interpolation in a table (max error ~1e-8 in double, float rounding on top);
// built with a series exp, since the browser module has no math library
constexpr int TSC = 64, TLIM = 10, TN = 2 * TLIM * TSC + 2;
inline float TT[TN], TDV[TN];
inline uint64_t ZP[25 * 11], ZS[4 * 16]; // Zobrist: [point][count + 5], [bar0, bar1, off0, off1][n]

inline double expSeries(double x) {
  int k = (int)(x * 1.4426950408889634 + (x >= 0 ? 0.5 : -0.5));
  double r = x - k * 0.6931471805599453, t = 1, s = 1;
  for (int i = 1; i < 25; i++) { t *= r / i; s += t; }
  for (; k > 0; k--) s *= 2;
  for (; k < 0; k++) s *= 0.5;
  return s;
}

inline void initTables() {
  for (int i = 0; i < TN; i++) {
    double x = (double)i / TSC - TLIM, e = expSeries(-2 * (x < 0 ? -x : x)), t = (1 - e) / (1 + e);
    if (x < 0) t = -t;
    TT[i] = (float)t;
    TDV[i] = (float)((1 - t * t) / TSC);
  }
  uint64_t s = 0x2545F4914F6CDD1Dull;
  auto next = [&]() { s += 0x9E3779B97F4A7C15ull; uint64_t z = s; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); };
  for (auto& z : ZP) z = next();
  for (auto& z : ZS) z = next();
}

inline float tanhT(float x) {
  float u = (x + TLIM) * TSC;
  if (u <= 0) return -1;
  if (u >= TN - 1) return 1;
  int i = (int)u;
  float t = u - i, t2 = t * t, t3 = t2 * t;
  return TT[i] * (2 * t3 - 3 * t2 + 1) + TDV[i] * (t3 - 2 * t2 + t) + TT[i + 1] * (3 * t2 - 2 * t3) + TDV[i + 1] * (t3 - t2);
}
inline float sigmoid(float z) { return 0.5f + 0.5f * tanhT(0.5f * z); }

// tanh for the hidden layer: 1 - 2 / (e^2x + 1) with e^2x = 2^k e^r, |r| <= ln2 / 2 (Cephes expf
// polynomial). Branch-free, so the loops over the hidden units vectorize (tanhT's table lookup does
// not; g++ needs -fno-trapping-math for the clamps). Absolute error ~1e-7.
inline float tanhV(float x) {
  x = x < -9.f ? -9.f : x; // tanh(9) = 1 - 3e-8
  x = x > 9.f ? 9.f : x;
  float u = 2.8853900817779268f * x;         // 2x / ln 2
  float k = (u + 12582912.f) - 12582912.f;   // rounded to an integer
  float r = 2.f * x - k * 0.693359375f + k * 2.12194440e-4f;
  float p = 1.9875691500e-4f;
  p = p * r + 1.3981999507e-3f; p = p * r + 8.3334519073e-3f; p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f; p = p * r + 5.0000001201e-1f;
  float e = (p * r * r + r + 1.f) * __builtin_bit_cast(float, ((int)k + 127) << 23);
  return 1.f - 2.f / (e + 1.f);
}

// b + sum of tanh(x[j]) * w[j], in 16 partial sums (a single running sum would not vectorize)
template <int H> inline float outSum(float b, const float* x, const float* w, float* t = nullptr) {
  static_assert(H % 16 == 0);
  float th[H], zs[16] = {};
  for (int j = 0; j < H; j++) th[j] = tanhV(x[j]);
  for (int j = 0; j < H; j += 16)
    for (int k = 0; k < 16; k++) zs[k] += th[j + k] * w[j + k];
  if (t) for (int j = 0; j < H; j++) t[j] = th[j];
  for (int k = 0; k < 16; k++) b += zs[k];
  return b;
}
// b + sum of x[j] * w[j] (no tanh), in 16 partial sums
template <int H> inline float dotH(float b, const float* x, const float* w) {
  float zs[16] = {};
  for (int j = 0; j < H; j += 16)
    for (int k = 0; k < 16; k++) zs[k] += x[j + k] * w[j + k];
  for (int k = 0; k < 16; k++) b += zs[k];
  return b;
}
// residual layer: out = in + u, u = tanh(W in + b), W is H rows of H numbers (input i's weights to all units).
// in and out may be the same array.
template <int H> inline void resLayer(const float* W, const float* b, const float* in, float* out, float* u) {
  float z[H];
  for (int j = 0; j < H; j++) z[j] = b[j];
  for (int i = 0; i < H; i++) { float v = in[i]; const float* w = W + i * H; for (int j = 0; j < H; j++) z[j] += v * w[j]; }
  for (int j = 0; j < H; j++) { u[j] = tanhV(z[j]); out[j] = in[j] + u[j]; }
}

// ---------------------------------------------------------------- net and position
// W1 is nin rows of h numbers (input i's weights to all hidden units), as in bot.js
// deep > 0: that many residual layers h += tanh(Wd h + bd) between the hidden layer and the output. With
// Wd = bd = 0 they are the identity, so a net deepened that way plays exactly like the one-layer net.
constexpr int DEEP_MAX = 2;
struct Net {
  int nin = 0, h = 0, deep = 0;
  float W1[NIN_MAX * HMAX], b1[HMAX], W2[HMAX], b2;
  float Wd[DEEP_MAX][HMAX * HMAX], bd[DEEP_MAX][HMAX];
};
// output sum b + W2 . h_last for the hidden sums x (before tanh)
template <int H> inline float netOut(const Net& n, float b, const float* x) {
  if (!n.deep) return outSum<H>(b, x, n.W2);
  float th[H], u[H];
  for (int j = 0; j < H; j++) th[j] = tanhV(x[j]);
  for (int k = 0; k < n.deep; k++) resLayer<H>(n.Wd[k], n.bd[k], th, th, u);
  return dotH<H>(b, th, n.W2);
}

// Everything that depends on the hidden size H (positions keep the hidden sums incrementally).
template <int H> struct Pos {
  int8_t b[25];                 // absolute board, + White, - Black (like engine.js)
  uint8_t bar[2], off[2];
  uint8_t cnt[7];               // remaining dice by face
  uint8_t turn, phase, t0, t1, pending, rolledDouble, limitHit;
  int8_t winner;
  int16_t pip[2], turnCount;
  uint64_t hash;                // board, bar and off
  alignas(16) float acc[2 * H]; // b1 + stone inputs, from White's (0..H-1) and Black's view
};

struct Rng {
  uint64_t s = 0x853C49E6748FEA9Bull;
  double next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return (double)((s * 0x2545F4914F6CDD1Dull) >> 11) * (1.0 / 9007199254740992.0); }
  int die() { return 1 + (int)(next() * 6); }
};

// open-addressing set of 64-bit keys, cleared in O(1) by a generation stamp
template <int LG> struct HashSetT {
  static constexpr int N = 1 << LG;
  uint64_t key[N];
  uint32_t stamp[N];
  uint32_t gen = 1;
  void clear() {
    if (++gen == 0) { for (auto& x : stamp) x = 0; gen = 1; }
  }
  bool insert(uint64_t k) { // false if already present
    uint32_t i = (uint32_t)(k ^ (k >> 29)) & (N - 1);
    for (;;) {
      if (stamp[i] != gen) { stamp[i] = gen; key[i] = k; return true; }
      if (key[i] == k) return false;
      i = (i + 1) & (N - 1);
    }
  }
};
using HashSet = HashSetT<15>;
using WalkSet = HashSetT<12>; // the ends of one forced walk: at most a few hundred

struct Cand { int path[MAXD]; int len; float q; };

// One per thread: rules, scratch space for the searches, statistics. H must equal the net's h.
template <int H> struct Engine {
  using Pos = bg::Pos<H>;
  int rule = MIXED, dm = 2, plyLimit = 0;
  bool mixed = true, rollAgain = false;
  const Net* net = nullptr;
  Rng rng;
  Pos fr[MAXD + 2];
  int mbuf[MAXD + 1][256];
  int path[MAXD];
  bool stop = false;
  HashSet seen, leaves;
  // Mixed: search of the opponent's forced walk, run from inside the enumeration of the mover's own segment
  Pos wfr[MAXD + 2];
  int wmbuf[MAXD + 1][256];
  WalkSet wseen, wleaves;
  float wbest = 0, wcut = -1;
  bool wstop = false;
  int moveDepth = 2; // Mixed, own throw: 2 = look at the opponent's forced walk of it, 1 = score the position
                     // as if they rolled next (cheaper; for pretraining only)
  static constexpr int MAXC = 1 << 14;
  Cand cands[MAXC];
  int ncand = 0;
  uint64_t nEval = 0;

  void setRule(int r, int limit, const Net* n) {
    rule = r; plyLimit = limit; net = n;
    mixed = r == MIXED; dm = r == RENE ? 4 : 2; rollAgain = r != MIXED;
  }

  // ---------------------------------------------------------------- position basics
  void reset(Pos& F) {
    for (auto& v : F.b) v = 0;
    F.bar[0] = F.bar[1] = F.off[0] = F.off[1] = 0;
    const int start[4][2] = { { 24, 2 }, { 13, 5 }, { 8, 3 }, { 6, 5 } };
    for (auto& rn : start) { F.b[rn[0]] += rn[1]; F.b[25 - rn[0]] -= rn[1]; }
    F.turn = 0; F.phase = ROLL; F.t0 = F.t1 = 0; F.pending = 0; F.rolledDouble = 0;
    F.winner = -1; F.turnCount = 0; F.limitHit = 0;
    zeroCnt(F);
    refresh(F);
  }
  static void zeroCnt(Pos& F) { for (auto& c : F.cnt) c = 0; }

  // recompute pips, hash and accumulators from scratch
  void refresh(Pos& F) const {
    for (int j = 0; j < H; j++) F.acc[j] = F.acc[H + j] = net->b1[j];
    F.pip[0] = 25 * F.bar[0]; F.pip[1] = 25 * F.bar[1];
    uint64_t h = 0;
    for (int a = 1; a <= 24; a++) {
      int v = F.b[a];
      h ^= ZP[a * 11 + v + 5];
      if (v > 0) { F.pip[0] += a * v; for (int k = 0; k < v; k++) stone(F.acc, 0, a, k, 1); }
      if (v < 0) { F.pip[1] += (25 - a) * -v; for (int k = 0; k < -v; k++) stone(F.acc, 1, a, k, 1); }
    }
    for (int i = 0; i < 4; i++) h ^= ZS[i * 16 + (i < 2 ? F.bar[i] : F.off[i - 2])];
    F.hash = h;
  }

  // thermometer unit k of player x's stack on absolute point a switched on (sign 1) or off (-1), both views.
  // As in bot.js, a stack sits at its owner's relative point in either view; only own/opponent block differs.
  void stone(float* acc, int x, int a, int k, int sign) const {
    int r = (x == 0 ? a : 25 - a) - 1;
    const float* w0 = net->W1 + ((x == 0 ? 0 : 120) + r * 5 + k) * H; // White's view
    const float* w1 = net->W1 + ((x == 1 ? 0 : 120) + r * 5 + k) * H; // Black's view
    if (sign > 0) for (int j = 0; j < H; j++) { acc[j] += w0[j]; acc[H + j] += w1[j]; }
    else for (int j = 0; j < H; j++) { acc[j] -= w0[j]; acc[H + j] -= w1[j]; }
  }

  static void setPt(Pos& F, int a, int nv) { F.hash ^= ZP[a * 11 + F.b[a] + 5] ^ ZP[a * 11 + nv + 5]; F.b[a] = (int8_t)nv; }
  static void setSide(Pos& F, int i, uint8_t* arr, int j, int nv) { F.hash ^= ZS[i * 16 + arr[j]] ^ ZS[i * 16 + nv]; arr[j] = (uint8_t)nv; }

  // apply move m of player p (no dice bookkeeping, no turn logic)
  void make(Pos& F, int p, int m) const {
    int f = mFrom(m), t = mTo(m), sg = p == 0 ? 1 : -1, q = 1 - p;
    if (f == BAR) {
      setSide(F, p, F.bar, p, F.bar[p] - 1);
      F.pip[p] -= 25;
    } else {
      int a = p == 0 ? f : 25 - f, c = F.b[a] * sg;
      setPt(F, a, F.b[a] - sg);
      stone(F.acc, p, a, c - 1, -1);
      F.pip[p] -= f;
    }
    if (t == OFF) {
      setSide(F, 2 + p, F.off, p, F.off[p] + 1);
    } else {
      int a = p == 0 ? t : 25 - t;
      if (F.b[a] * sg == -1) {
        setPt(F, a, 0);
        stone(F.acc, q, a, 0, -1);
        setSide(F, q, F.bar, q, F.bar[q] + 1);
        F.pip[q] += 25 - (q == 0 ? a : 25 - a);
      }
      int c = F.b[a] * sg;
      setPt(F, a, F.b[a] + sg);
      stone(F.acc, p, a, c, 1);
      F.pip[p] += t;
    }
  }

  // ---------------------------------------------------------------- rules (mirrors engine.js)
  static int own(const Pos& F, int p, int r) { int v = F.b[p == 0 ? r : 25 - r]; return p == 0 ? (v > 0 ? v : 0) : (v < 0 ? -v : 0); }
  static bool canLand(const Pos& F, int p, int r) { int v = F.b[p == 0 ? r : 25 - r] * (p == 0 ? 1 : -1); return v > -2 && v < MAX_STACK; }
  static int highest(const Pos& F, int p) { for (int r = 24; r >= 1; r--) if (own(F, p, r)) return r; return 0; }
  static bool entryBlocked(const Pos& F, int p) { for (int d = 1; d <= 6; d++) if (canLand(F, p, 25 - d)) return false; return true; }

  // legal single-die moves of p into out[n..]; returns the new n
  int gen(const Pos& F, int p, int d, int* out, int n) const {
    if (F.bar[p] > 0) {
      if (canLand(F, p, 25 - d)) out[n++] = enc(BAR, 25 - d, d, 0);
      return n;
    }
    int top = highest(F, p);
    bool home = top <= 6;
    // Mixed: bearing off is mandatory when the die equals the point of one of your stones
    if (mixed && home && own(F, p, d)) { out[n++] = enc(d, OFF, d, 0); return n; }
    // Mixed: backward moves while no stone is off and no stone is benched
    bool canBack = mixed && F.off[0] == 0 && F.off[1] == 0 && F.bar[1 - p] == 0;
    for (int r = 1; r <= top; r++) {
      if (!own(F, p, r)) continue;
      int t = r - d;
      if (t >= 1) { if (canLand(F, p, t)) out[n++] = enc(r, t, d, 0); }
      else if (home && (t == 0 || r == top)) out[n++] = enc(r, OFF, d, 0);
      if (canBack && r > 6 && r + d <= 24 && canLand(F, p, r + d)) out[n++] = enc(r, r + d, d, 1);
    }
    return n;
  }
  int genAll(const Pos& F, int p, int* out) const {
    int n = 0;
    for (int d = 1; d <= 6; d++) if (F.cnt[d]) n = gen(F, p, d, out, n);
    return n;
  }
  bool anyMove(const Pos& F, int p) const {
    int out[64];
    for (int d = 1; d <= 6; d++) if (F.cnt[d] && gen(F, p, d, out, 0)) return true;
    return false;
  }
  void setDice(Pos& F, int a, int b) const {
    zeroCnt(F);
    if (a == b) F.cnt[a] = (uint8_t)dm; else { F.cnt[a]++; F.cnt[b]++; }
  }

  // ---------------------------------------------------------------- game flow
  void roll(Pos& s, int a, int b) const {
    s.t0 = (uint8_t)a; s.t1 = (uint8_t)b; s.rolledDouble = a == b;
    if (mixed) s.pending = 1;
    setDice(s, a, b);
    s.phase = MOVE;
    settle(s);
  }
  void applyMove(Pos& s, int m) const {
    s.cnt[mDie(m)]--;
    make(s, s.turn, m);
    settle(s);
  }
  void settle(Pos& s) const {
    int p = s.turn;
    if (s.off[p] == STONES) { s.winner = (int8_t)p; s.phase = OVER; zeroCnt(s); return; }
    if (!anyMove(s, p)) { zeroCnt(s); finishSegment(s); }
  }
  void finishSegment(Pos& s) const {
    if (s.phase == WALK) { s.phase = ROLL; return; }
    if (rollAgain && s.rolledDouble) { s.phase = ROLL; return; } // the same player must roll again
    nextTurn(s);
  }
  void nextTurn(Pos& s) const {
    int q = 1 - s.turn;
    if (plyLimit && ++s.turnCount >= plyLimit) {
      int a = s.pip[0], b = s.pip[1];
      s.winner = (int8_t)(a < b ? 0 : a > b ? 1 : s.turn); // equal: the side that just moved
      s.phase = OVER; zeroCnt(s); s.limitHit = 1;
      return;
    }
    s.turn = (uint8_t)q; s.rolledDouble = 0; zeroCnt(s); s.phase = ROLL;
    if (!mixed) return;
    // benched with no opening: q does not throw, the opponent throws again
    if (s.bar[q] > 0 && entryBlocked(s, q)) { s.turn = (uint8_t)(1 - q); s.pending = 0; return; }
    if (s.pending) { // q walks the same throw
      setDice(s, s.t0, s.t1);
      s.pending = 0; s.phase = WALK;
      settle(s);
    }
  }

  // ---------------------------------------------------------------- evaluation
  // net output for F seen by side P: P(P wins) with P to roll and nobody holding dice
  float evalPos(const Pos& F, int P) {
    nEval++;
    const float* W1 = net->W1;
    const float* acc = F.acc + P * H;
    const float* B = W1 + 240 * H;
    int Q = 1 - P;
    float e0 = F.bar[P] / 5.f, e1 = F.bar[Q] / 5.f, e2 = F.off[P] / 15.f, e3 = F.off[Q] / 15.f, e4 = F.pip[P] / 200.f, e5 = F.pip[Q] / 200.f;
    float x[H];
    for (int j = 0; j < H; j++)
      x[j] = acc[j] + e0 * B[j] + e1 * B[H + j] + e2 * B[2 * H + j] + e3 * B[3 * H + j] + e4 * B[4 * H + j] + e5 * B[5 * H + j];
    return sigmoid(netOut<H>(*net, net->b2 + (mixed ? (F.pip[Q] - F.pip[P]) / 60.f : 0.f), x));
  }

  // bot.js valueFor: P(player p wins) for a position where the roller is s.turn
  float valueFor(const Pos& s, int p) {
    if (s.phase == OVER) return s.winner == p ? 1.f : 0.f;
    float y = evalPos(s, s.turn);
    return s.turn == p ? y : 1 - y;
  }

  // value for mover p of C, a finished segment (phase MOVE or WALK): runs the segment-end transition
  // of engine.js virtually (no copy), then scores the resulting position like valueFor. Mixed, after
  // the mover's own throw: the opponent's forced walk of it is searched (walkSearch), unless moveDepth is 1.
  // cut: the caller only cares whether the value beats cut (P(p) > cut); a walk that already sinks to cut
  // stops there, and the result is then an upper bound <= cut instead of the exact value.
  float leafValue(const Pos& C, int p, float cut = -1.f) {
    if (C.off[p] == STONES) return 1;
    if (C.phase == WALK || (rollAgain && C.rolledDouble)) return evalPos(C, p);
    if (plyLimit && C.turnCount + 1 >= plyLimit) {
      int a = C.pip[0], b = C.pip[1];
      return (a < b ? 0 : a > b ? 1 : p) == p ? 1.f : 0.f;
    }
    int q = 1 - p;
    if (!mixed) return 1 - evalPos(C, q);
    if (C.bar[q] > 0 && entryBlocked(C, q)) return evalPos(C, p);
    if (moveDepth >= 2) {
      wfr[0] = C;
      setDice(wfr[0], C.t0, C.t1); // q walks the same throw
      float v = walkSearch(q, p, cut);
      if (v >= 0) return v;
    }
    return 1 - evalPos(C, q); // q rolls at once (no legal walk, or depth 1)
  }

  // Mixed: the walker w must play out the dice in wfr[0] (a WALK position). The dice are known, so there is
  // no chance node: w takes the end state that is best for w, i.e. worst for p if w is the opponent.
  // Returns P(p wins), or -1 if w has no legal move at all.
  // cut: stop as soon as the walker's best is at or below cut (w is the opponent) / at or above it (w is p).
  float walkSearch(int w, int p, float cut = -1.f) {
    wseen.clear(); wleaves.clear();
    wbest = w == p ? 0.f : 1.f;
    wcut = w == p ? 2.f : cut;
    wstop = false;
    return wdfs(0, w, p) ? wbest : -1.f;
  }
  bool wdfs(int d, int w, int p) {
    Pos& F = wfr[d];
    int* out = wmbuf[d];
    int n = genAll(F, w, out);
    if (n == 0) return false;
    Pos& C = wfr[d + 1];
    for (int i = 0; i < n; i++) {
      int m = out[i];
      C = F;
      C.cnt[mDie(m)]--;
      make(C, w, m);
      bool leaf = C.off[w] == STONES;
      if (!leaf) {
        uint64_t code = 0;
        int left = 0;
        for (int k = 1; k <= 6; k++) { code = code * 5 + C.cnt[k]; left += C.cnt[k]; }
        if (left == 0) leaf = true;
        else {
          if (!wseen.insert(C.hash ^ (code * 0x9E3779B97F4A7C15ull))) continue;
          if (wdfs(d + 1, w, p)) { if (wstop) return true; continue; }
          leaf = true; // no remaining die can be played
        }
      }
      if (!wleaves.insert(C.hash)) continue;
      float v = C.off[w] == STONES ? (w == p ? 1.f : 0.f) : (w == p ? evalPos(C, w) : 1 - evalPos(C, w)); // w rolls next
      if (w == p ? v > wbest : v < wbest) {
        wbest = v;
        if (w == p ? wbest >= wcut : wbest <= wcut) { wstop = true; return true; }
      }
    }
    return true;
  }
  // value for p of s, the position before a walk (phase WALK) or of any other position
  float nodeLeaf(const Pos& s, int p) {
    if (mixed && s.phase == WALK && moveDepth >= 2) {
      wfr[0] = s;
      float v = walkSearch(s.turn, p);
      if (v >= 0) return v;
    }
    return valueFor(s, p);
  }

  // ---------------------------------------------------------------- segment enumeration
  // Calls cb(C, len) once for every distinct end of the current MOVE/WALK segment of s (C: the position
  // after the moves path[0..len), before the segment-end transition). cb returns false to stop.
  template <class CB> void enumerate(const Pos& s, CB&& cb) {
    fr[0] = s;
    seen.clear(); leaves.clear(); stop = false;
    dfs(0, s.turn, cb);
  }
  template <class CB> bool dfs(int d, int p, CB& cb) {
    Pos& F = fr[d];
    int* out = mbuf[d];
    int n = genAll(F, p, out);
    if (n == 0) return false;
    Pos& C = fr[d + 1];
    for (int i = 0; i < n && !stop; i++) {
      int m = out[i];
      C = F;
      C.cnt[mDie(m)]--;
      make(C, p, m);
      path[d] = m;
      bool leaf = C.off[p] == STONES;
      if (!leaf) {
        uint64_t code = 0;
        int left = 0;
        for (int k = 1; k <= 6; k++) { code = code * 5 + C.cnt[k]; left += C.cnt[k]; }
        if (left == 0) leaf = true;
        else {
          if (!seen.insert(C.hash ^ (code * 0x9E3779B97F4A7C15ull))) continue;
          if (dfs(d + 1, p, cb)) continue;
          leaf = true; // no remaining die can be played
        }
      }
      if (!leaves.insert(C.hash)) continue;
      if (!cb(C, d + 1)) stop = true;
    }
    return true;
  }

  // the end state of a segment: s with the moves of a candidate played and the segment-end transition run
  void applyPath(Pos& c, const int* pth, int len) const {
    for (int k = 0; k < len; k++) { c.cnt[mDie(pth[k])]--; make(c, c.turn, pth[k]); }
  }
  void playPath(Pos& c, const int* pth, int len) const { applyPath(c, pth, len); settle(c); }

  // Mixed, own throw: scoring an end means a whole walk search, so the ends are visited best-first by their
  // cheap value (the position scored without the walk search). That makes the cutoffs inside leafValue bite
  // early; with survive > 0 only the first `survive` ends are scored at all (an approximation).
  int survive = 0;
  Cand rk[MAXC];
  int ord[MAXC], nrk = 0;
  template <class CB> void forEnds(const Pos& s, CB&& cb) {
    if (!(mixed && moveDepth >= 2 && s.phase == MOVE)) { enumerate(s, cb); return; }
    int p = s.turn, md = moveDepth;
    moveDepth = 1;
    nrk = 0;
    enumerate(s, [&](Pos& C, int len) {
      if (nrk == MAXC) return false;
      Cand& c = rk[nrk];
      for (int k = 0; k < len; k++) c.path[k] = path[k];
      c.len = len;
      c.q = leafValue(C, p);
      int j = nrk++;
      for (; j > 0 && rk[ord[j - 1]].q < c.q; j--) ord[j] = ord[j - 1];
      ord[j] = nrk - 1;
      return true;
    });
    moveDepth = md;
    int n = survive > 0 && survive < nrk ? survive : nrk;
    for (int i = 0; i < n; i++) {
      const Cand& c = rk[ord[i]];
      Pos C = s;
      applyPath(C, c.path, c.len);
      for (int k = 0; k < c.len; k++) path[k] = c.path[k];
      if (!cb(C, c.len)) break;
    }
  }

  // all ends of the segment of s with their 1-ply value for s.turn into cands, best first
  int rootCands(const Pos& s) {
    int p = s.turn;
    ncand = 0;
    enumerate(s, [&](Pos& C, int len) {
      if (ncand == MAXC) return false;
      Cand& c = cands[ncand++];
      for (int k = 0; k < len; k++) c.path[k] = path[k];
      c.len = len;
      c.q = leafValue(C, p);
      return true;
    });
    // insertion sort, stable (few hundred candidates)
    for (int i = 1; i < ncand; i++) {
      Cand c = cands[i];
      int j = i;
      for (; j > 0 && cands[j - 1].q < c.q; j--) cands[j] = cands[j - 1];
      cands[j] = c;
    }
    return ncand;
  }

  // ---------------------------------------------------------------- 1-ply play (bot.js bestPlan)
  int bestLen = 0, bestPath[MAXD];
  float best1(Pos& s) {
    refresh(s); // the incrementally kept accumulators are stale once training has changed the weights
    int p = s.turn;
    float best = -1;
    bestLen = 0;
    forEnds(s, [&](Pos& C, int len) {
      float v = leafValue(C, p, best);
      if (v > best) { best = v; bestLen = len; for (int k = 0; k < len; k++) bestPath[k] = path[k]; }
      return true;
    });
    return best;
  }
  void playBest(Pos& s, int* moves) {
    best1(s);
    for (int k = 0; k < bestLen && (s.phase == MOVE || s.phase == WALK); k++) { applyMove(s, bestPath[k]); if (moves) ++*moves; }
  }

  // softmax play, for exploration in training: each end of the segment is taken with probability
  // ~ exp((v - best) / temp), v its value as in best1. Sampled as the argmax of v + temp * Gumbel noise
  // (exact), so each end only needs scoring up to its own cut, as in best1. softBest: the best end's value
  // for the mover, softOff: an end worse than the best was taken; both are only filled in when needBest
  // is set (each end is then also scored up to the best plain value so far).
  float temp = 0, softBest = 0;
  bool softOff = false, needBest = false;
  static double lnT(double x) { // natural log, x > 0 (no libm here)
    int e = 0;
    while (x >= 2) { x *= 0.5; e++; }
    while (x < 1) { x *= 2; e--; }
    double z = (x - 1) / (x + 1), z2 = z * z, s = 0, zk = z;
    for (int k = 1; k <= 19; k += 2) { s += zk / k; zk *= z2; }
    return e * 0.6931471805599453 + 2 * s;
  }
  void playSoft(Pos& s, int* moves) {
    softOff = false;
    if (temp <= 0) { playBest(s, moves); softBest = 0; return; }
    refresh(s);
    int p = s.turn;
    float top = -1e30f, val = 0, best = -1; // top: best perturbed value so far, val: its plain value
    bestLen = 0;
    forEnds(s, [&](Pos& C, int len) {
      double u = rng.next();
      float g = (float)(-lnT(-lnT(u > 0 ? u : 1e-300))), cut = top - temp * g;
      float v = leafValue(C, p, needBest && best < cut ? best : cut);
      if (v > best) best = v;
      if (v > cut) { top = v + temp * g; val = v; bestLen = len; for (int k = 0; k < len; k++) bestPath[k] = path[k]; }
      return true;
    });
    if (needBest) { softBest = best; softOff = best > val; }
    for (int k = 0; k < bestLen && (s.phase == MOVE || s.phase == WALK); k++) { applyMove(s, bestPath[k]); if (moves) ++*moves; }
  }

  // greedy baseline of bot.js (hit bonus 0): single moves until the turn passes or the segment ends
  void playGreedy(Pos& s, int* moves) {
    int p = s.turn, out[256];
    while (s.phase == MOVE || s.phase == WALK) {
      int n = genAll(s, p, out), bm = 0;
      double bv = -1e9;
      for (int i = 0; i < n; i++) {
        int m = out[i];
        double v = (mBack(m) ? -mDie(m) : mDie(m)) + (mTo(m) == OFF ? 6 : 0) + rng.next() * 0.1;
        if (v > bv) { bv = v; bm = m; }
      }
      applyMove(s, bm);
      if (moves) ++*moves;
      if (s.turn != p) break;
    }
  }
  void playRandom(Pos& s, int* moves) {
    int p = s.turn, out[256];
    while (s.phase == MOVE || s.phase == WALK) {
      int n = genAll(s, p, out);
      applyMove(s, out[(int)(rng.next() * n)]);
      if (moves) ++*moves;
      if (s.turn != p) break;
    }
  }

  // ---------------------------------------------------------------- multi-ply search (analysis.js evalState)
  // Expectimax over the dice: P(player p wins) after searching d more rolls. With the full window
  // (alpha = -1, beta = 2) the value is exact. With a narrower window this is alpha-beta for dice games
  // (Ballard's Star1): decision nodes cut as usual; a chance node cuts once the rolls seen so far, with
  // the unseen ones bounded by 0 and 1, put its average outside the window. Fail-soft: a result <= alpha
  // is an upper bound, a result >= beta a lower bound, anything in between exact.
  bool ab = true; // false: always the full window inside the tree (no cutoffs), for comparisons
  float node(const Pos& s, int p, int d, float alpha, float beta) {
    if (!ab) { alpha = -1; beta = 2; }
    if (s.phase == OVER) return s.winner == p ? 1.f : 0.f;
    if (d <= 1 && s.phase != MOVE) return nodeLeaf(s, p); // MOVE = just rolled: still needs its best reply
    if (s.phase == ROLL) {
      float sum = 0;
      int seen36 = 0;
      for (int a = 1; a <= 6; a++) for (int b = a; b <= 6; b++) {
        int w36 = a == b ? 1 : 2;
        seen36 += w36;
        float w = w36 / 36.f, rest = (36 - seen36) / 36.f;
        Pos c = s;
        roll(c, a, b);
        float lo = (alpha - sum - rest) / w, hi = (beta - sum) / w;
        float v = node(c, p, d - 1, lo, hi);
        sum += w * v;
        if (v <= lo) return sum + rest; // even if all other rolls score 1: <= alpha
        if (v >= hi) return sum;        // even if all other rolls score 0: >= beta
      }
      return sum;
    }
    // decision node: the side to move picks the end state that is best for itself
    bool mine = s.turn == p;
    int mover = s.turn;
    if (d <= 1) { // its ends are leaves: score them while enumerating, stop at a cutoff
      float best = mine ? -1.f : 2.f;
      forEnds(s, [&](Pos& C, int) {
        // the mover wants the highest value for itself: a candidate whose walk sinks to the best so far (or out of the window) is dropped
        float v = leafValue(C, mover, mine ? (best > alpha ? best : alpha) : (best < beta ? 1 - best : 1 - beta));
        if (!mine) v = 1 - v;
        if (mine ? v > best : v < best) best = v;
        return mine ? best < beta : best > alpha;
      });
      return best;
    }
    // keep the KEEP best ends by 1-ply value (all of them if there are no more), search those deeper
    Cand top[KEEP];
    int nt = 0;
    forEnds(s, [&](Pos& C, int len) {
      float q = leafValue(C, mover, nt < KEEP ? -1.f : mine ? top[KEEP - 1].q : 1 - top[KEEP - 1].q);
      if (!mine) q = 1 - q;
      int i = nt;
      while (i > 0 && (mine ? q > top[i - 1].q : q < top[i - 1].q)) i--;
      if (i < KEEP) {
        for (int k = (nt < KEEP ? nt : KEEP - 1); k > i; k--) top[k] = top[k - 1];
        for (int k = 0; k < len; k++) top[i].path[k] = path[k];
        top[i].len = len; top[i].q = q;
        if (nt < KEEP) nt++;
      }
      return true;
    });
    float best = mine ? -1.f : 2.f;
    for (int i = 0; i < nt; i++) {
      Pos c = s;
      playPath(c, top[i].path, top[i].len);
      float v = node(c, p, d, alpha, beta);
      if (mine) { if (v > best) best = v; if (best > alpha) alpha = best; if (alpha >= beta) break; }
      else { if (v < best) best = v; if (best < beta) beta = best; if (alpha >= beta) break; }
    }
    return best;
  }

  // Best plan for the side to move at the given depth (the argmax of an analysis at that depth).
  // prune: alpha-beta across the candidates (exact best move and value, the rest are only bounds).
  // Fills bestPath/bestLen, returns the value.
  float bestPlan(const Pos& s0, int depth, bool prune) {
    Pos s = s0;
    refresh(s);
    int p = s.turn, n = rootCands(s);
    bestLen = 0;
    if (n == 0) return -1;
    float best = -1;
    int bi = 0;
    for (int i = 0; i < n; i++) {
      if (depth <= 1) break;
      Pos c = s;
      playPath(c, cands[i].path, cands[i].len);
      float v = node(c, p, depth, prune ? best : -1.f, 2.f);
      if (v > best) { best = v; bi = i; }
    }
    if (depth <= 1) best = cands[0].q;
    bestLen = cands[bi].len;
    for (int k = 0; k < bestLen; k++) bestPath[k] = cands[bi].path[k];
    return best;
  }
};

// ---------------------------------------------------------------- TD(lambda) trainer state
// Scaled eligibility trace (E_true = c * E), so the per-step decay is O(1). The update only visits the
// W1 rows whose input has been active in this game (all other rows have a zero trace), plus b1, W2, b2 and the
// residual layers (dense).
template <int H> struct TD {
  using Pos = bg::Pos<H>;
  using Engine = bg::Engine<H>;
  Net* net = nullptr;
  double lambda = 0.7, c = 1, y = 0.5;
  double E[NIN_MAX * H + 2 * H + 1 + DEEP_MAX * (H * H + H)];
  uint8_t touched[NIN_MAX];
  int rows[NIN_MAX], nrows = 0;
  float h[H];                             // tanh of the hidden layer
  float hl[DEEP_MAX + 1][H], ul[DEEP_MAX][H]; // deep nets: hl[0] = h, hl[k + 1] = hl[k] + ul[k] (residual layer k)
  int idx[64], nf = 0;
  float val[64], bias = 0;

  void init(Net* n, double lam) {
    net = n; lambda = lam;
    for (auto& e : E) e = 0;
    for (auto& t : touched) t = 0;
    nrows = 0; c = 1;
  }
  int eB1() const { return net->nin * H; }
  int eD(int k) const { return eB1() + 2 * H + 1 + k * (H * H + H); } // residual layer k: W, then b
  void clear() {
    for (int r = 0; r < nrows; r++) { touched[rows[r]] = 0; for (int j = 0; j < H; j++) E[rows[r] * H + j] = 0; }
    for (int i = eB1(); i < eD(net->deep); i++) E[i] = 0;
    nrows = 0; c = 1;
  }
  // sparse inputs of s seen by the side to roll (bot.js features)
  void features(const Engine& g, const Pos& s) {
    int P = s.turn, Q = 1 - P, n = 0;
    for (int r = 1; r <= 24; r++) {
      int a = Engine::own(s, P, r), b = Engine::own(s, Q, r);
      for (int k = 0; k < a; k++) { idx[n] = (r - 1) * 5 + k; val[n++] = 1; }
      for (int k = 0; k < b; k++) { idx[n] = 120 + (r - 1) * 5 + k; val[n++] = 1; }
    }
    float extra[6] = { s.bar[P] / 5.f, s.bar[Q] / 5.f, s.off[P] / 15.f, s.off[Q] / 15.f, s.pip[P] / 200.f, s.pip[Q] / 200.f };
    for (int k = 0; k < 6; k++) { idx[n] = 240 + k; val[n++] = extra[k]; }
    nf = n;
    bias = g.mixed ? (s.pip[Q] - s.pip[P]) / 60.f : 0.f;
  }
  // P(side to roll wins) in s; leaves the features and hidden activations for trace()
  double forward(const Engine& g, const Pos& s) {
    features(g, s);
    const float* W1 = net->W1;
    for (int j = 0; j < H; j++) h[j] = net->b1[j];
    for (int t = 0; t < nf; t++) { const float* w = W1 + idx[t] * H; float v = val[t]; for (int j = 0; j < H; j++) h[j] += v * w[j]; }
    if (!net->deep) return y = sigmoid(outSum<H>(net->b2 + bias, h, net->W2, h));
    for (int j = 0; j < H; j++) hl[0][j] = h[j] = tanhV(h[j]);
    for (int k = 0; k < net->deep; k++) resLayer<H>(net->Wd[k], net->bd[k], hl[k], hl[k + 1], ul[k]);
    return y = sigmoid(dotH<H>(net->b2 + bias, hl[net->deep], net->W2));
  }
  // E = lambda * E + sign * grad(output), right after forward()
  void trace(int sign) {
    if (lambda == 0) clear(); else c *= lambda;
    if (c < 1e-150) { for (auto& e : E) e *= c; c = 1; }
    for (int t = 0; t < nf; t++) { int r = idx[t]; if (!touched[r]) { touched[r] = 1; rows[nrows++] = r; } }
    double s = sign * y * (1 - y) / c, dh[H];
    int oB1 = eB1(), oW2 = oB1 + H, oB2 = oW2 + H, D = net->deep;
    E[oB2] += s;
    if (!D) {
      for (int j = 0; j < H; j++) {
        E[oW2 + j] += s * h[j];
        dh[j] = s * net->W2[j] * (1 - h[j] * h[j]);
        E[oB1 + j] += dh[j];
      }
    } else { // back through the residual layers; dh = gradient by the current layer's input
      for (int j = 0; j < H; j++) { E[oW2 + j] += s * hl[D][j]; dh[j] = s * net->W2[j]; }
      for (int k = D - 1; k >= 0; k--) {
        double dz[H], *Ew = E + eD(k), *Eb = Ew + H * H;
        for (int j = 0; j < H; j++) { dz[j] = dh[j] * (1 - ul[k][j] * ul[k][j]); Eb[j] += dz[j]; }
        for (int i = 0; i < H; i++) {
          double v = hl[k][i], g = 0, *e = Ew + i * H;
          const float* w = net->Wd[k] + i * H;
          for (int j = 0; j < H; j++) { e[j] += v * dz[j]; g += w[j] * dz[j]; }
          dh[i] += g;
        }
      }
      for (int j = 0; j < H; j++) { dh[j] *= 1 - h[j] * h[j]; E[oB1 + j] += dh[j]; }
    }
    for (int t = 0; t < nf; t++) { double* e = E + idx[t] * H; double v = val[t]; for (int j = 0; j < H; j++) e[j] += v * dh[j]; }
  }
  // w += delta * E_true
  void update(double delta) {
    double k = delta * c;
    float* W1 = net->W1;
    for (int r = 0; r < nrows; r++) { int o = rows[r] * H; for (int j = o; j < o + H; j++) W1[j] += (float)(k * E[j]); }
    const double* e = E + eB1();
    for (int j = 0; j < H; j++) net->b1[j] += (float)(k * e[j]);
    for (int j = 0; j < H; j++) net->W2[j] += (float)(k * e[H + j]);
    net->b2 += (float)(k * e[2 * H]);
    for (int l = 0; l < net->deep; l++) {
      const double* d = E + eD(l);
      float *W = net->Wd[l], *b = net->bd[l];
      for (int i = 0; i < H * H; i++) W[i] += (float)(k * d[i]);
      for (int j = 0; j < H; j++) b[j] += (float)(k * d[H * H + j]);
    }
  }
};

} // namespace bg
