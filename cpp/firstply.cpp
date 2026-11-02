// P(White wins) at the start of a Mixed game (White to roll): the net's value, the d-ply expectimax, and
// rollouts by the 1-ply bot on both sides (LIMIT 200, a game cut there is a draw worth 1/2).
//   [TIME=s] [VR=1] ./firstply net.json depth games [threads]      (depth 0 = net value only; games 0 = no rollouts)
// Build: g++ -std=c++20 -O3 -march=native -fno-trapping-math -pthread -o firstply firstply.cpp (in cpp/)
// Rollouts are stratified by the first throw (game i starts with ordered throw i % 36). TIME stops them cleanly.
// VR=1: luck-based variance reduction (subtract each throw's 1-ply value minus the 36-throw mean); unbiased, but
// it cut the variance only ~3.5x at ~25x the cost per game, so it does not pay.
// "depth" is node()'s: 1 and 2 both = White's best move averaged over the throws (Black's walk searched), 3 = also Black's next throw.
#include "core.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
using namespace bg;
static std::vector<double> nums(const std::string& t, const char* key) {
  std::vector<double> out; size_t i = t.find(std::string("\"") + key + "\"");
  if (i == std::string::npos) return out;
  i = t.find(':', i) + 1; bool arr = t[i] == '['; if (arr) i++;
  const char* p = t.c_str() + i;
  for (;;) { while (*p == ' ' || *p == ',' || *p == '\n') p++; if (*p == ']' || !*p) break;
    char* e; out.push_back(strtod(p, &e)); if (e == p || !arr) break; p = e; }
  return out;
}
static Net* load(const char* f) {
  std::ifstream in(f); std::stringstream ss; ss << in.rdbuf(); std::string t = ss.str();
  auto W1 = nums(t, "W1"), b1 = nums(t, "b1"), W2 = nums(t, "W2"), b2 = nums(t, "b2");
  Net* n = new Net(); n->h = b1.size(); n->nin = W1.size() / n->h;
  for (size_t i = 0; i < W1.size(); i++) n->W1[i] = W1[i];
  for (int j = 0; j < n->h; j++) { n->b1[j] = b1[j]; n->W2[j] = W2[j]; }
  n->b2 = b2[0];
  auto Wd = nums(t, "Wd"), bd = nums(t, "bd"); n->deep = bd.size() / n->h;
  for (int k = 0; k < n->deep; k++) { for (int i = 0; i < n->h * n->h; i++) n->Wd[k][i] = Wd[k * n->h * n->h + i];
    for (int j = 0; j < n->h; j++) n->bd[k][j] = bd[k * n->h + j]; }
  return n;
}
static const bool VR = getenv("VR") && atoi(getenv("VR"));
static const double TSEC = getenv("TIME") ? atof(getenv("TIME")) : 0;
static uint64_t mix(uint64_t z) { z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
template <int H> void run(Net* N, int depth, long games, int threads) {
  auto e = std::make_unique<Engine<H>>(); e->setRule(MIXED, 200, N); e->limitDraw = true; e->ab = false;
  Pos<H> s; e->reset(s);
  printf("net value of the start (White to roll): %.5f\n", e->valueFor(s, 0));
  for (int d = 1; d <= depth; d++) {
    auto t0 = std::chrono::steady_clock::now();
    double sum = 0;
    if (d == 1) { // 1-ply: average over the 36 throws of White's best end (Black's forced walk searched)
      for (int a = 1; a <= 6; a++) for (int b = a; b <= 6; b++) { Pos<H> c = s; e->roll(c, a, b); float v = e->best1(c);
        sum += (a == b ? 1 : 2) / 36.0 * v; printf("  %d-%d %.4f", a, b, v); if (b == 6) printf("\n"); }
    } else sum = e->node(s, 0, d, -1, 2);
    printf("depth %d: %.5f  (%.1f s)\n", d, sum, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    fflush(stdout);
  }
  if (!games) return;
  // rollouts, stratified by the first throw: game i has first throw i % 36 (ordered), the rest random
  std::atomic<long> next{0};
  std::vector<long> N_(threads); std::vector<double> S(threads), SS(threads), W(threads), D(threads);
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> ts;
  for (int t = 0; t < threads; t++) ts.emplace_back([&, t]() {
    auto g = std::make_unique<Engine<H>>(); g->setRule(MIXED, 200, N); g->limitDraw = true;
    const long B = 1000;
    for (long i0; (i0 = next.fetch_add(B)) < games;) {
      if (TSEC && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > TSEC) break;
      for (long i = i0; i < i0 + B && i < games; i++) {
        Pos<H> p; g->reset(p); Rng r; r.s = mix(i * 0x9E3779B97F4A7C15ull + 12345) | 1;
        int steps = 0; double luck = 0; bool first = true;
        while (p.phase != OVER) {
          if (p.phase != ROLL) { g->playBest(p, &steps); continue; }
          int a, b;
          if (first) { int k = i % 36; a = 1 + k / 6; b = 1 + k % 6; first = false; } else { a = r.die(); b = r.die(); }
          if (VR) { // luck of this throw for White: its 1-ply value minus the mean over all 36 throws
            int m = p.turn; double E = 0, va = 0;
            for (int x = 1; x <= 6; x++) for (int y = x; y <= 6; y++) {
              Pos<H> c = p; g->roll(c, x, y);
              double v = c.phase == MOVE && c.turn == m ? g->best1(c) : g->nodeLeaf(c, m); // else: no move possible
              if (m == 1) v = 1 - v;
              E += (x == y ? 1 : 2) / 36.0 * v;
              if ((x == a && y == b) || (x == b && y == a)) va = v;
            }
            luck += va - E;
          }
          g->roll(p, a, b);
        }
        double v = (p.winner < 0 ? 0.5 : p.winner == 0 ? 1 : 0) - luck;
        N_[t]++; S[t] += v; SS[t] += v * v; W[t] += p.winner == 0; D[t] += p.winner < 0;
      }
    }
  });
  for (auto& x : ts) x.join();
  double s1 = 0, s2 = 0, w = 0, dr = 0;
  games = 0; for (int t = 0; t < threads; t++) { games += N_[t]; s1 += S[t]; s2 += SS[t]; w += W[t]; dr += D[t]; }
  double m = s1 / games, sd = std::sqrt((s2 / games - m * m) / games);
  printf("rollouts: %ld games (1-ply bot both sides, limit 200 = draw): P(White) = %.5f +- %.5f   (wins %.5f, draws %.5f)  %.0f s\n",
         games, m, sd, w / games, dr / games, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}
int main(int argc, char** argv) {
  initTables(); Net* N = load(argv[1]);
  int depth = atoi(argv[2]); long games = atol(argv[3]); int threads = argc > 4 ? atoi(argv[4]) : 8;
#define C(hh) if (N->h == hh) { run<hh>(N, depth, games, threads); return 0; }
  BG_SIZES(C)
  return 1;
}
