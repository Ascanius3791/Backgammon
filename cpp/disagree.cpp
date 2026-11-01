// Disagreement set: positions where two nets' values differ most, from games between them (1-ply play).
//   [TIME=seconds] [OPENING=n] [LIMIT=n] [SEED=n] ./disagree a.json b.json out.txt [N] [threads]
// Build: g++ -std=c++20 -O3 -march=native -fno-trapping-math -pthread -o disagree disagree.cpp (in cpp/)
// The games are played as in match.cpp (A and B alternate colours, OPENING random openings) for TIME seconds
// (default 5). Every position before a roll is scored by both nets (P(side to roll wins), depth 1); duplicates
// are merged (exact board, bar, off and side to roll). The N (default 2000) with the largest |vA - vB| are
// written to out.txt, one per line: b[0..24] bar0 bar1 off0 off1 turn vA vB (b as in core.h, + White).
// bgtrain BSET=out.txt starts games from them; match.cpp START=out.txt plays pairs from them.
#include "core.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
using namespace bg;

static std::vector<double> nums(const std::string& t, const char* key) {
  std::vector<double> out;
  size_t i = t.find(std::string("\"") + key + "\"");
  if (i == std::string::npos) return out;
  i = t.find(':', i) + 1;
  bool arr = t[i] == '[';
  if (arr) i++;
  const char* p = t.c_str() + i;
  for (;;) {
    while (*p == ' ' || *p == ',' || *p == '\n') p++;
    if (*p == ']' || !*p) break;
    char* e;
    out.push_back(strtod(p, &e));
    if (e == p || !arr) break;
    p = e;
  }
  return out;
}
static Net* load(const char* f) {
  std::ifstream in(f);
  std::stringstream ss; ss << in.rdbuf();
  std::string t = ss.str();
  auto W1 = nums(t, "W1"), b1 = nums(t, "b1"), W2 = nums(t, "W2"), b2 = nums(t, "b2");
  Net* n = new Net();
  n->h = b1.size(); n->nin = W1.size() / n->h;
  for (size_t i = 0; i < W1.size(); i++) n->W1[i] = W1[i];
  for (int j = 0; j < n->h; j++) { n->b1[j] = b1[j]; n->W2[j] = W2[j]; }
  n->b2 = b2[0];
  auto Wd = nums(t, "Wd"), bd = nums(t, "bd"); // residual layers, if any
  n->deep = bd.size() / n->h;
  for (int k = 0; k < n->deep; k++) {
    for (int i = 0; i < n->h * n->h; i++) n->Wd[k][i] = Wd[k * n->h * n->h + i];
    for (int j = 0; j < n->h; j++) n->bd[k][j] = bd[k * n->h + j];
  }
  return n;
}

const int LIMIT = getenv("LIMIT") ? atoi(getenv("LIMIT")) : 200;
const int OPENING = getenv("OPENING") ? atoi(getenv("OPENING")) : 0;

struct Seen { std::string key; float va, vb; };
using Map = std::unordered_map<std::string, Seen>;

template <int H> static std::string keyOf(const Pos<H>& s) {
  std::string k(30, '\0');
  for (int i = 0; i < 25; i++) k[i] = (char)s.b[i];
  k[25] = s.bar[0]; k[26] = s.bar[1]; k[27] = s.off[0]; k[28] = s.off[1]; k[29] = s.turn;
  return k;
}

// one game; every position before a roll goes into m
template <int HA, int HB> static void game(Engine<HA>& ea, Engine<HB>& eb, Rng& r, bool aWhite, Map& m, long& npos) {
  Pos<HA> pa; Pos<HB> pb;
  ea.reset(pa); eb.reset(pb);
  int path[MAXD], len, n = 0;
  int open = OPENING ? (int)(r.next() * (OPENING + 1)) : 0;
  while (pa.phase != OVER && ++n < 4000) {
    if (pa.phase == ROLL) {
      if (!open) {
        npos++;
        std::string k = keyOf(pa);
        if (!m.count(k)) m[k] = { k, ea.evalPos(pa, pa.turn), eb.evalPos(pb, pb.turn) };
      }
      int a = r.die(), b = r.die(); ea.roll(pa, a, b); eb.roll(pb, a, b);
      continue;
    }
    if (open) { // one random segment
      int p = pa.turn, out[256];
      while (pa.phase == MOVE || pa.phase == WALK) {
        int mv = out[(int)(r.next() * ea.genAll(pa, p, out))];
        ea.applyMove(pa, mv); eb.applyMove(pb, mv);
        if (pa.turn != p) break;
      }
      open--;
      continue;
    }
    bool aMoves = (pa.turn == 0) == aWhite;
    if (aMoves) { ea.best1(pa); len = ea.bestLen; for (int k = 0; k < len; k++) path[k] = ea.bestPath[k]; }
    else { eb.best1(pb); len = eb.bestLen; for (int k = 0; k < len; k++) path[k] = eb.bestPath[k]; }
    for (int k = 0; k < len && (pa.phase == MOVE || pa.phase == WALK); k++) { ea.applyMove(pa, path[k]); eb.applyMove(pb, path[k]); }
  }
}

template <int HA, int HB> static void run(Net* A, Net* B, const char* outFile, int N, int threads) {
  const double T = getenv("TIME") ? atof(getenv("TIME")) : 5;
  const uint64_t seed = getenv("SEED") ? strtoull(getenv("SEED"), nullptr, 10) : 1;
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<Map> maps(threads);
  std::vector<long> games(threads), npos(threads);
  std::vector<std::thread> ts;
  for (int t = 0; t < threads; t++) ts.emplace_back([&, t]() {
    auto ea = std::make_unique<Engine<HA>>(); auto eb = std::make_unique<Engine<HB>>();
    ea->setRule(MIXED, LIMIT, A); eb->setRule(MIXED, LIMIT, B);
    Rng r; r.s = 0x9E3779B97F4A7C15ull * (seed * 1000 + t + 1) | 1;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < T) {
      game(*ea, *eb, r, games[t] % 2 == 0, maps[t], npos[t]);
      games[t]++;
    }
  });
  for (auto& t : ts) t.join();
  Map all;
  long g = 0, np = 0;
  for (int t = 0; t < threads; t++) { g += games[t]; np += npos[t]; for (auto& [k, v] : maps[t]) all.emplace(k, v); }
  std::vector<const Seen*> v;
  for (auto& [k, s] : all) v.push_back(&s);
  std::sort(v.begin(), v.end(), [](const Seen* a, const Seen* b) { return std::fabs(a->va - a->vb) > std::fabs(b->va - b->vb); });
  if ((int)v.size() > N) v.resize(N);
  FILE* o = fopen(outFile, "w");
  if (!o) { perror(outFile); exit(1); }
  for (auto* s : v) {
    for (int i = 0; i < 30; i++) fprintf(o, "%d ", i < 25 ? (int)(int8_t)s->key[i] : (int)(uint8_t)s->key[i]);
    fprintf(o, "%.4f %.4f\n", s->va, s->vb);
  }
  fclose(o);
  double sum = 0;
  for (auto* s : v) sum += std::fabs(s->va - s->vb);
  printf("%ld games, %ld positions, %zu distinct; wrote %zu to %s, |vA - vB| from %.3f to %.3f (mean %.3f)\n", g, np, all.size(), v.size(), outFile,
         v.empty() ? 0 : std::fabs(v.front()->va - v.front()->vb), v.empty() ? 0 : std::fabs(v.back()->va - v.back()->vb), v.empty() ? 0 : sum / v.size());
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: [TIME=s] [OPENING=n] ./disagree a.json b.json out.txt [N] [threads]\n"); return 1; }
  initTables();
  Net *A = load(argv[1]), *B = load(argv[2]);
  int N = argc > 4 ? atoi(argv[4]) : 2000, threads = argc > 5 ? atoi(argv[5]) : 8;
#define CB(hb, ha) if (A->h == ha && B->h == hb) { run<ha, hb>(A, B, argv[3], N, threads); return 0; }
#define CA(ha) BG_SIZES_B(ha)
#define BG_SIZES_B(ha) CB(64, ha) CB(128, ha) CB(256, ha)
  BG_SIZES(CA)
  fprintf(stderr, "unsupported sizes\n");
  return 1;
}
