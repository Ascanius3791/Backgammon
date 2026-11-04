// Accuracy-head test data: positions labelled with the champion's static error against search, its hidden layer,
// and the values of other nets (ensemble).
//   [GAMES=n] [PICK=p] [T2=s] [T3=s] [SEED=n] ./sigma champion.json out.txt other1.json [other2.json ...] [THREADS=n]
// Build: g++ -std=c++20 -O3 -march=native -fno-trapping-math -pthread -o sigma sigma.cpp (in cpp/)
// 1. GAMES (default 20000) self-play games of the champion at 1 ply; odd games start with 1..10 random segments
//    (off-policy positions). Every position before a roll (after the random opening) is kept with chance PICK
//    (default 0.05). Duplicates are dropped (exact board, bar, off and side to roll).
// 2. For T2 seconds (default 240): each position s (side P to roll) gets v0 = net value, the per-throw values
//    node(roll(s, throw), P, 1) (P's best move after the throw, the opponent's walk searched), their weighted
//    mean v2 and weighted standard deviation vol2.
// 3. For T3 seconds (default 240): as many of those as fit also get v3 = node(s, P, 3) (adds the opponent's next throw).
// Output line: game offpolicy v0 v2 vol2 v3(-1 = none) vOther... h[0..H) (tanh of the champion's hidden layer).
// The turn count is reset to 0 so the turn limit plays no part.
#include "core.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
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
  std::ifstream in(f); if (!in) { fprintf(stderr, "cannot read %s\n", f); exit(1); }
  std::stringstream ss; ss << in.rdbuf(); std::string t = ss.str();
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
static double env(const char* k, double d) { return getenv(k) ? atof(getenv(k)) : d; }
static double secs(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

constexpr int H = 64;
struct Item { Pos<H> s; long game; bool off; float v0, v2 = -1, vol2 = 0, v3 = -1; std::vector<float> vo; float h[H]; };

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: ./sigma champion.json out.txt other.json ...\n"); return 1; }
  initTables();
  Net* C = load(argv[1]);
  std::vector<Net*> others;
  for (int i = 3; i < argc; i++) others.push_back(load(argv[i]));
  for (Net* n : others) if (n->h != H || C->h != H) { fprintf(stderr, "all nets must have %d hidden units\n", H); return 1; }
  const long GAMES = env("GAMES", 20000);
  const double PICK = env("PICK", 0.05), T2 = env("T2", 240), T3 = env("T3", 240);
  const uint64_t SEED = env("SEED", 1);
  const int TH = env("THREADS", 8);

  // 1. collect
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::vector<Item>> got(TH);
  std::atomic<long> nextGame{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < TH; t++) ts.emplace_back([&, t]() {
    auto e = std::make_unique<Engine<H>>(); e->setRule(MIXED, 200, C); e->limitDraw = true;
    for (long g; (g = nextGame++) < GAMES;) {
      Rng r; r.s = 0x9E3779B97F4A7C15ull * (SEED * 1000003 + g + 1) | 1;
      Pos<H> p; e->reset(p);
      int open = g % 2 ? 1 + (int)(r.next() * 10) : 0, out[256];
      while (p.phase != OVER && p.turnCount < 200) {
        if (p.phase == ROLL) {
          if (!open && r.next() < PICK) { Item it; it.s = p; it.s.turnCount = 0; it.game = g; it.off = g % 2; got[t].push_back(it); }
          e->roll(p, r.die(), r.die());
          continue;
        }
        if (open) {
          int m = p.turn;
          while (p.phase == MOVE || p.phase == WALK) {
            e->applyMove(p, out[(int)(r.next() * e->genAll(p, m, out))]);
            if (p.turn != m) break;
          }
          open--;
          continue;
        }
        e->playBest(p, nullptr);
      }
    }
  });
  for (auto& x : ts) x.join();
  ts.clear();
  std::vector<Item> items;
  std::unordered_set<std::string> seen;
  for (auto& v : got) for (auto& it : v) {
    std::string k(30, '\0');
    for (int i = 0; i < 25; i++) k[i] = (char)it.s.b[i];
    k[25] = it.s.bar[0]; k[26] = it.s.bar[1]; k[27] = it.s.off[0]; k[28] = it.s.off[1]; k[29] = it.s.turn;
    if (seen.insert(k).second) items.push_back(it);
  }
  long picked = 0; for (auto& v : got) picked += v.size();
  // shuffle so that the time-bounded labelling is a random subset
  Rng sh; sh.s = SEED * 0xD1B54A32D192ED03ull | 1;
  for (long i = (long)items.size() - 1; i > 0; i--) std::swap(items[i], items[(long)(sh.next() * (i + 1))]);
  printf("collect: %ld games, %ld positions picked, %zu distinct, %.1f s\n", GAMES, picked, items.size(), secs(t0));
  fflush(stdout);

  // 2. static values, hidden layer, depth-2 labels
  t0 = std::chrono::steady_clock::now();
  std::atomic<long> next{0};
  for (int t = 0; t < TH; t++) ts.emplace_back([&]() {
    auto e = std::make_unique<Engine<H>>(); e->setRule(MIXED, 200, C); e->limitDraw = true; e->ab = true;
    auto o = std::make_unique<Engine<H>>(); o->setRule(MIXED, 200, C);
    for (long i; (i = next++) < (long)items.size() && secs(t0) < T2;) {
      Item& it = items[i];
      Pos<H> s = it.s; e->refresh(s);
      int P = s.turn;
      it.v0 = e->evalPos(s, P);
      { // hidden layer as in evalPos
        const float* acc = s.acc + P * H; const float* B = C->W1 + 240 * H; int Q = 1 - P;
        float ex[6] = { s.bar[P] / 5.f, s.bar[Q] / 5.f, s.off[P] / 15.f, s.off[Q] / 15.f, s.pip[P] / 200.f, s.pip[Q] / 200.f };
        for (int j = 0; j < H; j++) { float x = acc[j]; for (int k = 0; k < 6; k++) x += ex[k] * B[k * H + j]; it.h[j] = std::tanh(x); }
      }
      for (Net* n : others) { o->net = n; Pos<H> q = it.s; o->refresh(q); it.vo.push_back(o->evalPos(q, P)); }
      double m = 0, m2 = 0;
      for (int a = 1; a <= 6; a++) for (int b = a; b <= 6; b++) {
        double w = (a == b ? 1 : 2) / 36.0;
        Pos<H> c = s; e->roll(c, a, b);
        double v = e->node(c, P, 1, -1, 2);
        m += w * v; m2 += w * v * v;
      }
      it.v2 = m; it.vol2 = std::sqrt(std::max(0.0, m2 - m * m));
    }
  });
  for (auto& x : ts) x.join();
  ts.clear();
  long n2 = 0; for (auto& it : items) n2 += it.v2 >= 0;
  printf("depth 2: %ld positions, %.1f s\n", n2, secs(t0));
  fflush(stdout);

  // 3. depth-3 labels on as many as fit
  t0 = std::chrono::steady_clock::now();
  next = 0;
  for (int t = 0; t < TH; t++) ts.emplace_back([&]() {
    auto e = std::make_unique<Engine<H>>(); e->setRule(MIXED, 200, C); e->limitDraw = true; e->ab = true;
    for (long i; (i = next++) < n2 && secs(t0) < T3;) {
      Item& it = items[i];
      Pos<H> s = it.s; e->refresh(s);
      it.v3 = e->node(s, s.turn, 3, -1, 2); // a search that ends after the deadline still counts
    }
  });
  for (auto& x : ts) x.join();
  long n3 = 0; for (auto& it : items) n3 += it.v3 >= 0;
  printf("depth 3: %ld positions, %.1f s\n", n3, secs(t0));

  FILE* f = fopen(argv[2], "w");
  if (!f) { perror(argv[2]); return 1; }
  for (auto& it : items) {
    if (it.v2 < 0) continue;
    fprintf(f, "%ld %d %.5f %.5f %.5f %.5f", it.game, (int)it.off, it.v0, it.v2, it.vol2, it.v3);
    for (float v : it.vo) fprintf(f, " %.5f", v);
    for (int j = 0; j < H; j++) fprintf(f, " %.4f", it.h[j]);
    fprintf(f, "\n");
  }
  fclose(f);
  return 0;
}
