// Off-policy consistency of Mixed nets: how far the 1-ply value of a move is from one more throw of search.
//   [N=positions] [SEED=n] [THREADS=n] ./offpolicy ref.json net1.json [net2.json ...]
// Build: g++ -std=c++20 -O3 -march=native -fno-trapping-math -pthread -o offpolicy offpolicy.cpp (in cpp/)
// Positions: the start with each of the 21 throws (every move), plus N (default 300) positions before a roll from
// self-play of ref at 1 ply (seeded, deduplicated, the same for every net) with one random throw each. For a move
// (an end of the mover's segment) v1 = its 1-ply value (the opponent's best walk, then the net) and v2 = node depth 2
// after it (the walk, then the opponent's next throw searched). A net that is right about a move has v1 ~ v2.
// Reported per net: mean |v1 - v2| for the start's moves (all, and without the net's own choice), and for the
// mid-game positions for the net's chosen move and for one random move (the same move for every net).
#include "core.h"
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
constexpr int H = 64;
struct Task { Pos<H> s; int a, b; bool start; uint64_t pick; };
struct Path { int p[MAXD]; int len; };

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: ./offpolicy ref.json net.json ...\n"); return 1; }
  initTables();
  const int N = getenv("N") ? atoi(getenv("N")) : 300, TH = getenv("THREADS") ? atoi(getenv("THREADS")) : 8;
  const uint64_t SEED = getenv("SEED") ? strtoull(getenv("SEED"), nullptr, 10) : 1;
  Net* ref = load(argv[1]);
  if (ref->h != H) { fprintf(stderr, "64 hidden units only\n"); return 1; }
  // tasks
  std::vector<Task> tasks;
  auto e = std::make_unique<Engine<H>>(); e->setRule(MIXED, 0, ref);
  { Pos<H> s; e->reset(s); for (int a = 1; a <= 6; a++) for (int b = a; b <= 6; b++) tasks.push_back({ s, a, b, true, 0 }); }
  Rng r; r.s = SEED * 0x9E3779B97F4A7C15ull | 1;
  std::unordered_set<uint64_t> seen;
  while ((int)tasks.size() < 21 + N) {
    Pos<H> s; e->reset(s);
    int n = 0;
    while (s.phase != OVER && n++ < 400 && (int)tasks.size() < 21 + N) {
      if (s.phase == ROLL) {
        if (n > 1 && r.next() < 0.05 && seen.insert(s.hash ^ (uint64_t)s.turn * 0x9E3779B97F4A7C15ull).second) {
          Pos<H> c = s; c.turnCount = 0;
          tasks.push_back({ c, r.die(), r.die(), false, (uint64_t)(r.next() * 1e9) });
        }
        e->roll(s, r.die(), r.die());
      } else e->playBest(s, nullptr);
    }
  }
  printf("%zu tasks (21 start throws, %d mid-game positions from %s, seed %llu)\n", tasks.size(), N, argv[1], (unsigned long long)SEED);
  printf("%-40s %10s %14s %12s %12s %8s\n", "net", "start all", "start not own", "mid own", "mid random", "time");
  for (int ni = 2; ni < argc; ni++) {
    Net* net = load(argv[ni]);
    if (net->h != H) { printf("%s: skipped (hidden size)\n", argv[ni]); continue; }
    auto t0 = std::chrono::steady_clock::now();
    std::atomic<int> next{0};
    std::vector<double> sA(TH), sN(TH), sO(TH), sR(TH);
    std::vector<long> nA(TH), nN(TH), nO(TH), nR(TH);
    std::vector<std::thread> ts;
    for (int t = 0; t < TH; t++) ts.emplace_back([&, t]() {
      auto g = std::make_unique<Engine<H>>(); g->setRule(MIXED, 0, net); g->ab = true;
      std::vector<Path> ends;
      for (int i; (i = next++) < (int)tasks.size();) {
        const Task& k = tasks[i];
        Pos<H> s = k.s; g->refresh(s); g->roll(s, k.a, k.b);
        if (s.phase != MOVE) continue; // no legal move
        int p = s.turn;
        ends.clear();
        g->moveDepth = 1; // plain enumeration: the same order for every net
        g->forEnds(s, [&](Pos<H>&, int len) { Path q; q.len = len; for (int m = 0; m < len; m++) q.p[m] = g->path[m]; ends.push_back(q); return true; });
        g->moveDepth = 2;
        std::vector<float> v1(ends.size());
        int best = 0;
        for (size_t j = 0; j < ends.size(); j++) {
          Pos<H> C = s; g->applyPath(C, ends[j].p, ends[j].len);
          v1[j] = g->leafValue(C, p);
          if (v1[j] > v1[best]) best = (int)j;
        }
        auto deep = [&](int j) { Pos<H> c = s; g->playPath(c, ends[j].p, ends[j].len); return g->node(c, p, 2, -1.f, 2.f); };
        if (k.start) {
          for (size_t j = 0; j < ends.size(); j++) {
            double d = std::fabs(v1[j] - deep((int)j));
            sA[t] += d; nA[t]++;
            if ((int)j != best) { sN[t] += d; nN[t]++; }
          }
        } else {
          sO[t] += std::fabs(v1[best] - deep(best)); nO[t]++;
          int j = (int)(k.pick % ends.size());
          sR[t] += std::fabs(v1[j] - deep(j)); nR[t]++;
        }
      }
    });
    for (auto& x : ts) x.join();
    double a = 0, b = 0, c = 0, d = 0; long na = 0, nb = 0, nc = 0, nd = 0;
    for (int t = 0; t < TH; t++) { a += sA[t]; b += sN[t]; c += sO[t]; d += sR[t]; na += nA[t]; nb += nN[t]; nc += nO[t]; nd += nR[t]; }
    printf("%-40s %10.4f %14.4f %12.4f %12.4f %7.0fs\n", argv[ni], a / na, b / nb, c / nc, d / nd,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    fflush(stdout);
  }
  return 0;
}
