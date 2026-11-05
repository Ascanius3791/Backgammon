// How accurate is a net's accuracy head? Self-play of the net (1-ply) from random openings; at every decision
// (each throw's and walk's segment, both sides) every legal end is scored. Per end: the position the 1-ply value
// scores (core.h Engine::endState, after the opponent's best walk) as a clean position before a roll, then
//   v0 = the net's value of it, v2 = node(it, side to roll, 2) (2 rolls searched), e = v0 - v2,
//   sigma = sqrt(exp(head)): the head's prediction of |e| (its training target, rms).
// Positions are deduplicated (board, bar, off, side to roll). One line per position to out.txt:
//   game played(0/1) v0 v2 sigma
//   [OPENING=10] [TIME=120] [THREADS=8] [SEED=1] ./sigmacheck net.json out.txt
// Build: g++ -std=c++20 -O3 -march=native -fno-trapping-math -pthread -o sigmacheck sigmacheck.cpp (in cpp/)
#include "core.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
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
  auto S1 = nums(t, "S1"), Sb1 = nums(t, "Sb1"), S2 = nums(t, "S2"), Sb2 = nums(t, "Sb2");
  constexpr int K = Net::SIG_K;
  if ((int)S1.size() != n->h * K || (int)Sb1.size() != K || (int)S2.size() != K || Sb2.size() != 1) {
    fprintf(stderr, "%s has no accuracy head\n", f); exit(1);
  }
  n->sig = true;
  for (int i = 0; i < n->h * K; i++) n->S1[i] = S1[i];
  for (int k = 0; k < K; k++) { n->Sb1[k] = Sb1[k]; n->S2[k] = S2[k]; }
  n->Sb2 = Sb2[0];
  return n;
}
constexpr int H = 64;

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: ./sigmacheck net.json out.txt\n"); return 1; }
  initTables();
  const int TH = getenv("THREADS") ? atoi(getenv("THREADS")) : 8, OPENING = getenv("OPENING") ? atoi(getenv("OPENING")) : 10;
  const double TIME = getenv("TIME") ? atof(getenv("TIME")) : 120;
  const uint64_t SEED = getenv("SEED") ? strtoull(getenv("SEED"), nullptr, 10) : 1;
  Net* net = load(argv[1]);
  if (net->h != H) { fprintf(stderr, "64 hidden units only\n"); return 1; }
  FILE* out = fopen(argv[2], "w");
  std::mutex mu;
  std::unordered_set<uint64_t> seen;
  std::atomic<int> nextGame{0};
  std::atomic<long> decisions{0};
  auto t0 = std::chrono::steady_clock::now();
  auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  std::vector<std::thread> ts;
  for (int t = 0; t < TH; t++) ts.emplace_back([&]() {
    auto g = std::make_unique<Engine<H>>(); g->setRule(MIXED, 0, net); g->moveDepth = 2; g->plyLimit = 200; g->limitDraw = true;
    struct End { Pos<H> R; int view; uint64_t chash; };
    std::vector<End> ends;
    while (elapsed() < TIME) {
      int game = nextGame++;
      Rng r; r.s = (SEED * 1000003ull + game) * 0x9E3779B97F4A7C15ull | 1;
      Pos<H> s; g->reset(s);
      int open = (int)(r.next() * (OPENING + 1));
      while (s.phase != OVER && elapsed() < TIME) {
        if (s.phase == ROLL) { g->roll(s, r.die(), r.die()); continue; }
        int p = s.turn;
        if (open) { // one random segment
          int mv[256];
          while ((s.phase == MOVE || s.phase == WALK) && s.turn == p) g->applyMove(s, mv[(int)(r.next() * g->genAll(s, p, mv))]);
          open--;
          continue;
        }
        // every end of the segment, then the 1-ply choice
        ends.clear();
        g->forEnds(s, [&](Pos<H>& C, int) {
          End e; Pos<H> o; float term;
          if (!g->endState(C, p, o, e.view, &term)) return true; // a final value: nothing to predict
          g->reset(e.R);
          for (int i = 0; i < 25; i++) e.R.b[i] = o.b[i];
          e.R.bar[0] = o.bar[0]; e.R.bar[1] = o.bar[1]; e.R.off[0] = o.off[0]; e.R.off[1] = o.off[1]; e.R.turn = (uint8_t)e.view;
          e.chash = C.hash;
          ends.push_back(e);
          return true;
        });
        g->best1(s);
        int blen = g->bestLen, bpath[MAXD];
        for (int k = 0; k < blen; k++) bpath[k] = g->bestPath[k];
        Pos<H> pc = s; g->applyPath(pc, bpath, blen);
        for (auto& e : ends) {
          g->refresh(e.R);
          uint64_t key = e.R.hash ^ (uint64_t)(e.R.turn + 1) * 0x9E3779B97F4A7C15ull;
          { std::lock_guard<std::mutex> l(mu); if (!seen.insert(key).second) continue; }
          bool played = e.chash == pc.hash;
          float v0 = g->evalPos(e.R, e.view), v2 = g->node(e.R, e.view, 2, -1.f, 2.f);
          double sg = std::sqrt(std::exp((double)g->sigmaLog(e.R, e.view)));
          std::lock_guard<std::mutex> l(mu);
          fprintf(out, "%d %d %.6f %.6f %.6f\n", game, played ? 1 : 0, v0, v2, sg);
        }
        decisions++;
        for (int k = 0; k < blen && (s.phase == MOVE || s.phase == WALK); k++) g->applyMove(s, bpath[k]);
      }
    }
  });
  for (auto& x : ts) x.join();
  fclose(out);
  printf("%d games started, %ld decisions, %zu distinct positions, %.0f s\n", nextGame.load(), decisions.load(), seen.size(), elapsed());
  return 0;
}
