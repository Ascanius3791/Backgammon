// Mixed match between two nets of any compiled hidden sizes, 1-ply play.
//   [OPENING=n] [LIMIT=n] ./match a.json b.json [pairs] [threads]
// Build: g++ -std=c++20 -O3 -march=native -fno-trapping-math -pthread -o match match.cpp (in cpp/)
// LIMIT (default 200 turns): a game cut there is a draw (counted as unfinished); LIMITDRAW=0: the pip leader wins.
// Each dice sequence is played twice with colours swapped. Prints A's score and Elo with std error.
// OPENING=n: each pair starts from the same random opening, k ~ U{0..n} random segments (throws and walks),
// the moves and the dice drawn from the pair's seed, so both games of a pair are identical up to the colours.
// Also prints the pairs won 2-0 / lost 0-2 / split.
// START=file (from disagree.cpp): pair i starts from position i of the file (cycled), without an opening.
#include "core.h"
#include <algorithm>
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

std::atomic<int> unfinished{0};
std::atomic<long> turns{0}, games{0}; // game length in turns (throws), finished games
const int LIMIT = getenv("LIMIT") ? atoi(getenv("LIMIT")) : 200;
const int OPENING = getenv("OPENING") ? atoi(getenv("OPENING")) : 0;
const bool LIMITDRAW = getenv("LIMITDRAW") ? atoi(getenv("LIMITDRAW")) != 0 : true; // a game cut at LIMIT is a draw
struct Start { int8_t b[25]; uint8_t bar[2], off[2], turn; };
static std::vector<Start> starts; // START file: b[0..24] bar0 bar1 off0 off1 turn per line
static void readStarts(const char* f) {
  std::ifstream in(f);
  if (!in) { fprintf(stderr, "START: cannot read %s\n", f); exit(1); }
  for (std::string line; std::getline(in, line);) {
    std::istringstream ls(line);
    int v[30], i = 0;
    while (i < 30 && ls >> v[i]) i++;
    if (i < 30) continue;
    Start r;
    for (int k = 0; k < 25; k++) r.b[k] = (int8_t)v[k];
    r.bar[0] = v[25]; r.bar[1] = v[26]; r.off[0] = v[27]; r.off[1] = v[28]; r.turn = v[29];
    starts.push_back(r);
  }
}
template <int H> static void setStart(Engine<H>& e, Pos<H>& p, const Start& r) {
  for (int i = 0; i < 25; i++) p.b[i] = r.b[i];
  p.bar[0] = r.bar[0]; p.bar[1] = r.bar[1]; p.off[0] = r.off[0]; p.off[1] = r.off[1]; p.turn = r.turn;
  e.refresh(p);
}

// winner of one game, aWhite: net A plays White; dice from seed
template <int HA, int HB> int game(Engine<HA>& ea, Engine<HB>& eb, uint64_t seed, bool aWhite, const Start* st) {
  Pos<HA> pa; Pos<HB> pb;
  ea.reset(pa); eb.reset(pb);
  if (st) { setStart(ea, pa, *st); setStart(eb, pb, *st); }
  Rng r; r.s = seed;
  int path[MAXD], len, n = 0;
  Rng o; o.s = (seed ^ 0xD1B54A32D192ED03ull) | 1; // the opening's moves, the same in both games of a pair
  int open = OPENING && !st ? (int)(o.next() * (OPENING + 1)) : 0;
  while (pa.phase != OVER) {
    if (++n > 4000) { // only possible without LIMIT
      if (getenv("DUMP") && unfinished == 0) {
        printf("stalled: White pts");
        for (int a = 1; a <= 24; a++) if (pa.b[a] > 0) printf(" %d:%d", a, pa.b[a]);
        printf("  Black pts(own numbering)");
        for (int a = 1; a <= 24; a++) if (pa.b[a] < 0) printf(" %d:%d", 25 - a, -pa.b[a]);
        printf("  bar %d/%d off %d/%d pips %d/%d\n", pa.bar[0], pa.bar[1], pa.off[0], pa.off[1], pa.pip[0], pa.pip[1]);
      }
      unfinished++; return -1;
    }
    if (getenv("DUMP") && n > 3950 && unfinished == 0 && pa.phase == ROLL) {
      printf("  n %d turn %d pips %d/%d  W", n, pa.turn, pa.pip[0], pa.pip[1]);
      for (int a = 1; a <= 24; a++) if (pa.b[a] > 0) printf(" %d:%d", a, pa.b[a]);
      printf("  B");
      for (int a = 1; a <= 24; a++) if (pa.b[a] < 0) printf(" %d:%d", 25 - a, -pa.b[a]);
      printf("\n");
    }
    if (pa.phase == ROLL) { int a = r.die(), b = r.die(); ea.roll(pa, a, b); eb.roll(pb, a, b); continue; }
    if (open) { // one random segment, the same moves on both boards
      int p = pa.turn, out[256];
      while (pa.phase == MOVE || pa.phase == WALK) {
        int m = out[(int)(o.next() * ea.genAll(pa, p, out))];
        ea.applyMove(pa, m); eb.applyMove(pb, m);
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
  if (pa.limitHit && pa.winner < 0) { unfinished++; return -1; } // cut at LIMIT: a draw (LIMITDRAW=0: pip leader wins)
  turns += pa.turnCount; games++;
  return pa.winner;
}

template <int HA, int HB> void run(Net* A, Net* B, int pairs, int threads) {
  std::atomic<int> next{0};
  std::vector<double> sc(pairs);
  std::vector<std::thread> ts;
  for (int t = 0; t < threads; t++) ts.emplace_back([&]() {
    auto ea = std::make_unique<Engine<HA>>(); auto eb = std::make_unique<Engine<HB>>();
    ea->setRule(MIXED, LIMIT, A); eb->setRule(MIXED, LIMIT, B);
    ea->limitDraw = eb->limitDraw = LIMITDRAW;
    for (int i; (i = next++) < pairs;) {
      uint64_t seed = 0x9E3779B97F4A7C15ull * (i + 1) | 1;
      double s = 0;
      const Start* st = starts.empty() ? nullptr : &starts[i % starts.size()];
      int w1 = game(*ea, *eb, seed, true, st), w2 = game(*ea, *eb, seed, false, st);
      s += w1 == 0 ? 1 : w1 < 0 ? 0.5 : 0;
      s += w2 == 1 ? 1 : w2 < 0 ? 0.5 : 0;
      sc[i] = s / 2;
    }
  });
  for (auto& t : ts) t.join();
  double m = 0, v = 0;
  for (double x : sc) m += x;
  m /= pairs;
  for (double x : sc) v += (x - m) * (x - m);
  double se = std::sqrt(v / (pairs - 1) / pairs);
  auto elo = [](double p) { return -400 * std::log10(1 / p - 1); };
  int won = 0, lost = 0;
  for (double x : sc) { won += x == 1; lost += x == 0; }
  double q = won + lost ? (double)won / (won + lost) : 0.5, qse = won + lost ? std::sqrt(q * (1 - q) / (won + lost)) : 0;
  printf("pairs: A won %d, lost %d, split %d   A's share of decided pairs %.4f +- %.4f\n", won, lost, pairs - won - lost, q, qse);
  printf("A score %.4f +- %.4f   Elo %+.1f +- %.1f   (%d pairs, H %d vs %d, %d unfinished)\n", m, se, elo(m), (elo(m + se) - elo(m - se)) / 2, pairs, HA, HB, unfinished.load());
  printf("mean game length %.1f turns (finished games)\n", (double)turns / std::max(1L, games.load()));
}

int main(int argc, char** argv) {
  initTables();
  Net *A = load(argv[1]), *B = load(argv[2]);
  if (getenv("START")) readStarts(getenv("START"));
  int pairs = argc > 3 ? atoi(argv[3]) : 2000, threads = argc > 4 ? atoi(argv[4]) : 8;
#define CB(hb, ha) if (A->h == ha && B->h == hb) { run<ha, hb>(A, B, pairs, threads); return 0; }
#define CA(ha) BG_SIZES_B(ha)
#define BG_SIZES_B(ha) CB(64, ha) CB(128, ha) CB(256, ha)
  BG_SIZES(CA)
  fprintf(stderr, "unsupported sizes\n");
  return 1;
}
