// TD(lambda) self-play training on the C++ core, native and multi-threaded. Same command line, files
// and training scheme as train.js:
//   RULE=paul|rene|mixed [PRE=n] [WORKERS=n] [LIMIT=n] [LOG=n] [TIME=seconds] ./bgtrain [games] [alpha] [lambda] [evalEvery]
// Resumes from weights_<rule>.json. Writes weights_<rule>.json/.js and metrics_<rule>.js (for plot.html)
// next to the executable. PRE: the first PRE games are played by the greedy baseline instead of the net.
// WORKERS (default 4): self-play threads sharing one set of weights, updated without locks (Hogwild),
// each with its own eligibility trace. TIME: stop starting new games after this many seconds
// (games is then only an upper bound); the last partial row is logged and evaluated.
// HIDDEN (64, 128 or 256): hidden units of a new net. A smaller saved net is widened to it without
// changing its output: the new units get small random input weights and output weight 0.
// VSGREEDY (Mixed, default 1): 0 = after the warm-up, pure self-play instead of every second game
// against the greedy baseline.
// GAMMA (default 1): per-step discount of the TD target toward 0.5 (a game that never ends is a draw),
// target = 0.5 + GAMMA * (V(next) - 0.5). Below 1, a win now beats the same win later, so a winning side
// stops stalling (e.g. walking a runner back and forth behind a prime instead of coming home).
// LIMITDRAW (Mixed, default 0): 1 = a game cut at LIMIT counts as a draw (target 0.5) instead of a win for
// the pip leader, so the side that is ahead only scores by actually bearing off.
// TRAIN_DEPTH (Mixed, default 2): how the self-play games choose their own throw. 2 = the full design: each
// candidate is scored after the opponent's forced walk of that throw (2-ply minimax). 1 = scored as if the
// opponent rolled next, ignoring the walk: several times faster, meant for pretraining only. The evaluation
// games against the greedy baseline always use depth 2.
// EXPLORE (default 0): chance per segment (a throw or a walk, either side) that the net's side plays random moves
// instead of its best, so positions the net rates badly still get visited and corrected. Without it, a move whose
// walk search hits one falsely low position is never played, and the error is never corrected.
// REPLAY (Mixed, games per cycle; 0 = off, the online TD above): replay-buffer training. Cycles of
//   1. play REPLAY games with the frozen net (workers, as in the online mode, EXPLORE included) and append them to a
//      buffer of the last KEEP games (positions before every roll, and the result);
//   2. REPS * REPLAY game visits on games drawn at random from the buffer: the lambda-return of every position is
//      computed with the current net (V(next) bootstrap, GAMMA as above), then the position is moved toward it by
//      ALPHA * (return - V) (ALPHA falls linearly to ALPHAEND over TIME). The weights change only in this phase.
// TEMP (default 0): softmax exploration in the self-play games: each end of a segment (the net's side, throw or
// walk) is taken with probability ~ exp((v - best) / TEMP), v its value in win probability as in playBest.
// TEMPEND: TEMP falls linearly to it over TIME. The plain TD target then learns the value of the softmax
// policy, not of the greedy one that is played later. TEMPFIX=1 corrects for that (Watkins' Q(lambda)): in a
// step where some end worse than the best was taken, the target is the mover's best value of the throw (the
// greedy continuation) instead of the next position, and the eligibility trace is cut after the update.
// SEARCH (default 0): n >= 2 = search distillation instead of TD(lambda): every position of the self-play
// games (before a roll or a walk) is trained toward its depth-n expectimax value (Engine::node, full
// window; n = 2 looks one roll further than the net), discounted toward 0.5 by GAMMA.
#include "core.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

using namespace bg;

static std::string readFile(const std::string& f) {
  std::ifstream in(f, std::ios::binary);
  if (!in) return "";
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}
static void writeFile(const std::string& f, const std::string& s) {
  std::string tmp = f + ".tmp";
  FILE* o = fopen(tmp.c_str(), "wb");
  if (!o) { perror(tmp.c_str()); exit(1); }
  fwrite(s.data(), 1, s.size(), o);
  fclose(o);
  rename(tmp.c_str(), f.c_str()); // never leaves a half-written file behind
}

// the numbers of "key":[...] (or the single number of "key":x) in a JSON text
static std::vector<double> jsonNumbers(const std::string& t, const char* key) {
  std::vector<double> out;
  size_t i = t.find(std::string("\"") + key + "\"");
  if (i == std::string::npos) return out;
  i = t.find(':', i) + 1;
  while (t[i] == ' ') i++;
  bool arr = t[i] == '[';
  if (arr) i++;
  const char* p = t.c_str() + i;
  for (;;) {
    while (*p == ' ' || *p == ',' || *p == '\n') p++;
    if (*p == ']' || !*p) break;
    char* e;
    out.push_back(strtod(p, &e));
    if (e == p) break;
    p = e;
    if (!arr) break;
  }
  return out;
}

// like bot.js toJSON: weights rounded to 1e-5
static std::string num5(double x) {
  char b[32];
  snprintf(b, sizeof b, "%.5f", std::round(x * 1e5) / 1e5);
  std::string s = b;
  while (s.back() == '0') s.pop_back();
  if (s.back() == '.') s.pop_back();
  return s == "-0" ? "0" : s;
}
static std::string toJSON(const Net& n) {
  std::string s = "{\"W1\":[";
  for (int i = 0; i < n.nin * n.h; i++) s += (i ? "," : "") + num5(n.W1[i]);
  s += "],\"b1\":[";
  for (int i = 0; i < n.h; i++) s += (i ? "," : "") + num5(n.b1[i]);
  s += "],\"W2\":[";
  for (int i = 0; i < n.h; i++) s += (i ? "," : "") + num5(n.W2[i]);
  char b[64];
  snprintf(b, sizeof b, "],\"b2\":%.9g}", n.b2);
  return s + b;
}

struct Opts { int rule, pre, turnLimit, maxPlies; double alpha, lambda; long last; bool vsGreedy; double gamma; bool limitDraw; int search; double explore; double temp; bool tempFix; };
struct Run { // everything main sets up before the training proper
  int rule, PRE, WORKERS, VSGREEDY, LIMITDRAW, SEARCH, TRAIN_DEPTH;
  double EXPLORE, TEMP, TEMPEND;
  int TEMPFIX;
  long GAMES, EVAL_EVERY, LOG_EVERY;
  double ALPHA, LAMBDA, TIME, GAMMA;
  int TURN_LIMIT, MAX_PLIES;
  std::string dir, ruleName, RULE_UP, FILE_W, METRICS;
  Net* net;
  long REPLAY, KEEP; int REPS; double ALPHAEND; // replay-buffer mode (REPLAY = games per cycle, 0 = off)
};
struct Result { int plies, steps; double sqErr; bool capped; };

// greedySide: 2 = the greedy baseline plays both sides, 0/1 = it plays that side, -1 = pure self-play
template <int H> static Result selfPlayGame(Engine<H>& g, TD<H>& td, Pos<H>& s, int greedySide, const Opts& o) {
  g.reset(s);
  td.clear();
  int plies = 0, steps = 0;
  double sqErr = 0, cur = 0;
  while (s.phase != OVER && plies++ < o.maxPlies) {
    // gradient of White's winning chance in the current state goes into the eligibility trace
    double y = td.forward(g, s);
    cur = s.turn == 0 ? y : 1 - y;
    td.trace(s.turn == 0 ? 1 : -1);

    // one step: the roll, the mover's segment and, in Mixed, the opponent's forced walk of that throw.
    // The net only ever sees positions with nobody holding dice (no walk inputs; the walk is searched).
    if (s.phase == ROLL) { int a = g.rng.die(), b = g.rng.die(); g.roll(s, a, b); }
    int firstP = -1; // TEMPFIX: who played the first segment of the step, and the best value it had
    double firstBest = 0;
    bool off = false;
    while (s.phase == MOVE || s.phase == WALK) {
      int p = s.turn;
      bool soft = false;
      if (greedySide == 2 || s.turn == greedySide) g.playGreedy(s, nullptr);
      else if (o.explore > 0 && g.rng.next() < o.explore) g.playRandom(s, nullptr);
      else if (o.temp > 0) { g.playSoft(s, nullptr); soft = true; off |= g.softOff; }
      else g.playBest(s, nullptr);
      if (firstP < 0) { firstP = soft ? p : 2; firstBest = g.softBest; }
    }

    double nw;
    bool fix = o.tempFix && off && firstP < 2; // a greedy-baseline first segment has no best value to back up
    if (fix) nw = 0.5 + o.gamma * ((firstP == 0 ? firstBest : 1 - firstBest) - 0.5);
    else if (s.phase == OVER) nw = s.limitHit && o.limitDraw ? 0.5 : s.winner == 0 ? 1 : 0;
    else { double y2 = td.forward(g, s); nw = 0.5 + o.gamma * ((s.turn == 0 ? y2 : 1 - y2) - 0.5); }
    sqErr += (nw - cur) * (nw - cur);
    steps++;
    td.update(o.alpha * (nw - cur));
    if (fix) td.clear(); // the rest of the game follows a non-greedy end: no credit for the positions before it
  }
  bool capped = s.phase != OVER;
  if (capped) td.update(o.alpha * (0.5 - cur)); // never finished: a draw
  return { plies, steps, sqErr, capped || s.limitHit };
}

// search distillation: a self-play game (net, 1-ply) whose positions are each trained toward their
// depth-o.search value; td runs with lambda 0, so a trace holds just the gradient of the current position
template <int H> static Result searchGame(Engine<H>& g, TD<H>& td, Pos<H>& s, const Opts& o) {
  g.reset(s);
  int plies = 0, steps = 0;
  double sqErr = 0;
  while (s.phase != OVER && plies++ < o.maxPlies) {
    g.refresh(s); // node() plays on from s's accumulators, which the updates have made stale
    double t = 0.5 + o.gamma * (g.node(s, s.turn, o.search, -1.f, 2.f) - 0.5);
    double y = td.forward(g, s); // P(side to move wins), like node(s, s.turn, ...)
    td.trace(1);
    td.update(o.alpha * (t - y));
    sqErr += (t - y) * (t - y);
    steps++;
    if (s.phase == ROLL) { int a = g.rng.die(), b = g.rng.die(); g.roll(s, a, b); }
    if (s.phase == MOVE || s.phase == WALK) g.playBest(s, nullptr);
  }
  return { plies, steps, sqErr, s.phase != OVER || s.limitHit };
}

template <int H> static int run(const Run& R);
template <int H> static int runReplay(const Run& R);

int main(int argc, char** argv) {
  initTables();
  const char* ruleName = getenv("RULE") ? getenv("RULE") : "paul";
  int rule = !strcmp(ruleName, "mixed") ? MIXED : !strcmp(ruleName, "rene") ? RENE : !strcmp(ruleName, "paul") ? PAUL : -1;
  if (rule < 0) { fprintf(stderr, "RULE must be paul, rene or mixed\n"); return 1; }
  auto envInt = [](const char* k, int d) { const char* v = getenv(k); return v ? atoi(v) : d; };
  const int PRE = envInt("PRE", 0), WORKERS = envInt("WORKERS", 4);
  const long GAMES = argc > 1 && atol(argv[1]) ? atol(argv[1]) : 5000;
  const double ALPHA = argc > 2 && atof(argv[2]) ? atof(argv[2]) : 0.05;
  const double LAMBDA = argc > 3 && atof(argv[3]) ? atof(argv[3]) : 0.7;
  const int TURN_LIMIT = envInt("LIMIT", 200); // Mixed: longer games are decided by pip count (0 = off)
  const int MAX_PLIES = rule == MIXED ? 1500 : 3000;
  const long EVAL_EVERY = argc > 4 && atol(argv[4]) ? atol(argv[4]) : 500;
  const long LOG_EVERY = envInt("LOG", (int)std::max(1L, std::min(100L, EVAL_EVERY)));
  const double TIME = envInt("TIME", 0);

  // files live next to the executable (the project directory)
  char exe[4096];
  ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
  std::string dir = n > 0 ? std::string(exe, n) : std::string(argv[0]);
  dir = dir.substr(0, dir.rfind('/') + 1);
  const std::string FILE_W = dir + "weights_" + ruleName + ".json", METRICS = dir + "metrics_" + ruleName + ".js";
  std::string RULE_UP = ruleName;
  for (auto& c : RULE_UP) c = (char)toupper(c);

  const int HIDDEN = envInt("HIDDEN", 0);
  if (HIDDEN && !hiddenOk(HIDDEN)) { fprintf(stderr, "HIDDEN must be one of the compiled sizes (see BG_SIZES in core.h)\n"); return 1; }
  auto net = std::make_unique<Net>();
  net->nin = 246;
  std::mt19937_64 rnd(std::random_device{}());
  std::uniform_real_distribution<double> u(-0.1, 0.1);
  std::string wt = readFile(FILE_W);
  if (wt.size()) {
    auto W1 = jsonNumbers(wt, "W1"), b1 = jsonNumbers(wt, "b1"), W2 = jsonNumbers(wt, "W2"), b2 = jsonNumbers(wt, "b2");
    if (wt.find("\"W2b\"") != std::string::npos) { fprintf(stderr, "two-head nets are not supported\n"); return 1; }
    const int h = (int)b1.size();
    if (!hiddenOk(h) || (int)W1.size() != net->nin * h || (int)W2.size() != h || b2.size() != 1) { fprintf(stderr, "%s: unexpected sizes\n", FILE_W.c_str()); return 1; }
    if (HIDDEN && HIDDEN < h) { fprintf(stderr, "%s has %d hidden units, more than HIDDEN=%d\n", FILE_W.c_str(), h, HIDDEN); return 1; }
    net->h = HIDDEN ? HIDDEN : h;
    for (int i = 0; i < net->nin; i++)
      for (int j = 0; j < net->h; j++) net->W1[i * net->h + j] = j < h ? (float)W1[i * h + j] : (float)u(rnd);
    for (int j = 0; j < net->h; j++) { net->b1[j] = j < h ? (float)b1[j] : 0; net->W2[j] = j < h ? (float)W2[j] : 0; }
    net->b2 = (float)b2[0];
    if (net->h > h) printf("widened %s from %d to %d hidden units\n", FILE_W.c_str(), h, net->h);
  } else { // like bot.js newNet
    net->h = HIDDEN ? HIDDEN : 64;
    for (int i = 0; i < net->nin * net->h; i++) net->W1[i] = (float)u(rnd);
    for (int j = 0; j < net->h; j++) { net->b1[j] = 0; net->W2[j] = (float)u(rnd); }
    net->b2 = 0;
  }
  const Run R { rule, PRE, WORKERS, envInt("VSGREEDY", 1), envInt("LIMITDRAW", 0), envInt("SEARCH", 0), envInt("TRAIN_DEPTH", 2), getenv("EXPLORE") ? atof(getenv("EXPLORE")) : 0.0, getenv("TEMP") ? atof(getenv("TEMP")) : 0.0, getenv("TEMPEND") ? atof(getenv("TEMPEND")) : getenv("TEMP") ? atof(getenv("TEMP")) : 0.0, envInt("TEMPFIX", 0), GAMES, EVAL_EVERY, LOG_EVERY, ALPHA, LAMBDA, TIME, getenv("GAMMA") ? atof(getenv("GAMMA")) : 1.0, TURN_LIMIT, MAX_PLIES, dir, ruleName, RULE_UP, FILE_W, METRICS, net.get(), (long)envInt("REPLAY", 0), (long)envInt("KEEP", 100000), envInt("REPS", 8), getenv("ALPHAEND") ? atof(getenv("ALPHAEND")) : ALPHA };
  switch (net->h) {
#define BG_RUN(n) case n: return R.REPLAY ? runReplay<n>(R) : run<n>(R);
    BG_SIZES(BG_RUN)
  }
  return 1;
}

template <int H> static int run(const Run& R) {
  const int rule = R.rule, PRE = R.PRE, WORKERS = R.WORKERS, TURN_LIMIT = R.TURN_LIMIT, MAX_PLIES = R.MAX_PLIES;
  const long GAMES = R.GAMES, EVAL_EVERY = R.EVAL_EVERY, LOG_EVERY = R.LOG_EVERY;
  const double ALPHA = R.ALPHA, LAMBDA = R.LAMBDA, TIME = R.TIME;
  const std::string &dir = R.dir, &FILE_W = R.FILE_W, &METRICS = R.METRICS, &RULE_UP = R.RULE_UP;
  const char* ruleName = R.ruleName.c_str();
  Net* net = R.net;
  using Engine = bg::Engine<H>;
  using TD = bg::TD<H>;
  using Pos = bg::Pos<H>;

  // metrics: the rows written so far are kept as text, new rows appended
  std::string oldRows;
  long startGame = 0;
  {
    std::string m = readFile(METRICS);
    size_t a = m.find('['), b = m.rfind(']');
    if (a != std::string::npos && b != std::string::npos && b > a) oldRows = m.substr(a + 1, b - a - 1);
    size_t g = oldRows.rfind("\"game\":");
    if (g != std::string::npos) startGame = atol(oldRows.c_str() + g + 7);
  }
  std::vector<std::string> rows;
  auto save = [&]() {
    std::string w = toJSON(*net);
    writeFile(FILE_W, w);
    writeFile(dir + "weights_" + ruleName + ".js", "window.BOT_WEIGHTS_" + RULE_UP + "=" + w + ";\n");
    std::string all = oldRows;
    for (auto& r : rows) { if (!all.empty()) all += ","; all += r; }
    writeFile(METRICS, "window.METRICS_" + RULE_UP + "=[" + all + "];\n");
  };

  // Mixed games never end between aimless players, so there the yardstick is the greedy baseline.
  // Plays on a snapshot of the weights; the workers keep training meanwhile.
  auto evalEng = std::make_unique<Engine>();
  auto snap = std::make_unique<Net>();
  evalEng->rng.s = std::random_device{}() | 1;
  auto evalMatch = [&](int games, int& wins, int& played) {
    *snap = *net;
    evalEng->setRule(rule, 0, snap.get());
    const int cap = rule == MIXED ? 800 : 5000;
    Pos s;
    wins = played = 0;
    for (int i = 0; i < games; i++) {
      int bot = i % 2, rolls = 0, moves = 0;
      evalEng->reset(s);
      while (s.phase != OVER && rolls + moves < cap) {
        if (s.phase == ROLL) { int a = evalEng->rng.die(), b = evalEng->rng.die(); evalEng->roll(s, a, b); rolls++; }
        else if (s.turn == bot) evalEng->playBest(s, &moves);
        else if (rule == MIXED) evalEng->playGreedy(s, &moves);
        else evalEng->playRandom(s, &moves);
      }
      if (s.phase == OVER) { played++; if (s.winner == bot) wins++; }
    }
  };

  const Opts o { rule, PRE, TURN_LIMIT, MAX_PLIES, ALPHA, LAMBDA, startGame + GAMES, R.VSGREEDY != 0, R.GAMMA, R.LIMITDRAW != 0, R.SEARCH, R.EXPLORE, R.TEMP, R.TEMPFIX != 0 };
  std::atomic<long> counter { startGame };
  std::mutex mtx;
  std::condition_variable cv;
  struct Row { long game; double sq, plies; long steps, games, capped; bool eval; };
  std::deque<Row> queue;
  Row acc {};
  long done = 0;
  int finished = 0;
  const auto t0 = std::chrono::steady_clock::now();

  std::vector<std::thread> threads;
  std::random_device rd;
  for (int k = 0; k < WORKERS; k++) {
    uint64_t seed = ((uint64_t)rd() << 32 | rd()) | 1;
    threads.emplace_back([&, seed]() {
      auto g = std::make_unique<Engine>();
      auto td = std::make_unique<TD>();
      g->setRule(o.rule, o.rule == MIXED ? o.turnLimit : 0, net);
      g->rng.s = seed;
      g->moveDepth = R.TRAIN_DEPTH;
      g->needBest = o.tempFix;
      td->init(net, o.search ? 0 : o.lambda);
      Pos s;
      for (;;) {
        if (TIME > 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > TIME) break;
        long gi = ++counter; // global game number decides who plays
        if (gi > o.last) break;
        // After the warm-up, half the Mixed games are against the greedy baseline: aimless net-vs-net games
        // never end, and a net that sees no finished games learns that nothing matters.
        bool mix = o.rule == MIXED && o.vsGreedy && gi > o.pre && gi % 2 == 0;
        Opts og = o; // ALPHAEND: alpha falls linearly over TIME, as in replay mode
        if (TIME > 0) {
          double f = std::min(1.0, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / TIME);
          og.alpha = ALPHA + (R.ALPHAEND - ALPHA) * f;
          og.temp = R.TEMP + (R.TEMPEND - R.TEMP) * f;
        }
        g->temp = (float)og.temp;
        Result r = o.search ? searchGame<H>(*g, *td, s, og) : selfPlayGame<H>(*g, *td, s, gi <= o.pre ? 2 : mix ? (gi % 4 == 0 ? 0 : 1) : -1, og);
        std::lock_guard<std::mutex> lk(mtx);
        long i = ++done;
        acc.sq += r.sqErr; acc.steps += r.steps; acc.plies += r.plies; acc.games++; acc.capped += r.capped;
        if (i % LOG_EVERY == 0 || i == GAMES) {
          acc.game = startGame + i;
          acc.eval = i % EVAL_EVERY == 0 || i == GAMES;
          queue.push_back(acc);
          acc = Row {};
          cv.notify_one();
        }
      }
      std::lock_guard<std::mutex> lk(mtx);
      finished++;
      cv.notify_one();
    });
  }

  auto emit = [&](const Row& r) {
    double tdErr = r.sq / std::max(r.steps, 1L), plies = r.plies / r.games, capped = (double)r.capped / r.games;
    char buf[256];
    snprintf(buf, sizeof buf, "{\"game\":%ld,\"tdErr\":%.8g,\"plies\":%.8g,\"capped\":%.8g", r.game, tdErr, plies, capped);
    std::string row = buf, evalText;
    if (r.eval) {
      int games = rule == MIXED ? 20 : 100, wins, played;
      evalMatch(games, wins, played);
      if (played) snprintf(buf, sizeof buf, ",\"winRate\":%.8g,\"unfinished\":%d", (double)wins / played, games - played);
      else snprintf(buf, sizeof buf, ",\"winRate\":null,\"unfinished\":%d", games - played);
      row += buf;
      snprintf(buf, sizeof buf, "  vs %s: %.8g (%d unfinished)", rule == MIXED ? "greedy" : "random", played ? (double)wins / played : 0.0, games - played);
      evalText = played ? buf : std::string("  vs ") + (rule == MIXED ? "greedy" : "random") + ": null (" + std::to_string(games - played) + " unfinished)";
    }
    rows.push_back(row + "}");
    save();
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("game %ld  %.0fs  tdErr %.4f  plies %.0f  capped %.0f%%%s\n", r.game, sec, tdErr, plies, capped * 100, evalText.c_str());
    fflush(stdout);
  };
  for (;;) {
    Row r;
    {
      std::unique_lock<std::mutex> lk(mtx);
      cv.wait(lk, [&] { return !queue.empty() || finished == WORKERS; });
      if (queue.empty()) break;
      r = queue.front();
      queue.pop_front();
    }
    emit(r);
  }
  for (auto& t : threads) t.join();
  if (acc.games) { // stopped by TIME between two log rows
    acc.game = startGame + done;
    acc.eval = true;
    emit(acc);
  }
  save();
  return 0;
}

// ---------------------------------------------------------------- replay buffer
struct Rec { int8_t b[25]; uint8_t bar[2], off[2], turn; };
struct GameRec { std::vector<Rec> pos; float z = 0.5f; bool capped = false; };

// like selfPlayGame, but the net is frozen: every position before a roll is recorded instead of trained on
template <int H> static void recordGame(Engine<H>& g, Pos<H>& s, int greedySide, const Opts& o, GameRec& gr) {
  g.reset(s);
  gr.pos.clear();
  int plies = 0;
  while (s.phase != OVER && plies++ < o.maxPlies) {
    Rec r;
    for (int i = 0; i < 25; i++) r.b[i] = s.b[i];
    r.bar[0] = s.bar[0]; r.bar[1] = s.bar[1]; r.off[0] = s.off[0]; r.off[1] = s.off[1]; r.turn = s.turn;
    gr.pos.push_back(r);
    if (s.phase == ROLL) { int a = g.rng.die(), b = g.rng.die(); g.roll(s, a, b); }
    while (s.phase == MOVE || s.phase == WALK) {
      if (greedySide == 2 || s.turn == greedySide) g.playGreedy(s, nullptr);
      else if (o.explore > 0 && g.rng.next() < o.explore) g.playRandom(s, nullptr);
      else g.playBest(s, nullptr);
    }
  }
  gr.capped = s.phase != OVER;
  gr.z = gr.capped ? 0.5f : s.limitHit && o.limitDraw ? 0.5f : s.winner == 0 ? 1.f : 0.f;
  if (s.limitHit) gr.capped = true;
}

// position of a record with everything the net's inputs need (no dice, no accumulators)
template <int H> static void unpack(const Rec& r, Pos<H>& s) {
  for (int i = 0; i < 25; i++) s.b[i] = r.b[i];
  s.bar[0] = r.bar[0]; s.bar[1] = r.bar[1]; s.off[0] = r.off[0]; s.off[1] = r.off[1]; s.turn = r.turn; s.phase = ROLL;
  s.pip[0] = (int16_t)(25 * r.bar[0]); s.pip[1] = (int16_t)(25 * r.bar[1]);
  for (int a = 1; a <= 24; a++) { if (s.b[a] > 0) s.pip[0] += a * s.b[a]; if (s.b[a] < 0) s.pip[1] += (25 - a) * -s.b[a]; }
}

// One visit: lambda-returns of the whole game with the current net, then one pass of moving each position toward
// its return. Returns the sum of squared errors before the update and the number of positions.
template <int H> static void visitGame(Engine<H>& g, TD<H>& td, const GameRec& gr, double alpha, double lambda, double gamma, std::vector<float>& G, double& sq, long& n) {
  const int T = (int)gr.pos.size();
  static thread_local Pos<H> s;
  G.assign(T, 0.f);
  float next = gr.z, gnext = gr.z; // value and return of the position after t (White's chance)
  for (int t = T - 1; t >= 0; t--) {
    G[t] = (float)((1 - lambda) * next + lambda * gnext);
    gnext = G[t];
    unpack(gr.pos[t], s);
    double y = td.forward(g, s);
    double vw = s.turn == 0 ? y : 1 - y;
    next = (float)(0.5 + gamma * (vw - 0.5)); // what the position before t bootstraps from
  }
  for (int t = 0; t < T; t++) {
    unpack(gr.pos[t], s);
    double y = td.forward(g, s), tgt = s.turn == 0 ? G[t] : 1 - G[t];
    sq += (tgt - y) * (tgt - y); n++;
    td.trace(1);
    td.update(alpha * (tgt - y));
  }
}

template <int H> static int runReplay(const Run& R) {
  using Engine = bg::Engine<H>;
  using TD = bg::TD<H>;
  using Pos = bg::Pos<H>;
  const int rule = R.rule, WORKERS = R.WORKERS;
  Net* net = R.net;
  const std::string &dir = R.dir, &FILE_W = R.FILE_W, &METRICS = R.METRICS, &RULE_UP = R.RULE_UP;
  const char* ruleName = R.ruleName.c_str();
  std::string oldRows;
  long startGame = 0;
  {
    std::string m = readFile(METRICS);
    size_t a = m.find('['), b = m.rfind(']');
    if (a != std::string::npos && b != std::string::npos && b > a) oldRows = m.substr(a + 1, b - a - 1);
    size_t g = oldRows.rfind("\"game\":");
    if (g != std::string::npos) startGame = atol(oldRows.c_str() + g + 7);
  }
  std::vector<std::string> rows;
  auto save = [&]() {
    std::string w = toJSON(*net);
    writeFile(FILE_W, w);
    writeFile(dir + "weights_" + ruleName + ".js", "window.BOT_WEIGHTS_" + RULE_UP + "=" + w + ";\n");
    std::string all = oldRows;
    for (auto& r : rows) { if (!all.empty()) all += ","; all += r; }
    writeFile(METRICS, "window.METRICS_" + RULE_UP + "=[" + all + "];\n");
  };
  const Opts o { rule, R.PRE, R.TURN_LIMIT, R.MAX_PLIES, R.ALPHA, R.LAMBDA, 0, R.VSGREEDY != 0, R.GAMMA, R.LIMITDRAW != 0, 0, R.EXPLORE, 0, false };
  std::vector<GameRec> buf;
  buf.reserve(R.KEEP);
  long bufNext = 0, games = startGame;
  const auto t0 = std::chrono::steady_clock::now();
  auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  auto timeUp = [&]() { return R.TIME > 0 && elapsed() > R.TIME; };
  std::random_device rd;
  std::mutex mtx;
  for (int cycle = 1; !timeUp(); cycle++) {
    // 1. play with the frozen net
    const double tg0 = elapsed();
    std::atomic<long> made { 0 };
    long plies = 0, cappedN = 0;
    {
      std::vector<std::thread> th;
      for (int k = 0; k < WORKERS; k++) {
        uint64_t seed = ((uint64_t)rd() << 32 | rd()) | 1;
        th.emplace_back([&, seed]() {
          auto g = std::make_unique<Engine>();
          Pos s;
          g->setRule(rule, rule == MIXED ? o.turnLimit : 0, net);
          g->rng.s = seed;
          g->moveDepth = R.TRAIN_DEPTH;
          GameRec gr;
          for (;;) {
            if (timeUp()) break;
            long i = made++;
            if (i >= R.REPLAY) break;
            long gi = startGame + (long)cycle * R.REPLAY + i; // alternates greedy/self-play games as the online mode does
            bool mix = rule == MIXED && o.vsGreedy && gi % 2 == 0;
            recordGame<H>(*g, s, mix ? (gi % 4 == 0 ? 0 : 1) : -1, o, gr);
            std::lock_guard<std::mutex> lk(mtx);
            plies += (long)gr.pos.size(); cappedN += gr.capped;
            if ((long)buf.size() < R.KEEP) buf.push_back(gr); else buf[bufNext++ % R.KEEP] = gr;
          }
        });
      }
      for (auto& t : th) t.join();
    }
    const long nNew = std::min(made.load(), R.REPLAY);
    games += nNew;
    const double tg1 = elapsed();
    // 2. train on the buffer (Hogwild over the workers)
    std::atomic<long> visits { 0 };
    const long total = R.REPS * nNew;
    double sq = 0;
    long cnt = 0, vis = 0;
    {
      std::vector<std::thread> th;
      for (int k = 0; k < WORKERS; k++) {
        uint64_t seed = ((uint64_t)rd() << 32 | rd()) | 1;
        th.emplace_back([&, seed]() {
          auto g = std::make_unique<Engine>();
          auto td = std::make_unique<TD>();
          g->setRule(rule, 0, net);
          td->init(net, 0);
          Rng rng; rng.s = seed;
          std::vector<float> G;
          double lsq = 0; long ln = 0, lv = 0;
          while (!timeUp()) {
            if (visits++ >= total) break;
            double alpha = R.TIME > 0 ? R.ALPHA + (R.ALPHAEND - R.ALPHA) * std::min(1.0, elapsed() / R.TIME) : R.ALPHA;
            const GameRec& gr = buf[(size_t)(rng.next() * buf.size())];
            visitGame<H>(*g, *td, gr, alpha, R.LAMBDA, R.GAMMA, G, lsq, ln); lv++;
          }
          std::lock_guard<std::mutex> lk(mtx);
          sq += lsq; cnt += ln; vis += lv;
        });
      }
      for (auto& t : th) t.join();
    }
    const double tg2 = elapsed();
    char row[256];
    snprintf(row, sizeof row, "{\"game\":%ld,\"tdErr\":%.8g,\"plies\":%.8g,\"capped\":%.8g}", games, sq / std::max(cnt, 1L), (double)plies / std::max(nNew, 1L), (double)cappedN / std::max(nNew, 1L));
    rows.push_back(row);
    save();
    printf("cycle %d  %.0fs  games %ld (buffer %zu)  play %.0fs (%.0f games/s)  train %.0fs (%ld visits)  tdErr %.4f  plies %.0f  capped %.0f%%\n", cycle, tg2, games, buf.size(),
           tg1 - tg0, nNew / std::max(tg1 - tg0, 1e-9), tg2 - tg1, vis, sq / std::max(cnt, 1L), (double)plies / std::max(nNew, 1L), 100.0 * cappedN / std::max(nNew, 1L));
    fflush(stdout);
  }
  save();
  return 0;
}
