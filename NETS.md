# Nets and how they were trained

Rules for this file: see [CLAUDE.md](CLAUDE.md). Newest entries first within each net. Elo is from
`cpp/match.cpp` (1-ply play, paired games with shared dice and colours swapped, normal start unless stated,
LIMIT 200), ± 1σ.

## Big champion — `weights_mixed_big.json` (GUI: "Big champion", Mixed)

Architecture: 246 inputs → 64 tanh hidden units → **2 residual layers** of 64 (h ← h + tanh(Wd·h + bd)) → 1 sigmoid
output (plus the fixed pip-count bias of Mixed). About 5× slower per evaluation than the small champion.

### v2 (current) — 2026-11-02

- **Start**: big champion v1 (below, commit 9acb3ac).
- **Training**: replay mode, constant alpha 0.01, 8 workers, 10 min wall time, 70,609 games. A game cut at the turn
  limit now counted as a **draw** (`LIMITDRAW=1`, which became the default with this run):
  `RULE=mixed WORKERS=8 REPLAY=10000 EXPLORE=0.02 TIME=600 ./bgtrain 100000000 0.01`
  (defaults: lambda 0.7, LIMIT 200, VSGREEDY 1, KEEP 100000, REPS 8).
  This ended the stalling seen under the old rule (pip leader wins at the limit): games cut at the limit fell from
  37–39% to 0–1%, game length from ~120 to ~58 plies, and training ran at ~330 instead of ~50 games/s.
- **Result** (2500 pairs each, limit = draw):
  - vs the small champion `weights_mixed.json` (bc9d952): **+50.1 ± 4.9 Elo** (won 809, lost 452, split 1239).
  - vs big champion v1 (9acb3ac): **+32.5 ± 4.8 Elo** (707 / 475 / 1318).
  - vs a sibling trained from v1 with a speed bonus (`SPEEDBONUS=0.1`, 4 workers, alpha 0.01, 10 min, old limit
    rule): +20.7 ± 4.8 Elo (671 / 524 / 1305). That bonus net itself was only +5.4 ± 4.9 vs v1, and the bonus did
    not stop the stalling (37–39% of its training games were still cut at the limit), so it was dropped.

### v1 — 2026-11-01 (commit 9acb3ac)

- **Start**: the small champion `weights_mixed.json` at commit bc9d952, deepened with `DEEP=2`: the two new layers
  start at Wd = bd = 0, which is exactly the identity. Check: 500 pairs vs the small champion were all split.
- **Stage 1**: replay mode, alpha 0.05 → 0.01 (linear over TIME), 4 workers on cores 0-3, 10 min, 40,000 games.
  Old limit rule (the pip leader wins at the limit).
  `RULE=mixed WORKERS=4 REPLAY=10000 EXPLORE=0.02 ALPHAEND=0.01 TIME=600 taskset -c 0-3 ./bgtrain 100000000 0.05`
  Result (3000 pairs): −6.5 ± 4.5 Elo vs the small champion (732 / 788 / 1480). A control with the same recipe
  without the extra layers (4 workers on cores 4-7, 200,000 games) reached +8.5 ± 4.5; deep vs control −0.8 ± 4.4.
- **Stage 2**: continued from stage 1 with the same recipe, restarted at alpha 0.05 → 0.01, 8 workers, 20 min wall
  time (TIME=1200, including ~1 min of pauses for snapshot matches), 100,000 games.
  `RULE=mixed WORKERS=8 REPLAY=10000 EXPLORE=0.02 ALPHAEND=0.01 TIME=1200 ./bgtrain 100000000 0.05`
  Snapshots vs the small champion (2500 pairs each): −27.9 (4 min), −31.5 (8 min; the restarted alpha 0.05 costs
  strength at first), −1.3 (12 min), +11.8 (17 min), and at the end
  **+26.6 ± 4.9 Elo** (won 706, lost 515, split 1279). Scored with limit = draw it was +28.1 ± 4.8, without a
  limit +27.4 ± 4.8.
- In total: 30 min of training, 140,000 games.

## Small champion — `weights_mixed.json` (GUI: "Main line (champion)", Mixed, the default)

Architecture: 246 → 64 tanh → 1. Reconstructed from commit messages and notes (written before this file existed):

- Commit bc9d952 (2026-11-01): 9 min of replay-buffer training from the previous champion (the one at commit
  5935296, still in `champion/weights_mixed.json`): WORKERS 4, REPLAY 10000, EXPLORE 0.02, alpha 0.05 → 0.01, old
  limit rule. Result: +34 Elo (15..54) vs the previous champion over 1200 games on shared dice.
- Commit 5935296: a 15-min extension of the one-net design, +88 Elo (78..98) vs the older 253-input champion over
  4800 games.

## Wide openings — `weights_mixed_openings.json` (GUI alternative, Mixed)

- **File**: commit c559e3e, 2026-11-01. Written by the session that trained it.
- **Start**: the small champion `weights_mixed.json` at commit bc9d952. Architecture 246 → 64 tanh → 1, no `DEEP`.
- **Training** (bgtrain at commit eef7f46, which added `OPENING`; old limit rule: a game cut at 200 turns was won by
  the pip leader):
  `RULE=mixed WORKERS=4 TIME=540 OPENING=10 REPLAY=10000 KEEP=100000 REPS=8 EXPLORE=0.02 ALPHAEND=0.01 ./bgtrain 100000000 0.05 0.7 100000000`
  Every game starts with k random segments (throws and walks, either side), k uniform in 0..10. They are not
  recorded or trained on. Wall time 540 s, 256,395 games (26 cycles).
- **Control**, run in parallel on the other 4 cores: the same command without `OPENING`, 267,532 games.
- **Result**, all against the small champion at bc9d952, with paired games (shared dice, colours swapped), 1-ply play,
  limit 200 with the old pip-leader rule (0 games cut):
  - From the normal start, 3000 pairs: **−5.4 ± 4.4 Elo** (control −2.3 ± 4.5; Wide openings vs control −7.9 ± 4.5).
    Pair counts weren't printed by match.cpp yet.
  - From shared random openings (`OPENING=10` in cpp/match.cpp), 5000 pairs: **+33.3 ± 3.5 Elo** (won 1494, lost
    1016, split 2490). Control −0.2 ± 3.4 (1205 / 1208 / 2587). Wide openings vs control +30.3 ± 3.5 (1457 / 1022 / 2521).
  - 3-2 opening at depth 1: 13/8 went from 3.6% to 34.5%, and depth 1 now nearly agrees with depth 2. Its favourite
    3-2 is 13/10 8→10 back (52.7%) instead of 24/22 24/21.
- Not a champion: from the normal start it is not better.

## Hard positions — `weights_mixed_hard.json` (GUI alternative, Mixed)

Architecture: 246 → 64 tanh → 1. Added to the GUI in commit 1b43942, 2026-11-02.

- **Start**: the small champion `weights_mixed.json` (bc9d952).
- **Disagreement set B**: the champion and Wide openings (`weights_mixed_openings.json`, c559e3e) played 1-ply
  games for 5 s on 8 threads, 4519 games, from the normal start. Every position before a roll was scored by both
  nets (depth 1) and duplicates were merged (exact board, bar, off and side to roll): 252,676 distinct positions.
  The 2000 with the largest |vA − vB| make up B. Their gaps are 0.63–0.85 in win probability, and the champion was
  the higher one in 920 of them. Built with
  `TIME=5 SEED=1 ./disagree weights_mixed.json weights_mixed_openings.json B.txt 2000` (cpp/disagree.cpp, commit 4a6bff0).
  That disagree binary predated the draw at the turn limit in core.h, which only mattered in games that reached it.
- **Training**: replay mode, alpha 0.05 → 0.01, 4 workers (beside the control on the other 4), 9 min, 220,000
  games. A game started from a random B position with chance 0.1: 9.7–10.1% of games and 10.3–10.8% of positions
  came from B starts. A game cut at the limit was a draw (`LIMITDRAW=1`, the default):
  `RULE=mixed WORKERS=4 REPLAY=10000 EXPLORE=0.02 ALPHAEND=0.01 TIME=540 BSET=B.txt BFRAC=0.1 ./bgtrain 100000000 0.05`
- **Control**: the same command without BSET/BFRAC, 210,000 games.
- **Result**: 3000 pairs each, limit = draw. Three kinds of start: the normal start, shared random openings
  (`OPENING=10`), and a held-out set B′ (`START=` in match.cpp). B′ holds 1000 top-disagreement positions from
  champion vs Wide openings games with another seed (`SEED=2`), with every position of B removed.

  | Hard positions vs | normal start | random openings | held-out B′ |
  |---|---|---|---|
  | champion `weights_mixed.json` (bc9d952) | −6.3 ± 4.4 (703 / 752 / 1545) | **+23.9 ± 4.4** (829 / 627 / 1544) | **+22.6 ± 4.4** (805 / 604 / 1591) |
  | control | −10.2 ± 4.4 (669 / 753 / 1578) | +18.3 ± 4.4 (778 / 625 / 1597) | +19.5 ± 4.4 (808 / 641 / 1551) |
  | Wide openings (c559e3e) | +22.9 ± 4.5 | −14.1 ± 4.4 | −10.7 ± 4.4 |

  The control vs the champion scored −9.4 / −0.1 / +3.0, and the champion vs Wide openings +3.4 / −36.6 / −35.3.
- Not a champion: from the normal start it is not better. It is meant for analysing games that leave the main line.
- **Failed continuation** (not kept): 30 more min from this net, in 6 rounds of 5 min, each a new bgtrain run with
  constant alpha 0.01 and B rebuilt before every round from the current net vs Wide openings (`SEED=101..106`).
  Those B sets had gaps of only 0.19–0.78, mean ≈ 0.3. The control was the same without B, from the control above.
  Result: −27.1 / −21.2 / −15.4 vs this net and −23.7 / −4.5 / −7.9 vs its control; the control itself fell
  −13.2 / −9.0 / +2.9 against its start. Each restart empties the replay buffer, which probably costs strength. Its
  matches had 4× as many games cut at the limit (training plies 54 → 76–83), so the continued nets started to stall.

## Accuracy head — `weights_mixed_sigma.json` (GUI alternative, Mixed) — 2026-11-05

Architecture: 246 → 64 tanh → 1, plus an accuracy head on the hidden layer (64 → 32 tanh → 1, core.h `Net::S1..`):
log σ² = the predicted squared error of the net's value of a position before a roll against a 2-roll search
(`node(s, side, 2)`). The head never feeds the value. The GUI shows σ per move in the analysis (as ±, for the
position the 1-ply value scores, after the opponent's best walk; bot.js `sigmaFor`, equal to core.h to 1e-5).

- **Start**: the small champion `weights_mixed.json` (bc9d952); the head started fresh (S1 ~ U(−0.2, 0.2), S2 = 0,
  Sb2 = log 0.002).
- **Training** (branch worktree-sigma-head, bgtrain of commit e1f1960), replay mode, 2 workers on
  2 CPUs, 220 s, 108,243 games (11 cycles). A game cut at the limit was a draw (`LIMITDRAW=1`):
  `RULE=mixed WORKERS=2 REPLAY=10000 EXPLORE=0.02 ALPHAEND=0.01 EXPFIX=1 EXPMODE=1 TIME=220 ./bgtrain 100000000 0.05`
  - `EXPMODE=1`: the 2% exploring moves take the end with the largest predicted σ instead of a random one.
  - `EXPFIX=1`: Watkins cut, so an explored move does not teach the earlier positions; they learn the value of the
    greedy move instead.
  - `SIGFRAC=0.005` (the default with EXPMODE): 0.5% of the visited positions train the head on their depth-2 error
    (`SIGALPHA=0.002`); head loss −6.3 → −7.5 (a constant σ scores about −5.2).
- **Result**: 10,000 pairs each from the normal start, limit = draw:

  | Accuracy head vs | Elo | won 2-0 / lost 0-2 / split |
  |---|---|---|
  | small champion `weights_mixed.json` (bc9d952) | +11.7 ± 2.4 | 2521 / 2195 / 5284 |
  | big champion `weights_mixed_big.json` (f11f256) | −32.1 ± 2.4 | 1940 / 2845 / 5215 |

- **Earlier runs of the same idea** (2026-11-04, 9 min with 4 runs sharing the machine, so about 92,000 games each;
  10,000 pairs vs the small champion): this recipe +10.9 ± 2.4; the plain recipe (control) +8.1, B vs control +9.5;
  `EXPFIX=1` with random exploration −36.8 (unexplained); `EXPMODE=2` (value + 5σ) −6.6. The same recipe for 20 min
  (488,000 games) was worse: −12.6 vs the small champion, −55.3 vs the big champion.
- Not a champion: +11.7 is within the run-to-run spread of the plain recipe. It is in the GUI for its σ.

## Value of the start position (Mixed) — 2026-11-02

P(White wins), White to roll from the normal start, measured with `cpp/firstply.cpp` for both champions
(small: `weights_mixed.json` at commit bc9d952; big v2: `weights_mixed_big.json` at commit f11f256):
`./firstply ../weights_mixed.json 3 72000 8` (run from the repo root with the binary built in cpp/).

| | Small champion | Big champion v2 |
|---|---|---|
| Net value of the start position | 0.5027 | 0.5276 |
| 1 ply: White's best move averaged over the 36 throws (Black's walk of the throw searched) | 0.4987 | 0.4859 |
| Also searching Black's next throw (`node` depth 3) | 0.5029 | 0.5101 |
| Rollouts, 72,000 games, 1-ply bot both sides, first throw stratified, limit 200 = draw (½) | 0.5052 ± 0.0019 | 0.5008 ± 0.0019 |
| Games cut at the limit in those rollouts | 1.24% | 0.08% |

So White starts at about an even 50%; the nets' static values are within ~±0.03 of that. Rollout speed on
8 threads: ~800 games/s (small), ~250/s (big); ±0.001 needs ~250k games. Luck-based variance reduction (`VR=1`)
was unbiased but cut the variance only ~3.5× at ~25× the cost per game, so it does not pay.

The lookahead weights the throws by probability: 21 distinct throws, a double 1/36, a non-double 2/36
(`Engine::node` in core.h, `ROLLS` in analysis.js).
