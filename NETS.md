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
