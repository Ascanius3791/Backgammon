# Bagammon: rules for every session

## Document how nets were trained (NETS.md)

Every champion (the small Mixed champion `weights_mixed.json`, the big champion `weights_mixed_big.json`, and any
net that replaces one of them) must have an entry in [NETS.md](NETS.md). Write or update the entry in the same
commit that changes the weights file.

When the user asks to "document a net's training" (or similar), add an entry to NETS.md for that net too. An entry
must make the training reproducible and the claim checkable:

- **File** and the commit that set it, and the **date**.
- **Start**: the net it was trained from (file + commit), or "from scratch"; architecture (inputs, hidden units,
  `DEEP` residual layers).
- **Training**, per stage: the exact command line with all environment variables (`RULE`, `WORKERS`, `REPLAY`,
  `EXPLORE`, alpha/`ALPHAEND`, `LIMIT`/`LIMITDRAW`, `DEEP`, ...), anything that differed from the defaults of that
  time, the **wall time**, and the number of **games** trained.
- **Result**: Elo ± 1σ, the pair counts (won 2-0 / lost 0-2 / split), how many pairs, and **against what reference**
  (file + commit). Matches follow the match protocol: `cpp/match.cpp`, paired games with shared dice, colours
  swapped, from the normal start unless stated, and say whether a game cut at the limit counted as a draw.
- Controls or comparisons that were run alongside, and failed variants worth knowing about (briefly).

## Turn limit

The turn limit (`LIMIT`, 200 turns) exists for training and test matches only, and a game that reaches it is a
draw (`LIMITDRAW=1`, the default in bgtrain and match.cpp). The GUI plays without a limit and does not know about it.
