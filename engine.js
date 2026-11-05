// Rules engine for the Bagammon variants. No DOM access, so a bot can use it later
// (browser: global BG, node: require).
//
// Coordinates: board[a], a = 1..24, absolute points. Signed stone count: + White (player 0),
// - Black (player 1). White moves 24 -> 1, Black moves 1 -> 24. "Relative" point r is the
// point as seen by the mover (r = distance to bear off): rel(0,a) = a, rel(1,a) = 25 - a.
// Home quadrant is r = 1..6, entry from the bench with die d lands on r = 25 - d.
(function (root) {
  'use strict';

  const MAX_STACK = 5;
  const STONES = 15;

  const RULES = {
    paul:  { name: "Paul's rules", doubleMoves: 2, rollAgain: true,  nonExact: true,  mixed: false },
    rene:  { name: "Rene's rules", doubleMoves: 4, rollAgain: true,  nonExact: true,  mixed: false },
    mixed: { name: 'Mixed rules',  doubleMoves: 2, rollAgain: false, nonExact: true , mixed: true  },
  };
  const PLAYER_NAMES = ['White', 'Black'];

  const rel = (p, x) => (p === 0 ? x : 25 - x);
  const sgn = (p) => (p === 0 ? 1 : -1);
  const ownAt = (s, p, r) => Math.max(0, s.board[rel(p, r)] * sgn(p));

  // opts.plyLimit (turns, 0 = off): a game that reaches it is a draw (phase 'over', winner null).
  // opts.limitDraw = false: the old rule, the side further ahead in pip count wins.
  function newGame(rule, opts) {
    const s = {
      rule, plyLimit: (opts && opts.plyLimit) || 0, limitDraw: !(opts && opts.limitDraw === false), turnCount: 0, limitHit: false, board: Array(25).fill(0), bar: [0, 0], off: [0, 0], turn: 0, phase: 'roll',
      dice: [], throw: null, pending: null, rolledDouble: false, winner: null, log: [], rolls: 0,
    };
    for (const [r, n] of [[24, 2], [13, 5], [8, 3], [6, 5]]) {
      s.board[rel(0, r)] += n;
      s.board[rel(1, r)] -= n;
    }
    return s;
  }

  // v > 0: own stones on the target, v < 0: enemy stones
  function canLand(s, p, r) {
    const v = s.board[rel(p, r)] * sgn(p);
    return v > -2 && v < MAX_STACK;
  }

  function highest(s, p) {
    for (let r = 24; r >= 1; r--) if (ownAt(s, p, r)) return r;
    return 0;
  }

  const allHome = (s, p) => s.bar[p] === 0 && highest(s, p) <= 6;

  function entryBlocked(s, p) {
    for (let d = 1; d <= 6; d++) if (canLand(s, p, 25 - d)) return false;
    return true;
  }

  // Legal single-die moves for player p. from/to are relative; to === 'off' bears off.
  // A move is legal on its own; there is no obligation to maximise the number of dice used.
  // Mixed: q has no legal move with any die, by the plain rules (their own backward moves included)
  function frozen(s, q) {
    for (let d = 1; d <= 6; d++) if (movesFor(s, q, d, true).length) return false;
    return true;
  }

  // Mixed: no backward move while the opponent cannot move with any throw (plain = true: without that rule)
  function movesFor(s, p, d, plain) {
    const R = RULES[s.rule];
    const out = [];
    if (s.bar[p] > 0) {
      if (canLand(s, p, 25 - d)) out.push({ from: 'bar', to: 25 - d, die: d });
      return out;
    }
    const home = allHome(s, p);
    const top = highest(s, p);
    const canBack = R.mixed && s.off[0] === 0 && s.off[1] === 0 && s.bar[1 - p] === 0; // no backward walking once any stone is off, or while any stone is benched
    let fz = null; // the opponent is frozen (computed when first needed)
    // Mixed: bearing off is mandatory when the die equals the point of one of your stones
    if (R.mixed && home && ownAt(s, p, d)) return [{ from: d, to: 'off', die: d }];
    for (let r = 1; r <= 24; r++) {
      if (!ownAt(s, p, r)) continue;
      const t = r - d;
      if (t >= 1) {
        if (canLand(s, p, t)) out.push({ from: r, to: t, die: d });
      } else if (home && (t === 0 || (R.nonExact && r === top))) {
        out.push({ from: r, to: 'off', die: d });
      }
      if (canBack && r > 6 && r + d <= 24 && canLand(s, p, r + d)) {
        if (fz === null && !plain) fz = frozen(s, 1 - p);
        if (plain || !fz) out.push({ from: r, to: r + d, die: d, back: true });
      }
    }
    return out;
  }

  function allMoves(s) {
    if (s.phase !== 'move' && s.phase !== 'walk') return [];
    const out = [];
    for (const d of new Set(s.dice)) out.push(...movesFor(s, s.turn, d));
    return out;
  }

  const expand = (t, R) => (t[0] === t[1] ? Array(R.doubleMoves).fill(t[0]) : [t[0], t[1]]);

  function roll(s, dice) {
    if (s.phase !== 'roll') return false;
    const R = RULES[s.rule];
    const t = dice || [1 + Math.floor(Math.random() * 6), 1 + Math.floor(Math.random() * 6)];
    s.throw = t;
    s.rolls = (s.rolls || 0) + 1;
    s.rolledDouble = t[0] === t[1];
    if (R.mixed) s.pending = t.slice();
    s.dice = expand(t, R);
    s.phase = 'move';
    s.log.push(`${PLAYER_NAMES[s.turn]} rolls ${t[0]}-${t[1]}`);
    settle(s);
    return true;
  }

  function applyMove(s, m) {
    const p = s.turn;
    const i = s.dice.indexOf(m.die);
    if (i < 0) return false;
    s.dice.splice(i, 1);
    if (m.from === 'bar') s.bar[p]--;
    else s.board[rel(p, m.from)] -= sgn(p);
    let note = '';
    if (m.to === 'off') {
      s.off[p]++;
    } else {
      const a = rel(p, m.to);
      if (s.board[a] * sgn(p) === -1) {
        s.board[a] = 0;
        s.bar[1 - p]++;
        note = ' (hit)';
      }
      s.board[a] += sgn(p);
    }
    const arrow = m.back ? '<-' : '->';
    s.log.push(`${PLAYER_NAMES[p]} ${m.from === 'bar' ? 'bar' : m.from} ${arrow} ${m.to === 'off' ? 'off' : m.to}${note}`);
    settle(s);
    return true;
  }

  // After every change: detect a win, drop dice that can no longer be played, advance.
  function settle(s) {
    const p = s.turn;
    if (s.off[p] === STONES) {
      s.winner = p;
      s.phase = 'over';
      s.dice = [];
      s.log.push(`${PLAYER_NAMES[p]} wins`);
      return;
    }
    if (!s.dice.some((d) => movesFor(s, p, d).length)) {
      if (s.dice.length) s.log.push(`${PLAYER_NAMES[p]} cannot play ${s.dice.join(',')}`);
      s.dice = [];
      finishSegment(s);
    }
  }

  function finishSegment(s) {
    const R = RULES[s.rule];
    if (s.phase === 'walk') { s.phase = 'roll'; return; }
    if (R.rollAgain && s.rolledDouble) { s.phase = 'roll'; return; } // the same player must roll again
    nextTurn(s);
  }

  const pip = (s, p) => {
    let n = 25 * s.bar[p];
    for (let r = 1; r <= 24; r++) n += r * ownAt(s, p, r);
    return n;
  };

  function nextTurn(s) {
    const R = RULES[s.rule];
    const q = 1 - s.turn;
    if (s.plyLimit && ++s.turnCount >= s.plyLimit) {
      const a = pip(s, 0), b = pip(s, 1);
      s.winner = s.limitDraw === false ? (a < b ? 0 : a > b ? 1 : s.turn) : null; // pip rule, equal: the side that just moved
      s.phase = 'over';
      s.dice = [];
      s.limitHit = true;
      s.log.push(s.winner === null ? `Turn limit: draw (pips ${a} vs ${b})` : `Turn limit: ${PLAYER_NAMES[s.winner]} wins on pip count (${a} vs ${b})`);
      return;
    }
    s.turn = q;
    s.rolledDouble = false;
    s.dice = [];
    s.phase = 'roll';
    if (!R.mixed) return;
    if (s.bar[q] > 0 && entryBlocked(s, q)) {
      // benched with no opening: q does not throw, the opponent throws again
      s.log.push(`${PLAYER_NAMES[q]} is benched with no opening and is skipped`);
      s.turn = 1 - q;
      s.pending = null;
      return;
    }
    if (s.pending) {
      s.dice = expand(s.pending, R);
      s.pending = null;
      s.phase = 'walk';
      s.log.push(`${PLAYER_NAMES[q]} walks ${s.dice.join(',')}`);
      settle(s);
    }
  }

  function resign(s, p) {
    if (s.phase === 'over') return false;
    s.winner = 1 - p;
    s.phase = 'over';
    s.dice = [];
    s.log.push(`${PLAYER_NAMES[p]} resigns`);
    return true;
  }

  const api = { pip, resign, RULES, PLAYER_NAMES, MAX_STACK, STONES, rel, sgn, ownAt, newGame, movesFor, allMoves, roll, applyMove, entryBlocked };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.BG = api;
})(typeof window !== 'undefined' ? window : globalThis);
