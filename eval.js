// node eval.js [games]  -- trained net vs random and vs a greedy pip/hit heuristic
const fs = require('fs');
const BG = require('./engine.js');
const L = require('./bot.js');
const RULE = process.env.RULE || 'paul';
const N = +process.argv[2] || 400;
const net = new L.Net(JSON.parse(fs.readFileSync(__dirname + '/weights_' + RULE + '.json')));
const bot = L.netBot(net);

const greedy = {
  move(s) {
    const p = s.turn;
    let best = null, bv = -1e9;
    for (const m of BG.allMoves(s)) {
      let v = (m.back ? -m.die : m.die) + (m.to === 'off' ? 6 : 0);
      if (m.to !== 'off' && s.board[BG.rel(p, m.to)] * BG.sgn(p) === -1) v += 15;
      v += Math.random() * 0.1;
      if (v > bv) { bv = v; best = m; }
    }
    return best;
  },
};
function match(a, b, n) {
  let w = 0, done = 0;
  for (let g = 0; g < n; g++) {
    const aw = g % 2 === 0;
    const r = L.playGame(aw ? [a, b] : [b, a], RULE);
    if (r >= 0) { done++; if ((r === 0) === aw) w++; }
  }
  return `${w}/${done} (${n - done} unfinished)`;
}
console.log('net vs random:', match(bot, L.randomBot, N));
console.log('net vs greedy:', match(bot, greedy, N));
console.log('greedy vs random:', match(greedy, L.randomBot, N));
