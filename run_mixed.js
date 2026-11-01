// Timed Mixed training run with in-between strength reports against a fixed champion:
// ./bgtrain trains in chunks; after each chunk match.js plays the current net against the champion on
// shared dice and prints Elo with a 95% interval. The reports use EVAL_FRAC of the total time (default 5%).
//   TOTAL=600 EVAL_FRAC=0.05 ROUNDS=6 WORKERS=8 [TRAIN_DEPTH=2] [OLD_DIR=../Bagammon] node run_mixed.js [alpha] [lambda]
// Resumes from weights_mixed.json like bgtrain does; the champion is OLD_DIR/weights_mixed.json.
const { spawnSync } = require('child_process');
const fs = require('fs');

const TOTAL = +process.env.TOTAL || 600, FRAC = process.env.EVAL_FRAC !== undefined ? +process.env.EVAL_FRAC : 0.05, ROUNDS = +process.env.ROUNDS || 6;
const WORKERS = process.env.WORKERS || '8', alpha = process.argv[2] || '0.05', lambda = process.argv[3] || '0.7';
const trainSecs = TOTAL * (1 - FRAC) / ROUNDS, evalSecs = TOTAL * FRAC / ROUNDS;
const env = Object.assign({}, process.env, { RULE: 'mixed', WORKERS, TIME: String(trainSecs), LOG: '1000' });
const t0 = Date.now(), lastGame = () => {
  const m = fs.readFileSync('metrics_mixed.js', 'utf8').match(/"game":(\d+)[^{]*\}\];/);
  return m ? +m[1] : 0;
};
const startGame = lastGame();
for (let r = 1; r <= ROUNDS; r++) {
  spawnSync('./bgtrain', ['100000000', alpha, lambda, '100000000'], { env, stdio: ['ignore', 'ignore', 'inherit'] });
  const m = spawnSync('node', ['match.js', '0', WORKERS], { env: Object.assign({}, process.env, { SECS: String(evalSecs), QUIET: '1' }), encoding: 'utf8' });
  const line = (m.stdout || '').trim().split('\n').pop();
  console.log(`round ${r}/${ROUNDS}  t ${((Date.now() - t0) / 1000).toFixed(0)}s  games ${lastGame() - startGame} (total ${lastGame()})  ${line}`);
}
