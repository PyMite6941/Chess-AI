// Prove the JS explainer (ai-lab/app/components/tactics.js) produces the SAME
// wording as the Python one (tactics.py), so explanations are identical anywhere.
//   python -c "...writes tactics_cases.json..."   then   node tactics_parity.mjs
import { readFileSync } from 'fs';

const tacticsUrl = new URL(
  'file:///C:/Users/gresh/OneDrive/ドキュメント/portfolio/ai-lab/app/components/tactics.js');
const { explainMove } = await import(tacticsUrl);

const cases = JSON.parse(readFileSync('tactics_cases.json', 'utf8'));
let bad = 0;
for (const { fen, uci, explanation } of cases) {
  const got = explainMove(fen, uci);
  if (got !== explanation) {
    bad++;
    if (bad <= 8) {
      console.log('MISMATCH');
      console.log('  fen :', fen);
      console.log('  py  :', explanation);
      console.log('  js  :', got);
    }
  }
}
console.log(`\nchecked ${cases.length} explanations, ${bad} mismatched`);
if (bad === 0) console.log('EXPLANATION PARITY OK — JS wording matches tactics.py exactly.');
else process.exit(1);
