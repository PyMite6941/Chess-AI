// Check the in-browser JS encoder matches board.py exactly.
// Run:  node parity_check.mjs   (after: python parity_check.py)
// Loads chess.js from the ai-lab app so it's the SAME version the demo uses.
import { readFileSync } from 'fs';
import { createRequire } from 'module';

const require = createRequire('C:/Users/gresh/OneDrive/ドキュメント/portfolio/ai-lab/');
const { Chess } = require('chess.js');

// --- EXACT copy of boardToTensor from ChessDemo.js (keep in lockstep) ---
const TYPE_IDX = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };
const N_PLANES = 19;
function fill(t, plane, v) { for (let i = 0; i < 64; i++) t[plane * 64 + i] = v; }
function boardToTensor(game) {
  const t = new Float32Array(N_PLANES * 64);
  const rows = game.board();
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const sq = rows[r][c];
      if (!sq) continue;
      const plane = TYPE_IDX[sq.type] + (sq.color === 'w' ? 0 : 6);
      t[plane * 64 + (7 - r) * 8 + c] = 1.0;
    }
  }
  const [, turn, castling, ep, halfmove] = game.fen().split(' ');
  if (turn === 'w') fill(t, 12, 1.0);
  if (castling.includes('K')) fill(t, 13, 1.0);
  if (castling.includes('Q')) fill(t, 14, 1.0);
  if (castling.includes('k')) fill(t, 15, 1.0);
  if (castling.includes('q')) fill(t, 16, 1.0);
  if (ep && ep !== '-') {
    const file = ep.charCodeAt(0) - 97;
    const rank = ep.charCodeAt(1) - 49;
    t[17 * 64 + rank * 8 + file] = 1.0;
  }
  fill(t, 18, Math.min(parseInt(halfmove || '0', 10), 100) / 100.0);
  return t;
}

const data = JSON.parse(readFileSync('parity_fens.json', 'utf8'));
let mismatchCases = 0;
const planeMismatch = new Array(N_PLANES).fill(0);
let firstBad = null;

for (const { fen, t: ref } of data.cases) {
  const got = boardToTensor(new Chess(fen));
  let bad = false;
  for (let i = 0; i < ref.length; i++) {
    if (Math.abs(got[i] - ref[i]) > 1e-6) {
      planeMismatch[Math.floor(i / 64)]++;
      bad = true;
      if (!firstBad) firstBad = { fen, idx: i, plane: Math.floor(i / 64), got: got[i], ref: ref[i] };
    }
  }
  if (bad) mismatchCases++;
}

console.log(`checked ${data.cases.length} positions, ${N_PLANES} planes`);
console.log(`mismatched positions: ${mismatchCases}`);
console.log('per-plane mismatched cells:',
  planeMismatch.map((m, p) => m ? `p${p}=${m}` : null).filter(Boolean).join(' ') || 'NONE');
if (mismatchCases === 0) {
  console.log('PARITY OK — JS encoder matches board.py exactly.');
} else {
  console.log('PARITY FAILED. first mismatch:', JSON.stringify(firstBad));
  process.exit(1);
}
