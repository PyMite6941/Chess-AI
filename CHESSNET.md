# ChessNet — Trained Model + In-Browser Demo

End-to-end guide for the Chess AI: how the model is trained, exported, deployed, and
played against in the browser on the AI Lab. Written 2026-07 while building it.

## What it is

A ResNet-style **policy/value network** (AlphaZero-style) trained by supervised learning
on real human games, then run **entirely in the browser** (ONNX Runtime Web, no server)
behind an **alpha-beta search**. Two layers, matching the AI Lab chess-ai page:

- **Layer 2 — ChessNet (the net):** `model.py` — 13×8×8 input, 64 channels × 5 residual
  blocks, a policy head over all 4096 `from*64+to` moves and a value head (win prob, tanh).
- **Layer 1 — search (the engine):** a policy-guided **iterative-deepening alpha-beta
  minimax** with a material evaluation, implemented in JS in the demo. The net orders/prunes
  candidate moves (opening + positional knowledge); the search adds tactics (won't hang
  pieces, grabs free material, finds short mates).

Live demo: **https://ai-lab-bice.vercel.app/projects/chess-ai**

## Files

| File | Role |
|---|---|
| `model.py` | `ChessNet` (policy+value), `save()`/`load()` |
| `board.py` | `board_to_tensor` (13×8×8), `move_to_index` (`from*64+to`), inverse |
| `train_supervised.py` | Supervised pre-training on the HuggingFace `adamkarvonen/chess_games` dataset (streaming). CPU/GPU/TPU, resumable checkpoints. |
| `export_onnx.py` | Exports `chessnet.pth` → single-file `chessnet.onnx` |
| `selfplay.py` / `mcts.py` | MCTS self-play stage (not used in the deployed model yet) |
| `chessnet.pth` | Latest trained weights (~3.6 MB) |
| `chessnet_checkpoint.pth` | Resume checkpoint (weights + next-epoch number) |
| `chessnet_1epoch_backup.pth` | Safety backup of the first trained model |

The **served model** lives at `../portfolio-website/assets/files/chessnet.onnx` and the demo
loads it via the AI Lab's `/api/assets/chessnet.onnx` proxy. The demo component is
`../ai-lab/app/components/ChessDemo.js`.

## Training

```bash
cd "Chess AI"
.venv/Scripts/activate                       # Windows (source .venv/bin/activate elsewhere)
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1    # REQUIRED on Windows (see gotchas)

# fresh run
python train_supervised.py --samples 200000 --epochs 8 --min-elo 1600 --save-every 100

# resume from checkpoint (continues from chessnet_checkpoint.pth's epoch)
python train_supervised.py --resume --samples 200000 --epochs 12 --save-every 100
```

- Dataset streams SAN game transcripts filtered by `--min-elo`; each game is replayed into
  `(board_tensor, policy_index, value)` training positions.
- Loss = policy cross-entropy + value MSE. `--save-every N` writes a mid-epoch checkpoint
  every N batches so an interrupted run never loses much.

### Progress so far (supervised)

Trained in repeated resume cycles (see the 35-min limit gotcha). Policy cross-entropy over
the 4096-move space (random = ln(4096) ≈ 8.31):

| Epoch | policy loss | value loss | notes |
|---|---|---|---|
| 1 | 5.42 | 0.87 | plays book openings but mixes in weak moves |
| 2 | 3.66 | 0.74 | |
| 3 | 3.23 | 0.65 | |
| 4 | 2.71 | 0.44 | |
| 5 | **2.57** | **0.40** | **currently deployed** — 1.e4→c5 (Sicilian), prioritizes O-O |
| 6 | 2.6783 | 0.5145 | regressed — CosineAnnealingLR reset on resume (see gotchas) |
| 7 | — | — | trained fine; losses lost to stdout buffering (use `python -u`) |
| 8 | 2.3024 | 0.3365 | LR stepped down to 2e-4 |
| 9 | **2.2232** | **0.3088** | best so far — on disk, **not yet deployed** |

Losses are only comparable *within* a cycle: each resume streams a different 250K positions.
A fixed validation set is still the missing piece for measuring this properly.

## Export + deploy

```bash
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
python export_onnx.py --checkpoint chessnet.pth --output chessnet.onnx   # ~3.6 MB
cp chessnet.onnx ../portfolio-website/assets/files/chessnet.onnx
cd ../portfolio-website && git add assets/files/chessnet.onnx && git commit && git push
# GitHub Pages serves it; the AI Lab /api/assets proxy picks it up. No AI Lab redeploy
# needed for a model-only update.
```

Deploying **demo code** changes (ChessDemo.js) is different — the AI Lab deploys via the
Vercel CLI, **not** git push:

```bash
cd ../ai-lab && vercel --prod --yes
```

## How the in-browser engine works (`ChessDemo.js`)

1. Loads onnxruntime-web from the CDN (ESM path — see gotcha) and creates a session from
   `/api/assets/chessnet.onnx`.
2. On the AI's turn, encodes the position with the exact `board.py` scheme
   (13×8×8 planes, square = `rank*8+file`) and runs the net once → `policy[4096]` + `value`.
3. Builds a **prior** per legal move = `policy_logit * PW` plus a large bonus for clearly
   winning captures; keeps the **top-16** candidates.
4. **Iterative-deepening alpha-beta** (depths 2→5) with a **material** leaf eval and MVV-LVA
   move ordering. A hard **1.4 s** budget with an in-search abort (checked every 64 nodes)
   keeps complex positions from freezing the tab; it keeps the best move from the last fully
   completed depth. The policy prior is blended into the root score so quiet/opening
   positions follow the net while the search overrides on real tactics.
5. Shows the value head as an eval readout.

chess.js is the speed ceiling (~hundreds of nodes/sec here), so the search reaches depth
2–4. That's enough to avoid one/two-move blunders and grab material; it is **not** a deep
tactical engine.

## Ways to make it stronger

- **More supervised epochs** — keep resuming; policy loss is still trending down.
- **GPU training** — `train_supervised.py` is built for Colab/Kaggle (T4/P100/TPU). A GPU run
  does in minutes what CPU does in hours. Fastest path to a much stronger net.
- **MCTS self-play** (`selfplay.py`) — the AlphaZero second stage, to go beyond human games.
- **Deeper search** — a faster JS move generator (bitboards) instead of chess.js would allow
  higher depth in the same time budget.

## Gotchas (learned the hard way)

- **~35-min background-run kill limit** in this environment → train via repeated
  `--resume` cycles with frequent `--save-every` checkpoints. Streaming returns fresh games
  each cycle, so resuming also adds data diversity.
- **Export needs `onnxscript`** (torch 2.12's ONNX exporter) and **`PYTHONUTF8=1`** — a `→`
  character in a status print crashes under Windows cp1252 and kills the epoch loop.
- **onnxruntime-web ESM CDN path:** import from `dist/esm/ort.min.js` — `dist/ort.min.mjs`
  **404s** (there are no `.mjs` files in onnxruntime-web@1.18.0). Wrong URL = the demo is
  stuck "Loading…". This had silently broken the MNIST demo too.
- **The committed `chessnet.onnx` must have real weights.** An early export produced a 14 KB
  *weightless* graph (external `.data` not merged) — `export_onnx.py` now consolidates it;
  a real model is ~3.6 MB. If it's ~14 KB, it's broken.
- **AI Lab deploys via `vercel --prod`, not GitHub push** (git push does not auto-deploy).
