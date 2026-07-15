# ChessNet — Next Steps

Roadmap for finishing / strengthening the Chess AI. See `CHESSNET.md` for how it all works.

## Current state (2026-07)

- **Deployed model:** 5-epoch supervised net, policy loss **2.57**, value **0.40**. Live at
  https://ai-lab-bice.vercel.app/projects/chess-ai — plays book openings (1.e4→c5), prioritizes
  castling, and now runs behind an in-browser **alpha-beta search** so it won't hang pieces.
- **Latest checkpoint on disk:** `chessnet_checkpoint.pth` = **next epoch 10** (epochs 8 and 9
  trained 2026-07-15 at `--lr 2e-4`, min-ELO 1700). `chessnet.pth` = those epoch-9 weights.
  `chessnet_1epoch_backup.pth` = safety backup.
- **The trained checkpoint is now clearly better than the deployed model** (policy 2.2232 vs
  2.57) but has **not** been exported or deployed. See "Deploy the epoch-9 model?" below.
- **The LR-reset gotcha is confirmed real, and the fix works.** Epoch 6 regressed to 2.6783 —
  *worse* than epoch 5's 2.57 — because CosineAnnealingLR restarts each resume and jumped the
  LR back up. Stepping `--lr` down each cycle (4e-4 → 3e-4 → 2e-4) reversed it: epoch 8 hit
  2.3024, epoch 9 hit **2.2232**, the best so far and inside the 2.2–2.4 plateau target.
- Epoch 7's numbers were lost to stdout buffering (see the `python -u` gotcha) — weights fine,
  printout gone.
- Training has been interrupted twice by machine restarts — nothing lost either time, just resume.

### Progress (min-ELO 1700 cycles)

| Epoch | policy | value | notes |
|---|---|---|---|
| 5 | 2.57 | 0.40 | **currently deployed** as `chessnet.onnx` |
| 6 | 2.6783 | 0.5145 | regressed — LR reset at `--lr 4e-4` |
| 7 | — | — | lost to stdout buffering (`--lr 3e-4`) |
| 8 | 2.3024 | 0.3365 | `--lr 2e-4` |
| 9 | **2.2232** | **0.3088** | best; on disk, not deployed |

### Deploy the epoch-9 model?

It is a real improvement on paper, but losses across cycles aren't strictly comparable (each
cycle streams a *different* 250K positions). Before overwriting the deployed model, either
A/B it by playing both, or build the fixed validation set (see Loose ends). The deployed
epoch-5 model is known-good and plays real opening theory; don't replace it blind.

> Note: losses aren't directly comparable between cycles (each streams a *different* 250K
> positions). To know if a new checkpoint is truly better, either eval on a fixed held-out set
> or just A/B the deployed model by playing it. Don't overwrite the deployed `chessnet.onnx`
> with a checkpoint unless it clearly plays better.

---

## 1. Resume training (do this first after restart)

```bash
cd "Chess AI"
.venv/Scripts/activate
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
python -u train_supervised.py --resume --samples 250000 --epochs 16 --lr 2e-4 --min-elo 1700 --save-every 100
```

- **Use `python -u`.** Without it, stdout is block-buffered when piped to a file, so the whole
  run looks frozen — no `Device:`, no `Building dataset`, no epoch losses — while the only
  thing that appears is an unrelated stderr warning. Epoch 7 trained fine but its losses were
  never flushed before the process was killed. Don't diagnose a "hang" from silence; check
  `chessnet_checkpoint.pth`'s mtime and epoch number instead, which is the real progress signal.

- Background runs die at **~35 min** in this environment → repeat this command each time it
  stops. Frequent `--save-every` means you never lose more than a few batches.
- **LR-reset gotcha:** the CosineAnnealingLR restarts every run, so LR jumps back up at the
  start of each resume and can bump the loss for a few hundred batches. To avoid this, lower
  `--lr` a little each cycle (e.g. 4e-4 → 3e-4 → 2e-4) as you get deeper, or add a proper
  resumable scheduler.
- Stop when policy loss plateaus (~2.2–2.4 on a fixed set) — supervised-only has a ceiling.

Then export + deploy the better model:
```bash
python export_onnx.py --checkpoint chessnet.pth --output chessnet.onnx     # ~3.6 MB
cp chessnet.onnx ../portfolio-website/assets/files/chessnet.onnx
cd ../portfolio-website && git add assets/files/chessnet.onnx && git commit -m "Update ChessNet" && git push
# GitHub Pages serves it; AI Lab proxies it. No AI Lab redeploy needed for a model-only change.
```

## 2. GPU training — the fastest path to "much stronger" (recommended)

CPU here does ~1 epoch / 20 min in 35-min windows. A free GPU does the whole run in minutes.
`train_supervised.py` already supports Colab/Kaggle (T4/P100/TPU, `--drive` for Google Drive
persistence).

- **Kaggle** (easiest, 30 h/week free P100, no setup): upload `model.py`, `board.py`,
  `train_supervised.py`; run `python train_supervised.py --samples 1000000 --epochs 15
  --out /kaggle/working`. Download `chessnet.pth`, export locally, deploy.
- **Colab**: `--drive` to checkpoint to Google Drive across session timeouts.
- Bigger data (500k–1M positions) + more epochs on GPU → a genuinely strong policy net.

## 3. MCTS self-play (the AlphaZero second stage — biggest strength upside)

`selfplay.py` + `mcts.py` exist but aren't wired into the deployed model. This is how the net
goes *beyond* imitating humans:

- Generate self-play games with MCTS guided by the current net, store `(position, visit-count
  policy, game result)`, retrain on that, repeat.
- Needs GPU to be practical (self-play is compute-heavy).
- This is the path from "plays like a decent human" to "genuinely strong."

## 4. Make the in-browser search deeper/stronger

chess.js caps the search at ~depth 2–4 (it's slow). To go deeper in the same time budget:

- Swap chess.js for a **bitboard** move generator (e.g. a WASM engine, or `js-chess-engine`)
  — 10–100× faster move gen → depth 5–7 feasible.
- Add a **transposition table** (Zobrist hashing) and **quiescence search** (extend captures)
  to the negamax in `ChessDemo.js` — big tactical accuracy gain, modest code.
- Use the **value head at leaves** (blended with material) for positional judgment, and the
  **policy for move ordering inside** the tree (PUCT-style), not just at the root — turns it
  into a mini-AlphaZero search. Costs more inference; batch positions to keep it fast.

## 5. Loose ends / polish

- [ ] **Interactive browser playthrough** of the deployed search demo was blocked by a flaky
  Chrome extension (renderer freezes during WASM load). Re-verify when it's stable: play a
  game, deliberately hang a piece, confirm ChessNet takes it and thinking stays ~1.5 s.
- [ ] The chess-ai page copy still describes the full "MCTS self-play" pipeline; once (3) is
  real, it matches. Until then it's slightly aspirational.
- [ ] `chess-bot/` is a separate **classical** engine (minimax, no ML). Its `NOTES.md`
  documents a FastAPI + Fly.io hosting plan that was never built (no `api.py`, stub
  Dockerfile). Ignore unless you want a hosted classical engine as an alternative.
- [ ] Consider a fixed **validation set** (a few thousand held-out positions) so training
  progress is measurable across cycles instead of noisy per-cycle loss.

## Quick reference — gotchas

- `PYTHONUTF8=1` required on Windows (a `→` print crashes cp1252 mid-epoch).
- Export needs `onnxscript` installed (torch 2.12 exporter).
- onnxruntime-web ESM import = `dist/esm/ort.min.js` (NOT `dist/ort.min.mjs`, which 404s).
- Real `chessnet.onnx` ≈ 3.6 MB; a ~14 KB file is a weightless stub (broken).
- AI Lab deploys via `vercel --prod`, not git push. Model file deploys via portfolio-website push.
