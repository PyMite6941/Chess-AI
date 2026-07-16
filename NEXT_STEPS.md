# ChessNet — Next Steps

Roadmap for finishing / strengthening the Chess AI. See `CHESSNET.md` for how it all works.

---

## RESUME HERE (paused 2026-07-16 ~01:00)

**The Stockfish-labelled model was trained on Kaggle but never got off Kaggle.** Everything
below is the state to pick up from.

### The one open question

**Does `/kaggle/working` still contain the files?** The draft session was torn down and
restarted several times. `/kaggle/working` persists across restarts *for saved notebooks*;
whether it survived for this **draft** (version 0, never saved) is **unverified** — the kernel
was stuck in "Draft Session Starting" when work paused, so the listing never ran.

Notebook: https://www.kaggle.com/code/mattgresham/notebook0e22426def/edit

First command to run in the notebook **console**:

```python
import os; print([(f, round(os.path.getsize('/kaggle/working/'+f)/1e6,2))
                  for f in sorted(os.listdir('/kaggle/working')) if f.endswith(('.pth','.npz'))])
```

**If the files are there** — download `chessnet.pth` + `sf_dataset.npz` from the right sidebar
(Output → /kaggle/working), then:

```bash
# RENAME on the way in — chessnet.pth would clobber the DEPLOYED model
mv ~/Downloads/chessnet.pth  "Chess AI/chessnet_stockfish.pth"
mv ~/Downloads/sf_dataset.npz "Chess AI/sf_dataset.npz"
cd "Chess AI" && export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
python evaluate.py --compare chessnet_stockfish.pth chessnet_kaggle.pth
```

**If the files are gone** — re-run Cells B2 then B3 in `KAGGLE.md` (~47 min labelling + ~1 min
training), and **hit Save Version immediately afterwards** so the output becomes a durable
versioned artifact downloadable via the Kaggle API, instead of depending on the Output panel.
That is the fix that should have been used once the tab started freezing.

### Results of the Stockfish run (2026-07-15, training loss only — NOT a verdict)

15/15 epochs on 200,000 Stockfish-labelled positions, batch 512, lr 2e-4, T4:

| | final |
|---|---|
| policy | 3.4795 |
| value | **0.0920** |
| file | `chessnet.pth`, 3.73 MB |

- **Value 0.0920 vs ~0.3 for human labels** is the headline — exactly what the Stockfish value
  target was meant to fix. Encouraging, but it is *training* loss.
- **Policy 3.4795 is NOT comparable to the human runs' ~2.5.** Different label distribution:
  predicting Stockfish's best move out of 4096 is a much harder target than predicting what a
  1700 player did. Comparing the two numbers is meaningless.
- Both losses were **still falling at epoch 15** — not converged, more epochs would likely help.

### The trap waiting at the comparison step

`evaluate.py`'s held-out set is **human-labelled** — it measures "predicts what a 1700 player
played." The Stockfish model was trained *not* to do that. **It may score worse on held-out
policy while being the stronger engine.** Do not read a policy regression as failure, and do
not read it as success either. The **tactics suite** is the more meaningful signal here, and
value MSE is directly comparable. If the result is mixed, say so — don't pick the flattering
framing.

---

## Current state (2026-07-15)

- **Deployed model:** the **Kaggle human-label net** (1M positions, 15 epochs, min-ELO 1700).
  Live at https://ai-lab-bice.vercel.app/projects/chess-ai, served as `chessnet.onnx`
  (sha256 prefix **a7cf8632649a21dd**, 3,698,740 bytes) from GitHub Pages via the AI Lab
  `/api/assets` proxy. **Verified live on both surfaces 2026-07-16.**
- **Local models** in `Chess AI/`:
  - `chessnet_kaggle.pth` — the deployed weights (human labels)
  - `chessnet_epoch9_backup.pth` — the previous incumbent (epoch 9, local)
  - `chessnet.pth` / `chessnet_checkpoint.pth` — the old local epoch-9 lineage
- **`chessnet_stockfish.pth` does not exist yet** — that's the file stuck on Kaggle.

### The lesson that decided the current deployment

Training loss and held-out data **disagreed, and the held-out set was right**:

| | epoch-9 (local) | Kaggle | |
|---|---|---|---|
| training loss | **2.2232** | 2.5215 | epoch-9 looks better... |
| held-out policy CE | 3.2655 | **2.8201** | ...but Kaggle actually is |
| held-out value MSE | 1.2673 | **1.0940** | |
| held-out top-1 | 25.2% | **28.7%** | |
| held-out top-5 | 55.8% | **61.3%** | |
| tactics | 3/5 | 3/5 | tie |

The ranking **flips**. Epoch-9's low training loss was **memorisation** — `build_records()`
re-reads the HF stream from the start every run, so it saw the same 250k positions 9×.
Reading training loss alone would have deployed the worse model.

**Always decide with `evaluate.py --compare`, never with training loss.**

### Progress (min-ELO 1700 local cycles — historical)

| Epoch | policy | value | notes |
|---|---|---|---|
| 5 | 2.57 | 0.40 | was deployed until 2026-07-15 |
| 6 | 2.6783 | 0.5145 | regressed — LR reset at `--lr 4e-4` |
| 7 | — | — | lost to stdout buffering (`--lr 3e-4`) |
| 8 | 2.3024 | 0.3365 | `--lr 2e-4` |
| 9 | **2.2232** | **0.3088** | best training loss — but memorisation (see above) |

**The LR-reset gotcha is confirmed real, and the fix works.** Epoch 6 regressed *below* epoch 5
because CosineAnnealingLR restarts each resume and jumps the LR back up. Stepping `--lr` down
each cycle (4e-4 → 3e-4 → 2e-4) reversed it. One uninterrupted GPU run avoids this entirely.

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

## 2. GPU training — the fastest path to "much stronger" (DECIDED: Kaggle)

**See `KAGGLE.md` for the full paste-and-go guide.** Matt chose the Kaggle path 2026-07-15.

CPU here does ~18 min/epoch on a 15W Core 7 150U and competes with Minecraft. A free Kaggle
P100 does the whole run in minutes. Upload `model.py` + `board.py` + `train_supervised.py`,
turn the GPU on, run one cell, download `chessnet.pth`. Cost $0, ~30 GPU h/week.

**Train fresh on GPU — don't `--resume`.** One uninterrupted run lets CosineAnnealingLR anneal
properly, which is precisely what the resume cycles broke (the LR-reset gotcha below).

**Not GCloud — this was investigated and ruled out (2026-07-15):**
- The **AI Lab is on Vercel, not GCloud** (`.vercel/project.json`, no Dockerfile/cloudbuild).
  There is no AI Lab GCloud to migrate to.
- The **chess demo has no backend**: `ChessDemo.js` loads `/api/assets/chessnet.onnx` (a static
  proxy to GitHub Pages) and runs inference in the visitor's browser. Nothing to host.
- The only GCP project is `pixel-ai` — *serving* infra (scale-to-zero Cloud Run + gateway) for
  a different model. Cloud Run serves requests; it can't train.
- Real GCP training = Vertex AI or a GPU VM = real money on paid billing.
  **Matt's rule: no money is to be spent on GCloud.**
- Colab is the other free option (`--drive` persists to Google Drive across timeouts).

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
