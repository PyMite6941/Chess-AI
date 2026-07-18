# ChessNet — Next Steps

Roadmap for finishing / strengthening the Chess AI. See `CHESSNET.md` for how it all works.

---

## A better way to train — MEASURED findings (2026-07-16)

After the Stockfish run lost, I profiled the actual training pipeline against the real dataset
(`adamkarvonen/chess_games`) instead of guessing. Numbers, and what they rule in/out:

**What I measured (60k-position build, min-ELO 1700, same pipeline as the deployed run):**

| finding | number | implication |
|---|---|---|
| distinct positions | **92.4%** (only 7.6% dupes) | dedup is a *small* lever — de-prioritised |
| opening plies (0–9) | **13.1%** of positions | openings are NOT over-represented; ply-sampling is small too |
| games per 60k positions | **785** (~76 plies/game) | 1M positions ≈ **13k games** |
| dataset lower-ELO median | **1884** | min-ELO 1700 trains on *below-to-median* players |
| games qualifying ≥1900 / ≥2100 | **46% / 13%** | 2000–2100 is the stronger-player sweet spot; ≥2300 too sparse (1.4%) |

**The two hypotheses I *disproved* by measuring:** opening over-representation and duplicate
positions. Both are <15%. I did NOT build dedup/ply-sampling — they weren't worth it. (Measuring
first is the lesson the Stockfish run taught.)

**The levers the evidence actually supports, ranked:**

1. **Raise min-ELO 1700 → ~2000 (imitation-quality).** The deployed net imitates *below-median*
   (~1884) players. 1900–2000 is above median at ~46% data; 2100 is top-13%. Cleanest quality
   win, no code — just `--min-elo 2000`. Fewer games qualify, so stream further / raise
   `--samples`.
2. **More data by moving the val boundary out (quantity).** Training is capped: a 1M-position run
   already eats ~13k of the ~20k games before the held-out region (val = games after 20,000). To
   scale, rebuild val from games after e.g. **100,000** and train on games 0–100,000 (~5–8M
   positions) with `--skip-games`/higher `--samples`. Then train **30–50 epochs to a real policy
   plateau** (the losing run was still falling at epoch 15). One clean run — no `--resume` (LR
   reset).
3. **Value-target discounting (value-head quality).** The value label is the final result stamped
   on every position → held-out value MSE stalls ~1.0. **Implemented** as opt-in
   `--value-discount 0.97` (default 1.0 = unchanged, deployed pipeline untouched). Softens
   early-position labels toward 0, keeps the decisive endgame at full signal. A/B it with
   `evaluate.py --compare`.
4. **Board-encoding gap (architectural ceiling).** `board.py` uses **13 planes and encodes NO
   castling rights, en-passant, or repetition** — the net literally cannot see whether castling
   is legal. AlphaZero-style nets use ~19–20 planes. Adding castling+EP planes is the biggest
   *true* ceiling for a net already at 27% top-1, BUT it requires matching the JS encoder in
   `../ai-lab/app/components/ChessDemo.js` (which builds the 13 planes in-browser) or the ONNX
   input shape breaks. Medium project — do it deliberately, both sides together.
5. **Capacity + augmentation.** 64ch×5 blocks (~0.9M params) is small; 128×10 helps *if* data is
   scaled first (but slows browser inference — measure the 1.4 s budget). Horizontal-mirror
   augmentation (file a↔h) is a free ~2× in the data-limited regime; skip it for positions with
   castling rights. Not yet implemented.

**Recommended next experiment (paste-ready, all on Kaggle):**
```bash
# stronger players + softer value labels + more epochs, one clean run
python -u train_supervised.py --min-elo 2000 --samples 1000000 --epochs 30 \
    --batch 512 --lr 2e-4 --value-discount 0.97 --out /kaggle/working
# build a matching held-out set and compare against the DEPLOYED model
python -u evaluate.py --build --skip-games 20000 --val-positions 5000
python -u evaluate.py --compare chessnet.pth chessnet_human1m.pth
# deploy only if it wins held-out AND tactics. Quick Save the version (durable artifact).
```
Higher-upside than any of the above long-term: **MCTS self-play** (`selfplay.py`, section below)
— the only path that goes *beyond* imitating humans at all.

---

## VERDICT (2026-07-16): the Stockfish run LOST — not deployed

The 200k-position Stockfish model was evaluated on a fresh human-labelled held-out set (5,000
positions from games after the first 20,000) plus the tactics suite, against the deployed
human-label model. **It lost on every metric, decisively:**

| metric | chessnet.pth (Stockfish) | chessnet_human1m.pth (deployed) | winner |
|---|---|---|---|
| policy CE | 6.0811 | **2.9778** | deployed |
| value MSE | 1.1424 | **0.9923** | deployed |
| top-1 % | 9.56 | **26.68** | deployed |
| top-5 % | 22.02 | **57.86** | deployed |
| tactics | 1/5 | **3/5** | deployed |

**The deployed model stays. `chessnet.onnx` = `a7cf8632649a21dd`, unchanged.**

### Why it lost — and what to fix if retrying

The magnitude is the tell. top-1 of **9.56%** and policy CE **6.08** (held-out) vs a training
policy loss of only 3.48 is a huge generalisation gap — this model is barely past early-epoch
quality. Two fair, objective signals (tactics 1/5, top-1 9.56%) both say it plays clearly
weaker chess. Root causes:

1. **Undertrained.** Training policy loss was *still falling* at epoch 15 (3.4795). 15 epochs
   was not enough for the policy head to learn Stockfish's single best move — a far sparser,
   harder target over 4096 classes than "imitate the 1700 human."
2. **Too little data.** 200k Stockfish positions vs 1M human positions. Label *quality* did not
   make up for 5× less *quantity* here, contrary to the hope.
3. The great training value loss (0.0920) **did not transfer** (held-out 1.1424). Caveat: the
   held-out value label is the game *outcome* (±1/0), while the Stockfish value head predicts a
   continuous *eval*, so this metric is somewhat unfair to it — but tactics + policy don't rely
   on that nuance and are clearly worse, so the verdict stands regardless.

**If you retry Stockfish labels:** label **500k–1M** positions (hours of CPU — budget it, save a
Kaggle version so it's durable) and train **30–50 epochs** until policy loss actually plateaus.
Anything less repeats this. Honestly, MCTS self-play (`selfplay.py`, roadmap item 3) is the
higher-upside path to a genuinely stronger net than more supervised Stockfish labels.

### Artifacts (still on Kaggle `/kaggle/working`, persist across sessions)

`chessnet.pth` (Stockfish, 3.73 MB), `sf_dataset.npz` (200k labels), `chessnet_human1m.pth`
(the deployed human model), `validation_set.npz` (the held-out set built for this compare).
Notebook: https://www.kaggle.com/code/mattgresham/notebook0e22426def/edit — the compare is
reproducible with `python evaluate.py --compare chessnet.pth chessnet_human1m.pth`.

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
