# Training ChessNet on Kaggle's free GPU

The laptop does ~18 min/epoch on a 15W Core 7 150U and fights Minecraft for the CPU. A free
Kaggle T4 does the whole run in minutes. `train_supervised.py` already supports it — it
auto-detects CUDA and takes `--out /kaggle/working`.

**Cost: $0.** ~30 GPU hours/week free. No card, no GCP, no billing.

> **Why not GCloud:** the AI Lab is on **Vercel**, not GCloud — there is no AI Lab GCloud to
> move this to. The chess demo has **no backend at all** (the ONNX runs in the visitor's
> browser via `/api/assets/chessnet.onnx`). The only GCP project is `pixel-ai`, which is
> *serving* infrastructure for a different model and cannot train. Real GCP training means
> Vertex AI or a GPU VM, which costs money. **Matt's rule: no money is to be spent on GCloud.**

---

## Setup (once per session)

1. **Phone-verify your Kaggle account first** — kaggle.com/settings → **Phone Verification**.
   Kaggle gates *both* the GPU **and** internet behind this, and this run needs both (GPU for
   speed, internet because the data streams from HuggingFace). Unverified = the accelerator
   dropdown stays greyed out.
2. kaggle.com → **Create** → **New Notebook**.
3. Right sidebar → **Session options**:
   - **Accelerator** → **GPU T4 x2**
   - **Internet** → **On**

   > **Pick T4, NOT P100.** Kaggle's P100 is compute capability **sm_60**, and its preinstalled
   > PyTorch only supports **sm_70+**, so the P100 dies with *"Tesla P100 with CUDA capability
   > sm_60 is not compatible with the current PyTorch installation"*. The T4 is sm_75 and works.
   > (Hit for real on 2026-07-15.) The script uses one GPU, so "x2" costs nothing.
4. Right sidebar → **Input** → **+ Add Input** → **Upload** → **New Dataset**. Upload
   `model.py`, `board.py`, `train_supervised.py`, `stockfish_label.py` from `Chess AI/`.
   Title it **`chessnet-src`**.

Then pick a path:

| Path | Cells | What you get |
|---|---|---|
| **A — human labels** | 1 → 2 → 3 → 4 | A better *1700-imitator*. Simple, ~40-70 min. |
| **B — Stockfish labels** | 1 → B1 → B2 → B3 → 3 → 4 | A genuinely stronger net. Slower (labelling is CPU-bound), but breaks the 1700 ceiling. |

Cell 1 and Cells 3–4 are shared. **Option B is the better model** — see why below.

---

## Cell 1 — setup + GPU check

```python
!pip -q install python-chess datasets

import glob, shutil, os
for name in ("model.py", "board.py", "train_supervised.py"):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, f"{name} not found under /kaggle/input — did the dataset upload?"
    shutil.copy(hits[0], f"/kaggle/working/{name}")
os.chdir("/kaggle/working")
print("code ready:", [f for f in os.listdir() if f.endswith(".py")])

# HARD CHECK. torch.cuda.is_available() is NOT enough — it returns True even on the
# unsupported P100, which then dies at the first kernel launch. Actually run a matmul.
import torch
if not torch.cuda.is_available():
    raise SystemExit("NO GPU — set Accelerator to GPU T4 x2")
print("GPU:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
# want (7,5) = T4.  (6,0) = P100 = BROKEN.
try:
    (torch.randn(1000, 1000, device="cuda") @ torch.randn(1000, 1000, device="cuda")).sum().item()
    print("GPU WORKS — matmul OK")
except Exception as e:
    raise SystemExit(f"GPU BROKEN ({e}) — switch Accelerator to GPU T4 x2")
```

Do not proceed unless you see `capability: (7, 5)` and `GPU WORKS`.

## Cell 2 — train

```python
!python -u train_supervised.py \
    --samples 1000000 \
    --epochs 15 \
    --batch 512 \
    --lr 2e-4 \
    --min-elo 1700 \
    --save-every 200 \
    --out /kaggle/working
```

**The first ~20–30 minutes look slow, and that's correct.** The `Building dataset...` phase is
**CPU-bound**, not GPU — it replays real games through python-chess to make training positions,
and Kaggle gives ~4 cores. The GPU idles. Don't kill it. Once `Dataset ready: 1,000,000 valid
positions` prints, the GPU takes over and epochs fly. Budget **40–70 min total**.

### Why these flags

| Flag | Why |
|---|---|
| `--samples 1000000` | 4x the laptop's 250k. More data is the main lever left. |
| `--epochs 15` | One clean run — see the LR note below. |
| `--batch 512` | 256 is a CPU-sized batch; a T4 (16 GB) goes bigger and faster. |
| `--lr 2e-4` | What was working locally by epoch 9. |
| `--min-elo 1700` | Matches the recent local cycles. |
| `--out /kaggle/working` | Kaggle only lets you download from here. |

### Train fresh — do NOT `--resume`

Deliberate. The **LR-reset gotcha** (`NEXT_STEPS.md`) exists *because* training was chopped into
resume cycles: `CosineAnnealingLR` restarts every run, so the LR jumps back up and loss
regresses — that's why epoch 6 (2.6783) came out worse than epoch 5 (2.57). One uninterrupted
run lets the cosine schedule anneal end-to-end, which is the entire point of it.

Nothing is lost: the epoch-9 weights are already exported and deployed, and
`chessnet_epoch5_backup.onnx` is the model before that.

---

# Option B — Stockfish labels (the ceiling-breaker)

**Do this instead of Cell 2 when you want a genuinely stronger net, not just a better
1700-imitator.**

The problem with Cell 2: `parse_game()` labels every position with

    policy = the move a ~1700-rated human actually played
    value  = the game's final result, stamped onto EVERY position in that game

So the net's ceiling is *imitating 1700-rated players*. More data and more epochs only make
it a better 1700-imitator — they cannot push past it. The value label is also badly noisy:
move 3 of a game lost on move 60 is labelled "losing" even if the position is dead equal,
which is very likely why value loss stalls around 0.3.

Stockfish replaces both targets:

    policy = Stockfish's best move            (~3500 rated, not ~1700)
    value  = Stockfish's eval of THIS position (not a result propagated backwards)

Positions still come from real human games, so the distribution stays realistic — only the
**labels** change. Label quality beats quantity: ~200k Stockfish-labelled positions should
beat 1M human-labelled ones.

**Cost:** Stockfish must analyse every position, so this is **CPU-bound and the GPU idles**
during labelling. Roughly 30–50 ms/position at depth 10. **Measure before committing** —
Cell B1 does a timed probe.

Upload `stockfish_label.py` alongside the other files (add it to the `chessnet-src` dataset,
or re-upload).

## Cell B1 — install Stockfish + timed probe

Self-contained — safe to run in a fresh session, and pulls `stockfish_label.py` straight from
the public repo so there's nothing to upload:

```python
# pip install is NOT optional even if Cell 1 ran earlier: a restarted session loses it,
# and stockfish_label.py then dies with ModuleNotFoundError: No module named 'chess'.
!pip -q install python-chess datasets
!apt-get -qq install -y stockfish
!wget -q -O stockfish_label.py https://raw.githubusercontent.com/PyMite6941/Chess-AI/master/stockfish_label.py
# Probe 2000 positions to get a REAL rate before committing to a long run.
!python -u stockfish_label.py --positions 2000 --depth 10 --every-n 4 --threads 4 --out /tmp/probe.npz
```

Read the `pos/s` line and do the arithmetic: `200000 / rate / 60` = minutes for the real run.
If that's more time than you have, drop `--depth` to 8 (~2x faster) or lower `--positions`.

## Cell B2 — label for real

```python
!pip -q install python-chess datasets
!apt-get -qq install -y stockfish
!python -u stockfish_label.py \
    --positions 200000 \
    --depth 10 \
    --every-n 4 \
    --workers 4 \
    --skip-games 0 \
    --out /kaggle/working/sf_dataset.npz
```

(The pip/apt lines are cheap no-ops if already installed, and save you from a
`ModuleNotFoundError` if the session restarted between cells.)

**Measured 2026-07-15: ~48 pos/s → 200k in ~70 min.** (`--workers` is the throughput
lever, not `--threads` — see the probe note above.)

### `--skip-games 0` is deliberate — do NOT set it to 20000

This looks backwards and isn't. The **validation set is built from games AFTER 20,000**
(Cell 3). So `--skip-games 20000` here would start labelling at game 20,001 — **the exact
games the validation set is drawn from** — leaking test data into training and quietly
invalidating every `evaluate.py --compare` result afterwards.

With `--skip-games 0`, 200k positions at ~19 per game (every 4th ply) ≈ **10,500 games**,
which sits comfortably inside the first 20,000 and never touches validation.

If you raise `--positions` a lot, check the arithmetic again:
`positions / (19 per game)` must stay **well under 20,000 games**.

`--skip-games 20000` keeps this clear of the validation set's games, so the held-out
comparison stays honest.

`--every-n 4` labels every 4th ply — consecutive plies are near-duplicate positions, so
labelling all of them spends Stockfish time for almost no extra information.

## Cell B3 — train on the Stockfish labels

```python
!python -u train_supervised.py \
    --records /kaggle/working/sf_dataset.npz \
    --epochs 15 \
    --batch 512 \
    --lr 2e-4 \
    --out /kaggle/working
```

`--records` skips streaming and game-parsing entirely — the positions and labels are already
built, so this is pure GPU work and fast.

Download `sf_dataset.npz` too if you want to retrain without re-labelling — it's the
expensive artifact.

---

## Cell 3 — build the fixed validation set

**Run this after Cell 2.** It's the yardstick for deciding whether this model actually beats the
deployed one. Build it here rather than on the laptop: it streams past 20,000 games, which is
slow locally and competes with Minecraft.

Why it's needed: `build_records()` walks the HF stream **from the start on every run** — no
shuffle, no seed, no offset. So training re-reads the same positions, and training loss measures
memorisation as much as skill. A held-out set is the only honest comparison. Validation games
are taken from **after** 20,000 games, well clear of the ~13,150 games the 1M-position training
window consumes.

```python
import numpy as np
from datasets import load_dataset
from train_supervised import parse_game

SKIP_GAMES, VAL_POSITIONS, MIN_ELO = 20000, 5000, 1700

hf = load_dataset("adamkarvonen/chess_games", split="train", streaming=True)
X, P, V = [], [], []
qualifying = 0
for row in hf:
    if len(X) >= VAL_POSITIONS:
        break
    t, r = row.get("transcript") or "", row.get("Result") or ""
    try:
        we, be = int(row.get("WhiteElo") or 0), int(row.get("BlackElo") or 0)
    except (TypeError, ValueError):
        continue
    if not t or not r or we < MIN_ELO or be < MIN_ELO:
        continue
    qualifying += 1
    if qualifying <= SKIP_GAMES:                       # inside the training window — skip cheap
        if qualifying % 2000 == 0:
            print(f"  skipped {qualifying:,}/{SKIP_GAMES:,} games...")
        continue
    try:
        positions = parse_game(t, r, MIN_ELO)
    except Exception:
        continue
    for tens, pol, val in positions or []:
        if len(X) >= VAL_POSITIONS:
            break
        X.append(np.asarray(tens, dtype=np.float32)); P.append(np.int64(pol)); V.append(np.float32(val))
    if len(X) and len(X) % 1000 < 80:
        print(f"  collected {len(X):,}/{VAL_POSITIONS:,}...")

np.savez_compressed("/kaggle/working/validation_set.npz",
                    X=np.stack(X), P=np.asarray(P, dtype=np.int64), V=np.asarray(V, dtype=np.float32),
                    skip_games=SKIP_GAMES, min_elo=MIN_ELO)
print(f"\nSaved validation_set.npz — {len(X):,} held-out positions")
```

This is slower than it looks — it has to stream past 20,000 games before collecting anything.
Expect ~10–20 min.

## Cell 4 — check what to download

```python
import os
for f in sorted(os.listdir("/kaggle/working")):
    if f.endswith((".pth", ".npz")):
        print(f, f"{os.path.getsize('/kaggle/working/' + f)/1e6:.1f} MB")
```

## Download

Right sidebar → **Output** → download **both**:
- **`chessnet.pth`** (~3.6 MB) — the trained weights
- **`validation_set.npz`** (~15 MB) — the fixed yardstick

Only `/kaggle/working` is downloadable, which is why `--out` points there.

---

## When the files come back

> **RENAME THE DOWNLOAD.** Kaggle's file is called `chessnet.pth` — and so is the local
> epoch-9 model you're comparing against. Dropping it straight in silently destroys the
> incumbent. Save it as **`chessnet_kaggle.pth`**.
>
> (`chessnet_epoch9_backup.pth` is a safety copy of the deployed epoch-9 weights, made
> 2026-07-15 for exactly this reason.)

Put `chessnet_kaggle.pth` and `validation_set.npz` into `Chess AI/`, then:

```bash
cd "Chess AI"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

# THE decision: new Kaggle model vs the deployed epoch-9 model,
# on held-out positions + tactics
python evaluate.py --compare chessnet_kaggle.pth chessnet_epoch9_backup.pth

# only if the Kaggle model wins:
python export_onnx.py --checkpoint chessnet_kaggle.pth --output chessnet.onnx   # ~3.7 MB
cp chessnet.onnx ../portfolio-website/assets/files/chessnet.onnx
cd ../portfolio-website && git add assets/files/chessnet.onnx && git commit && git push
```

GitHub Pages serves it and the AI Lab's `/api/assets` proxy picks it up — **no AI Lab redeploy
needed** for a model-only change.

**Deploy only if it wins on the held-out set AND doesn't regress on tactics.** Training loss is
not sufficient evidence — that's the whole reason `evaluate.py` exists.

## Gotchas

- **Verify the accelerator actually attached** (Cell 1). A CPU Kaggle notebook is *slower* than
  the laptop, and you'd burn weekly quota for nothing.
- **P100 is broken, T4 works** — see the Setup note.
- **Kaggle sessions die at ~9 h** and stop if the browser tab is closed too long. This run is far
  shorter, but `--save-every 200` checkpoints regardless.
- **Only `/kaggle/working` is downloadable.**
- The dataset streams from HuggingFace, so **Internet must be On** (Settings → Internet).
- Don't bother with `--drive` — that's the Colab path.
