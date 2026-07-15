# Training ChessNet on Kaggle's free GPU

The laptop does ~18 minutes per epoch on a 15W Core 7 150U and fights Minecraft for the CPU.
A free Kaggle T4 does the whole run in minutes. `train_supervised.py` already supports it —
it auto-detects CUDA and takes `--out /kaggle/working`.

**Cost: $0.** Kaggle gives ~30 GPU hours/week free. No card, no GCP, no billing.

> Why not GCloud: the AI Lab is on **Vercel**, not GCloud — there is no AI Lab GCloud to move
> this to. The chess demo has **no backend at all** (the ONNX runs in the visitor's browser via
> `/api/assets/chessnet.onnx`). The only GCP project is `pixel-ai`, which is *serving*
> infrastructure for a different model and cannot train. Real GCP training means Vertex AI or a
> GPU VM, which costs money. Matt's rule: **no money is to be spent on GCloud.**

---

## The 5-minute version

1. Go to **kaggle.com** → *Create* → **New Notebook**.
2. Right sidebar → **Session options** → **Accelerator** → **GPU T4 x2**.

   > **Pick T4, NOT P100.** Kaggle's P100 is compute capability **sm_60**, and its preinstalled
   > PyTorch only supports **sm_70+** — so the P100 errors out with *"Tesla P100 with CUDA
   > capability sm_60 is not compatible with the current PyTorch installation"* and is useless
   > here. The T4 is sm_75 and works. (Hit this for real on 2026-07-15.) The script uses one
   > GPU, so "x2" costs nothing.
3. Right sidebar → **Input** → **Upload** → *New Dataset*. Upload these three files from
   `Chess AI/`:
   - `model.py`
   - `board.py`
   - `train_supervised.py`

   Name it something like `chessnet-src`. It'll mount at
   `/kaggle/input/chessnet-src/`.
4. Paste the cell below into the notebook and run it.
5. When it finishes: right sidebar → **Output** → download **`chessnet.pth`**.
6. Drop that file into `Chess AI/` and tell Claude — the export + A/B + deploy is automated
   from there (see `CHESSNET.md`).

## The notebook cell

```python
!pip -q install python-chess datasets

import shutil, os
for f in ("model.py", "board.py", "train_supervised.py"):
    shutil.copy(f"/kaggle/input/chessnet-src/{f}", f"/kaggle/working/{f}")
os.chdir("/kaggle/working")

# HARD CHECK. `is_available()` alone is NOT enough — it returns True even on the
# unsupported P100, which then fails at the first kernel launch. Actually run a matmul.
import torch
if not torch.cuda.is_available():
    raise SystemExit("NO GPU — set Accelerator to GPU T4 x2")
print("GPU:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
# want (7,5) = T4.  (6,0) = P100 and is BROKEN with Kaggle's PyTorch.
try:
    (torch.randn(1000, 1000, device="cuda") @ torch.randn(1000, 1000, device="cuda")).sum().item()
    print("GPU WORKS — matmul OK")
except Exception as e:
    raise SystemExit(f"GPU BROKEN ({e}) — switch Accelerator to GPU T4 x2")

!python -u train_supervised.py \
    --samples 1000000 \
    --epochs 15 \
    --batch 512 \
    --lr 2e-4 \
    --min-elo 1700 \
    --save-every 200 \
    --out /kaggle/working
```

### Why these flags

| Flag | Why |
|---|---|
| `--samples 1000000` | 4x the laptop's 250k. The GPU can chew it, and more data is the main lever left. |
| `--epochs 15` | One clean run — see the LR note below. |
| `--batch 512` | 256 is a CPU-sized batch; a T4 (16 GB) has the memory to go bigger and faster. |
| `--lr 2e-4` | What was working locally by epoch 9. |
| `--min-elo 1700` | Matches the recent local cycles. |
| `--out /kaggle/working` | Kaggle only lets you download from here. |

### Train fresh — do NOT `--resume` from the local checkpoint

Deliberate. The **LR-reset gotcha** (see `NEXT_STEPS.md`) exists *because* training was chopped
into resume cycles: `CosineAnnealingLR` restarts every run, so the LR jumps back up and the loss
regresses — that's exactly why epoch 6 (2.6783) came out worse than epoch 5 (2.57). A single
uninterrupted GPU run lets the cosine schedule anneal properly end-to-end, which is the whole
point of it. Fresh on GPU should comfortably beat the current epoch-9 model (policy 2.2232).

Nothing is lost by starting over: the epoch-9 weights are already exported and deployed, and
`chessnet_epoch5_backup.onnx` is the previous model.

## Gotchas

- **Check the accelerator actually attached.** The cell prints `CUDA: True <gpu name>`. If it
  says `False`, you're burning your weekly quota on a CPU notebook that's slower than the laptop.
- **Kaggle sessions die at ~9 hours** and stop if the browser tab closes for too long. This run
  is far shorter, but `--save-every 200` checkpoints to `/kaggle/working` regardless.
- **Only `/kaggle/working` is downloadable.** Writing weights anywhere else loses them.
- The dataset streams from HuggingFace (`adamkarvonen/chess_games`) — it needs internet, which
  Kaggle notebooks have on by default. If it fails to fetch, check *Settings → Internet* is on.
- Don't bother with `--drive`; that's the Colab path.

## When the weights come back

Hand `chessnet.pth` to Claude, or do it yourself:

```bash
cd "Chess AI"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
python export_onnx.py --checkpoint chessnet.pth --output chessnet.onnx   # ~3.7 MB
# A/B it against the deployed model before shipping — see CHESSNET.md
cp chessnet.onnx ../portfolio-website/assets/files/chessnet.onnx
cd ../portfolio-website && git add assets/files/chessnet.onnx && git commit && git push
```

GitHub Pages serves it and the AI Lab's `/api/assets` proxy picks it up — **no AI Lab redeploy
is needed** for a model-only change.

**Always A/B before overwriting the deployed model.** Cross-run losses aren't comparable (each
run streams different positions). The epoch-9 deploy was justified by an actual head-to-head:
on a hanging-queen position, epoch 5 played `e4` and missed it while epoch 9 played `fxg4` and
took it. Do that check, not just "the number went down."
