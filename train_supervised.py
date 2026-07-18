"""
Supervised pre-training on real chess games from HuggingFace.

Supports CPU, CUDA GPU, and Google TPU (torch_xla).
Designed to survive session timeouts by saving mid-epoch checkpoints
to Google Drive (Colab) or Kaggle output — so training resumes exactly
where it left off across platforms.

Platform guide:
  Colab TPU   — fastest, use --drive to persist to Google Drive
  Colab GPU   — T4, good fallback
  Kaggle GPU  — 30 h/week free P100, no setup needed, saves to /kaggle/working
  Paperspace  — free A4000 tier, run same script

Usage:
    python train_supervised.py
    python train_supervised.py --drive                  # mount Drive on Colab
    python train_supervised.py --out /kaggle/working    # Kaggle output dir
    python train_supervised.py --samples 500000 --epochs 10
    python train_supervised.py --peek                   # inspect dataset fields
    python train_supervised.py --save-every 500         # checkpoint every 500 batches
"""

import argparse
import os
import shutil
import sys

import chess
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from board import board_to_tensor, move_to_index
from model import ChessNet, load, save

# TPU support — optional, falls back to GPU/CPU
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    USE_TPU = True
except ImportError:
    USE_TPU = False


def get_device():
    if USE_TPU:
        device = xm.xla_device()
        print(f"Device: TPU ({device})")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("Device: CPU (slow — consider Colab or Kaggle for GPU/TPU)")
    return device


def mount_drive():
    """Return the Drive save path. Drive must already be mounted in a Colab cell."""
    drive_root = "/content/drive"
    if not os.path.exists(drive_root):
        print("Drive is not mounted yet. Run this in a Colab notebook cell first:")
        print()
        print("  from google.colab import drive")
        print("  drive.mount('/content/drive')")
        print()
        print("Then re-run this script.")
        sys.exit(1)
    save_dir = os.path.join(drive_root, "MyDrive", "chess_ai")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Drive detected. Checkpoints → {save_dir}")
    return save_dir


class ChessPositionDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        tensor, policy_idx, value = self.records[idx]
        policy = np.zeros(4096, dtype=np.float32)
        policy[policy_idx] = 1.0
        return (
            torch.tensor(tensor),
            torch.tensor(policy),
            torch.tensor(value, dtype=torch.float32),
        )


RESULT_TO_OUTCOME = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}
TERMINAL_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}


def parse_game(transcript, result, min_elo, value_discount=1.0):
    """Replay a game transcript and return (tensor, policy_idx, value) per position.

    value_discount (<=1.0) softens the value label the further a position is from the
    game's end. The default 1.0 reproduces the original label: the final result stamped
    at full magnitude onto EVERY position — which is very noisy (move 3 of a game lost on
    move 60 is labelled "losing" even if the position is dead equal, and held-out value
    MSE stalls near ~1.0 as a result). With e.g. 0.97, a position N plies from the end is
    labelled outcome * 0.97**N, so early positions get a smaller-magnitude (less
    confident) target while positions near the decisive end keep the full signal. This is
    an opt-in training-quality lever; it does not change the deployed pipeline.
    """
    outcome = RESULT_TO_OUTCOME.get(result)
    if outcome is None:
        return []

    # Strip move numbers ("1." "12.") and split into SAN tokens
    import re
    tokens = re.sub(r"\d+\.", " ", transcript).split()
    tokens = [t for t in tokens if t not in TERMINAL_TOKENS]

    board = chess.Board()
    positions = []
    for san in tokens:
        try:
            move = board.parse_san(san)
        except Exception:
            break
        tensor = board_to_tensor(board)
        policy_idx = move_to_index(move)
        # Signed result from the side-to-move's perspective (value applied below).
        sign = 1.0 if board.turn == chess.WHITE else -1.0
        positions.append([tensor, policy_idx, sign * outcome])
        board.push(move)

    if value_discount < 1.0 and positions:
        n = len(positions)
        for i, pos in enumerate(positions):
            pos[2] *= value_discount ** (n - 1 - i)   # plies from the end
    return [tuple(p) for p in positions]


def load_records_npz(path):
    """Load a pre-labelled dataset (e.g. from stockfish_label.py) instead of streaming.

    Skips the HuggingFace stream and the game re-parsing entirely — the positions and
    labels are already built. See stockfish_label.py for why Stockfish labels beat the
    human-move/game-result labels build_records() produces.
    """
    import numpy as _np

    if not os.path.exists(path):
        sys.exit(f"--records {path} not found")
    d = _np.load(path)
    X, P, V = d["X"], d["P"], d["V"]
    labeller = str(d["labeller"]) if "labeller" in d else "unknown"
    extra = f", depth {int(d['depth'])}" if "depth" in d else ""
    print(f"Loaded {len(X):,} pre-labelled positions from {path} "
          f"(labeller: {labeller}{extra})")
    return [(X[i], int(P[i]), float(V[i])) for i in range(len(X))]


def build_records(hf_dataset, max_samples, min_elo=1500, skip_games=0, value_discount=1.0):
    """Stream games and build (tensor, policy_idx, value) records.

    skip_games: skip the first N qualifying games before collecting. Use this to train on a
    DIFFERENT slice than the deployed run, or to move past a held-out region. NOTE the
    held-out set in evaluate.py is built from games AFTER 20,000 — so for a bigger training
    run you push the val boundary out (build val after e.g. 100,000) and train on games
    0..100,000, rather than leaving them unused.
    """
    records = []
    games_seen = games_skipped = qualifying = 0
    for row in hf_dataset:
        if len(records) >= max_samples:
            break
        try:
            transcript = row.get("transcript") or ""
            result = row.get("Result") or ""
            white_elo = int(row.get("WhiteElo") or 0)
            black_elo = int(row.get("BlackElo") or 0)
            if not transcript or not result:
                games_skipped += 1
                continue
            if white_elo < min_elo or black_elo < min_elo:
                games_skipped += 1
                continue
            qualifying += 1
            if qualifying <= skip_games:
                continue
            positions = parse_game(transcript, result, min_elo, value_discount=value_discount)
            if not positions:
                games_skipped += 1
                continue
            for pos in positions:
                if len(records) >= max_samples:
                    break
                records.append(pos)
            games_seen += 1
        except Exception:
            games_skipped += 1
            continue
        if games_seen % 500 == 0 and games_seen > 0:
            print(f"  {len(records):,} positions from {games_seen} games ({games_skipped} skipped)...")
    return records


def do_optimizer_step(optimizer):
    if USE_TPU:
        xm.optimizer_step(optimizer)
    else:
        optimizer.step()


def write_checkpoint(model, epoch, out_dir):
    """Save model weights + next epoch number.

    Optimizer state is intentionally excluded: loading it after moving the model
    to GPU/TPU leaves the state tensors on CPU, crashing the first optimizer.step().
    Adam rebuilds momentum in a few batches with negligible quality loss.
    """
    state = {"model": model.state_dict(), "epoch": epoch}
    tmp = os.path.join(out_dir, "chessnet_tmp.pth")
    ckpt = os.path.join(out_dir, "chessnet_checkpoint.pth")
    if USE_TPU:
        xm.save(state, tmp)
    else:
        torch.save(state, tmp)
    shutil.move(tmp, ckpt)
    # Plain model-only file for export_onnx.py and play_human.py
    model_path = os.path.join(out_dir, "chessnet.pth")
    if USE_TPU:
        xm.save(model.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)


def load_checkpoint(out_dir, model):
    """Load model weights and return the next epoch to train."""
    ckpt = os.path.join(out_dir, "chessnet_checkpoint.pth")
    model_path = os.path.join(out_dir, "chessnet.pth")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        epoch = state.get("epoch", 1)
        print(f"Resumed from checkpoint — starting at epoch {epoch}")
        return epoch
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        print(f"Loaded model weights from {model_path} — starting at epoch 1")
    return 1


def run_epoch(model, loader, optimizer, device, epoch, total_epochs, out_dir, save_every):
    model.train()
    total_loss = total_p = total_v = 0.0
    batch_count = 0

    active_loader = pl.MpDeviceLoader(loader, device) if USE_TPU else loader

    for i, (boards, policies, values) in enumerate(active_loader):
        if not USE_TPU:
            boards = boards.to(device)
            policies = policies.to(device)
            values = values.to(device)

        pred_policies, pred_values = model(boards)
        policy_loss = torch.mean(
            -torch.sum(policies * torch.log_softmax(pred_policies, dim=1), dim=1)
        )
        value_loss = F.mse_loss(pred_values, values.unsqueeze(1))
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        do_optimizer_step(optimizer)

        total_loss += loss.item()
        total_p += policy_loss.item()
        total_v += value_loss.item()
        batch_count += 1

        if save_every > 0 and batch_count % save_every == 0:
            write_checkpoint(model, epoch, out_dir)
            print(f"    [mid-epoch save at batch {i+1}]")

    n = max(batch_count, 1)
    print(
        f"  Epoch {epoch}/{total_epochs} — "
        f"loss={total_loss/n:.4f}  policy={total_p/n:.4f}  value={total_v/n:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default=".",
                        help="Directory to save checkpoints (use /kaggle/working on Kaggle)")
    parser.add_argument("--drive", action="store_true",
                        help="Mount Google Drive (Colab only) and save there")
    parser.add_argument("--dataset", default="adamkarvonen/chess_games")
    parser.add_argument("--min-elo", type=int, default=1500,
                        help="Skip games where either player is below this ELO. Measured on the "
                             "dataset: median lower-ELO is ~1884, so 1700=87%% of games, "
                             "1900=47%%, 2100=13%% (a stronger imitation target for less data).")
    parser.add_argument("--skip-games", type=int, default=0,
                        help="Skip the first N qualifying games (train a different slice; keep "
                             "clear of the held-out region). See build_records docstring.")
    parser.add_argument("--value-discount", type=float, default=1.0,
                        help="Soften the value label by distance from game end (e.g. 0.97). "
                             "1.0 = original noisy 'final result on every position' label.")
    parser.add_argument("--peek", action="store_true",
                        help="Print first dataset row and exit")
    parser.add_argument("--save-every", type=int, default=500,
                        help="Save checkpoint every N batches (0 = epoch-end only)")
    parser.add_argument("--records", default=None,
                        help="train on a pre-labelled .npz (e.g. from stockfish_label.py) "
                             "instead of streaming+parsing games. Ignores --samples/--min-elo.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in --out directory")
    args = parser.parse_args()

    if args.drive:
        args.out = mount_drive()
    os.makedirs(args.out, exist_ok=True)

    device = get_device()

    # --records supplies positions directly, so don't open the stream at all
    # (saves the network round-trip and works offline).
    hf_ds = None
    if not args.records:
        print(f"Loading dataset: {args.dataset} ...")
        hf_ds = load_dataset(args.dataset, split="train", streaming=True)

    if args.peek:
        row = next(iter(hf_ds))
        print("\nDataset fields:")
        for k, v in row.items():
            print(f"  {k!r}: {repr(v)[:120]}")
        return

    model = ChessNet()
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(args.out, model)
    model.to(device)  # move BEFORE creating optimizer so state lives on correct device
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.records:
        records = load_records_npz(args.records)
    else:
        print(f"Building dataset ({args.samples:,} target positions, min ELO {args.min_elo})...")
        records = build_records(hf_ds, args.samples, min_elo=args.min_elo)
        print(f"Dataset ready: {len(records):,} valid positions")

    if not records:
        print("No valid positions found. Run with --peek to inspect the dataset fields.")
        return

    dataset = ChessPositionDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0 if USE_TPU else 2,
        drop_last=USE_TPU,
        pin_memory=(not USE_TPU and str(device) == "cuda"),
    )
    # T_max = number of epochs; scheduler.step() called once per epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nTraining epochs {start_epoch}–{args.epochs} ...")
    for epoch in range(start_epoch, args.epochs + 1):
        run_epoch(model, loader, optimizer, device, epoch, args.epochs,
                  args.out, args.save_every)
        scheduler.step()
        write_checkpoint(model, epoch + 1, args.out)
        print(f"  Epoch checkpoint saved → {args.out}")

    print(f"\nDone. chessnet.pth is in {args.out}")
    print("Next: python export_onnx.py --checkpoint", os.path.join(args.out, "chessnet.pth"))


if __name__ == "__main__":
    main()
