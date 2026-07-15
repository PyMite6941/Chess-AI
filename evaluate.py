"""Measure ChessNet properly instead of guessing from training loss.

Two independent checks:

  1. A FIXED, HELD-OUT validation set — the same positions every time, from games the
     training run never sees. This is the only way to compare two checkpoints honestly.
  2. A TACTICS SUITE — hand-written positions with one clearly correct move. Catches the
     thing loss can't tell you: does it actually see a free piece?

Why a held-out set is necessary here
------------------------------------
`train_supervised.py`'s build_records() walks the HuggingFace stream FROM THE START on
every run — no shuffle, no seed, no offset. So every training run reads the same games in
the same order, and a resumed run re-reads the exact positions it already trained on.
Training loss therefore measures memorisation as much as skill, and can't be trusted to
compare checkpoints.

Validation games are taken from AFTER --skip-games, which is set well past the end of the
training window, so there is no overlap:

    1,000,000 training positions / ~76 positions per game  ~=  13,150 games
    --skip-games 20000  ->  comfortably clear of that

Usage
-----
    # once — build the fixed set (streams past the training window; takes a few minutes)
    python evaluate.py --build --skip-games 20000 --val-positions 5000

    # score a checkpoint
    python evaluate.py --checkpoint chessnet.pth

    # compare two (this is the one that decides a deploy)
    python evaluate.py --compare chessnet.pth chessnet_1epoch_backup.pth

    # tactics only, no validation set needed
    python evaluate.py --checkpoint chessnet.pth --tactics-only

Keep it light while Matt is gaming:
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python evaluate.py ...
"""

import argparse
import os
import sys

import chess
import numpy as np
import torch
import torch.nn.functional as F

from board import board_to_tensor, move_to_index
from model import ChessNet, load

VAL_PATH = "validation_set.npz"

# Positions with one clearly best move. Loss can drop while these still fail, which is
# exactly why they exist. Each: (name, FEN, [accepted moves in SAN], why)
TACTICS = [
    ("hanging queen",
     "rnb1kbnr/pppp1ppp/8/4p3/6q1/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 3",
     ["fxg4"],
     "black queen on g4 is free to the f3 pawn"),
    ("mate in 1 (back rank)",
     "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
     ["Ra8#", "Ra8"],
     "rook to a8 is mate on the back rank"),
    ("free rook",
     "4k3/8/8/8/8/8/4r3/4K2R w K - 0 1",
     ["Rxh1", "Kxe2"],
     "undefended black rook on e2 is capturable by the king"),
    ("take the queen",
     "rnbqkbnr/ppp2ppp/8/3p4/3Q4/8/PPPP1PPP/RNB1KBNR b KQkq - 0 3",
     ["Qxd4"],
     "white queen on d4 hangs to the d8 queen"),
    ("promote",
     "8/P6k/8/8/8/8/7K/8 w - - 0 1",
     ["a8=Q", "a8=Q+", "a8=R"],
     "push the pawn and queen it"),
]


def device_of(prefer_cpu=True):
    if not prefer_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------- build validation set

def build_validation(skip_games, val_positions, min_elo, dataset):
    """Stream past the training window, then collect a fixed validation set."""
    from datasets import load_dataset

    # parse_game lives in the trainer; import lazily so --tactics-only needs no datasets dep
    from train_supervised import parse_game

    print(f"Streaming {dataset} — skipping the first {skip_games:,} qualifying games")
    print("(this is the training window; validation must not overlap it)")
    hf = load_dataset(dataset, split="train", streaming=True)

    tensors, policies, values = [], [], []
    qualifying = 0
    for row in hf:
        if len(tensors) >= val_positions:
            break
        transcript = row.get("transcript") or ""
        result = row.get("Result") or ""
        try:
            white_elo = int(row.get("WhiteElo") or 0)
            black_elo = int(row.get("BlackElo") or 0)
        except (TypeError, ValueError):
            continue
        if not transcript or not result:
            continue
        if white_elo < min_elo or black_elo < min_elo:
            continue

        qualifying += 1
        if qualifying <= skip_games:
            # Cheap skip: don't pay parse_game for games inside the training window.
            if qualifying % 2000 == 0:
                print(f"  skipped {qualifying:,}/{skip_games:,} games...")
            continue

        try:
            positions = parse_game(transcript, result, min_elo)
        except Exception:
            continue
        for t, p, v in positions or []:
            if len(tensors) >= val_positions:
                break
            tensors.append(np.asarray(t, dtype=np.float32))
            policies.append(np.int64(p))
            values.append(np.float32(v))
        if len(tensors) and len(tensors) % 1000 < 80:
            print(f"  collected {len(tensors):,}/{val_positions:,} validation positions...")

    if not tensors:
        sys.exit("No validation positions collected — check --min-elo / --skip-games.")

    X = np.stack(tensors)
    P = np.asarray(policies, dtype=np.int64)
    V = np.asarray(values, dtype=np.float32)
    np.savez_compressed(VAL_PATH, X=X, P=P, V=V,
                        skip_games=skip_games, min_elo=min_elo)
    mb = os.path.getsize(VAL_PATH) / 1e6
    print(f"\nSaved {VAL_PATH} — {len(X):,} positions, {mb:.1f} MB")
    print(f"Held out from games AFTER the first {skip_games:,} qualifying games (min ELO {min_elo}).")
    print("This file is the fixed yardstick — commit the numbers, not the file (it's regenerable).")


# ---------------------------------------------------------------------------- scoring

def load_validation():
    if not os.path.exists(VAL_PATH):
        sys.exit(f"{VAL_PATH} not found — build it first:\n"
                 f"  python evaluate.py --build --skip-games 20000 --val-positions 5000")
    d = np.load(VAL_PATH)
    return d["X"], d["P"], d["V"], int(d["skip_games"]), int(d["min_elo"])


@torch.no_grad()
def score(model, X, P, V, device, batch=256):
    """Policy CE, value MSE, and top-1/top-5 move accuracy over the validation set."""
    model.eval()
    tot_ce = tot_mse = 0.0
    top1 = top5 = n = 0
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        pb = torch.from_numpy(P[i:i + batch]).to(device)
        vb = torch.from_numpy(V[i:i + batch]).to(device)
        logits, val = model(xb)
        val = val.squeeze(-1)
        bs = xb.shape[0]
        tot_ce += F.cross_entropy(logits, pb, reduction="sum").item()
        tot_mse += F.mse_loss(val, vb, reduction="sum").item()
        top = logits.topk(5, dim=1).indices
        top1 += (top[:, 0] == pb).sum().item()
        top5 += (top == pb.unsqueeze(1)).any(dim=1).sum().item()
        n += bs
    return {"policy_ce": tot_ce / n, "value_mse": tot_mse / n,
            "top1": 100.0 * top1 / n, "top5": 100.0 * top5 / n, "n": n}


@torch.no_grad()
def tactics(model, device):
    """Does the raw policy head pick the obviously-correct move? (no search)"""
    model.eval()
    results = []
    for name, fen, accepted, why in TACTICS:
        b = chess.Board(fen)
        x = torch.from_numpy(board_to_tensor(b)[None].astype(np.float32)).to(device)
        logits, _ = model(x)
        pol = logits[0].cpu().numpy()
        best, best_s = None, -1e9
        for m in b.legal_moves:
            s = pol[move_to_index(m)]
            if s > best_s:
                best_s, best = s, m
        san = b.san(best) if best else "-"
        results.append((name, san, san in accepted, accepted[0], why))
    return results


def report_tactics(rows, label=""):
    passed = sum(1 for r in rows if r[2])
    print(f"\nTactics{' — ' + label if label else ''}  ({passed}/{len(rows)} passed)")
    print(f"  {'position':<22} {'played':<10} {'want':<8} ok")
    for name, san, ok, want, _why in rows:
        print(f"  {name:<22} {san:<10} {want:<8} {'PASS' if ok else 'FAIL'}")
    return passed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="build the fixed validation set")
    ap.add_argument("--skip-games", type=int, default=20000,
                    help="skip this many qualifying games first — must clear the training window")
    ap.add_argument("--val-positions", type=int, default=5000)
    ap.add_argument("--min-elo", type=int, default=1700)
    ap.add_argument("--dataset", default="adamkarvonen/chess_games")
    ap.add_argument("--checkpoint", default="chessnet.pth")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--tactics-only", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="use CUDA if present (default CPU)")
    args = ap.parse_args()

    if args.build:
        build_validation(args.skip_games, args.val_positions, args.min_elo, args.dataset)
        return

    device = device_of(prefer_cpu=not args.gpu)

    if args.compare:
        a_path, b_path = args.compare
        A, B = load(a_path).to(device), load(b_path).to(device)
        ta, tb = tactics(A, device), tactics(B, device)
        if args.tactics_only:
            pa, pb = report_tactics(ta, a_path), report_tactics(tb, b_path)
            print(f"\ntactics: {a_path} {pa}/{len(ta)}  vs  {b_path} {pb}/{len(tb)}")
            return
        X, P, V, skip, elo = load_validation()
        sa, sb = score(A, X, P, V, device), score(B, X, P, V, device)
        print(f"\nFixed validation set: {sa['n']:,} held-out positions "
              f"(games after the first {skip:,}, min ELO {elo})")
        print(f"\n  {'metric':<14} {a_path:<24} {b_path:<24} winner")
        for k, better_low, label in (("policy_ce", True, "policy CE"),
                                     ("value_mse", True, "value MSE"),
                                     ("top1", False, "top-1 %"),
                                     ("top5", False, "top-5 %")):
            va, vb_ = sa[k], sb[k]
            win = a_path if ((va < vb_) == better_low and va != vb_) else (b_path if va != vb_ else "tie")
            print(f"  {label:<14} {va:<24.4f} {vb_:<24.4f} {win}")
        pa, pb = report_tactics(ta, a_path), report_tactics(tb, b_path)
        print(f"\ntactics: {a_path} {pa}/{len(ta)}  vs  {b_path} {pb}/{len(tb)}")
        print("\nDeploy the winner only if it wins on the held-out set AND does not "
              "regress on tactics.")
        return

    model = load(args.checkpoint).to(device)
    rows = tactics(model, device)
    if args.tactics_only:
        report_tactics(rows, args.checkpoint)
        return
    X, P, V, skip, elo = load_validation()
    s = score(model, X, P, V, device)
    print(f"\n{args.checkpoint} on {s['n']:,} held-out positions "
          f"(games after the first {skip:,}, min ELO {elo})")
    print(f"  policy CE : {s['policy_ce']:.4f}   (random = ln(4096) = 8.318)")
    print(f"  value MSE : {s['value_mse']:.4f}")
    print(f"  top-1     : {s['top1']:.1f}%   (picks the human's actual move)")
    print(f"  top-5     : {s['top5']:.1f}%")
    report_tactics(rows, args.checkpoint)


if __name__ == "__main__":
    main()
