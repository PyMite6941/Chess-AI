"""Relabel real positions with Stockfish instead of 1700-rated humans.

The ceiling problem
-------------------
`train_supervised.py` learns from human games, so its targets are:

    policy = the move a ~1700-rated player actually played
    value  = the game's final result, stamped onto EVERY position in that game

That caps the net at *imitating 1700-rated humans*. More data and more epochs only
make it a better 1700-imitator. And the value label is very noisy: move 3 of a game
lost on move 60 is labelled "losing" even if the position is dead equal — which is
why value loss stalls around 0.3.

Stockfish fixes both targets:

    policy = Stockfish's best move          (~3500 rated, not ~1700)
    value  = Stockfish's eval of THIS position (not a result propagated backwards)

Label quality beats label quantity here: ~200k Stockfish-labelled positions should
beat 1M human-labelled ones, because the thing being imitated is far stronger.

Cost — MEASURED, not guessed
----------------------------
Stockfish must analyse every position, so this is CPU-bound (the GPU sits idle).

Measured on Kaggle 2026-07-15: a single engine with Threads=4, analysing one position
at a time, managed only **~18 pos/s** — 200k would have taken ~3 hours. Stockfish's
threads scale poorly on one shallow search, so the fix is process parallelism: N
separate engines on 1 thread each, labelling N positions at once (--workers). That
scales near-linearly.

Always run a small --positions probe first and read the pos/s line. Do not trust an
estimate in a docstring, including this one.

Positions come from real games (same HuggingFace stream) so the distribution stays
realistic — only the LABELS change.

Usage
-----
    # Kaggle (see KAGGLE.md)
    !apt-get -qq install -y stockfish
    python stockfish_label.py --positions 200000 --depth 10 --out sf_dataset.npz

    # then train on it — no streaming, no re-parsing
    python train_supervised.py --records sf_dataset.npz --epochs 15 --batch 512 --lr 2e-4

    # time it before committing to a big run
    python stockfish_label.py --positions 2000 --depth 10 --out /tmp/probe.npz
"""

import argparse
import os
import shutil
import sys
import time

import chess
import chess.engine
import numpy as np
from multiprocessing import Pool

from board import board_to_tensor, move_to_index

# Stockfish centipawns -> value in [-1, 1]. 400cp ~= winning, so tanh(cp/400) puts
# a decisive edge near +/-0.76 and leaves room above. Same convention the value head
# is already trained against (tanh output).
CP_SCALE = 400.0


def find_stockfish(explicit=None):
    """Locate the engine binary. Kaggle's apt puts it in /usr/games."""
    if explicit:
        if os.path.exists(explicit):
            return explicit
        sys.exit(f"--engine {explicit} does not exist")
    for c in ("stockfish", "stockfish.exe"):
        p = shutil.which(c)
        if p:
            return p
    for p in ("/usr/games/stockfish", "/usr/bin/stockfish", "/usr/local/bin/stockfish"):
        if os.path.exists(p):
            return p
    sys.exit("Stockfish not found. Kaggle/Linux: !apt-get -qq install -y stockfish\n"
             "Windows: download from stockfishchess.org and pass --engine PATH")


# --- worker-process globals (one Stockfish per worker) ----------------------------
_ENGINE = None
_DEPTH = 10


def _worker_init(engine_path, depth, threads, hash_mb):
    """Each worker owns ONE engine. Threads=1 per engine by default: Stockfish's
    internal threading scales badly on a single shallow search, so N single-threaded
    engines beat 1 N-threaded engine by a wide margin."""
    global _ENGINE, _DEPTH
    _DEPTH = depth
    _ENGINE = chess.engine.SimpleEngine.popen_uci(engine_path)
    _ENGINE.configure({"Threads": threads, "Hash": hash_mb})


def _label_fen(fen):
    """Label one position. Returns (fen, best_uci, cp) or None."""
    try:
        board = chess.Board(fen)
        info = _ENGINE.analyse(board, chess.engine.Limit(depth=_DEPTH))
        pv, score = info.get("pv"), info.get("score")
        if not pv or score is None:
            return None
        best = pv[0]
        if best not in board.legal_moves:
            return None
        cp = score.pov(board.turn).score(mate_score=100_000)
        if cp is None:
            return None
        return (fen, best.uci(), cp)
    except Exception:
        return None


def iter_positions(dataset, min_elo, skip_games, every_n):
    """Yield real positions from real games. Only the labels will come from Stockfish."""
    from datasets import load_dataset
    import re

    hf = load_dataset(dataset, split="train", streaming=True)
    qualifying = 0
    for row in hf:
        transcript = row.get("transcript") or ""
        result = row.get("Result") or ""
        try:
            we = int(row.get("WhiteElo") or 0)
            be = int(row.get("BlackElo") or 0)
        except (TypeError, ValueError):
            continue
        if not transcript or not result or we < min_elo or be < min_elo:
            continue
        qualifying += 1
        if qualifying <= skip_games:
            continue

        tokens = [t for t in re.sub(r"\d+\.", " ", transcript).split()
                  if t not in {"1-0", "0-1", "1/2-1/2", "*"}]
        board = chess.Board()
        for ply, san in enumerate(tokens):
            try:
                move = board.parse_san(san)
            except Exception:
                break
            # Sample every Nth ply: consecutive plies are nearly identical, so labelling
            # all of them wastes Stockfish time on redundant positions.
            if ply % every_n == 0 and not board.is_game_over():
                yield board.copy()
            board.push(move)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions", type=int, default=200_000)
    ap.add_argument("--depth", type=int, default=10,
                    help="Stockfish search depth. 10 is a good speed/quality trade; "
                         "8 is ~2x faster, 12 is much slower for modest gain.")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel Stockfish PROCESSES. This is the real throughput lever — "
                         "measured ~18 pos/s with 1 engine, near-linear scaling with N engines.")
    ap.add_argument("--threads", type=int, default=1,
                    help="threads PER engine. Keep at 1: Stockfish threads scale badly on one "
                         "shallow search, so more workers beats more threads.")
    ap.add_argument("--hash", type=int, default=128, help="Stockfish hash MB per engine")
    ap.add_argument("--every-n", type=int, default=4,
                    help="label every Nth ply — consecutive plies are near-duplicates")
    ap.add_argument("--min-elo", type=int, default=1700)
    ap.add_argument("--skip-games", type=int, default=0,
                    help="skip N games first; use 20000 to stay clear of the validation set")
    ap.add_argument("--dataset", default="adamkarvonen/chess_games")
    ap.add_argument("--engine", default=None, help="path to stockfish binary")
    ap.add_argument("--out", default="sf_dataset.npz")
    args = ap.parse_args()

    engine_path = find_stockfish(args.engine)
    print(f"Stockfish: {engine_path}")
    print(f"Labelling {args.positions:,} positions at depth {args.depth} "
          f"({args.workers} workers x {args.threads} thread, every {args.every_n}th ply, "
          f"min ELO {args.min_elo})")
    print("CPU-bound — the GPU is idle during this. Watch the rate line.\n")

    X, P, V = [], [], []
    t0 = time.time()
    skipped = 0

    fens = (b.fen() for b in iter_positions(args.dataset, args.min_elo,
                                            args.skip_games, args.every_n))

    pool = Pool(args.workers, initializer=_worker_init,
                initargs=(engine_path, args.depth, args.threads, args.hash))
    try:
        for result in pool.imap_unordered(_label_fen, fens, chunksize=8):
            if len(X) >= args.positions:
                break
            if result is None:
                skipped += 1
                continue
            fen, best_uci, cp = result
            board = chess.Board(fen)
            # Both labels from the side-to-move's perspective, matching board_to_tensor's
            # plane 12 (turn) and the value head's tanh output.
            X.append(board_to_tensor(board).astype(np.float32))
            P.append(np.int64(move_to_index(chess.Move.from_uci(best_uci))))
            V.append(np.float32(np.tanh(cp / CP_SCALE)))

            n = len(X)
            if n % 500 == 0:
                el = time.time() - t0
                rate = n / el
                eta = (args.positions - n) / rate / 60
                print(f"  {n:,}/{args.positions:,}  {rate:.1f} pos/s  ETA {eta:.0f} min  "
                      f"({skipped} skipped)")
    except KeyboardInterrupt:
        print("\nInterrupted — saving what we have.")
    finally:
        pool.terminate()
        pool.join()

    if not X:
        sys.exit("No positions labelled — check Stockfish installed and --min-elo.")

    np.savez_compressed(args.out,
                        X=np.stack(X), P=np.asarray(P, dtype=np.int64),
                        V=np.asarray(V, dtype=np.float32),
                        depth=args.depth, min_elo=args.min_elo, skip_games=args.skip_games,
                        labeller="stockfish")
    mins = (time.time() - t0) / 60
    print(f"\nSaved {args.out} — {len(X):,} Stockfish-labelled positions "
          f"({os.path.getsize(args.out)/1e6:.0f} MB) in {mins:.0f} min")
    print(f"Train on it:  python train_supervised.py --records {args.out} "
          f"--epochs 15 --batch 512 --lr 2e-4")


if __name__ == "__main__":
    main()
