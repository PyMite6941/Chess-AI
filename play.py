"""
Self-play training loop.

Run train_supervised.py first to give the model a head start,
then run this to continue improving via self-play.

Usage:
    python play.py
"""

import os
import torch
from torch.optim import Adam

from model import ChessNet, save, load
from replay import ReplayBuffer
from selfplay import play_game, train_step

CHECKPOINT = "chessnet.pth"
REPLAY_FILE = "replay.pkl"
GAMES_PER_ITER = 5
BATCH_SIZE = 128
SIMULATIONS = 200
ITERATIONS = 100


def main():
    if os.path.exists(CHECKPOINT):
        print(f"Loading checkpoint: {CHECKPOINT}")
        model = load(CHECKPOINT)
    else:
        print("No checkpoint found — starting from scratch.")
        print("Tip: run train_supervised.py first for much faster results.")
        model = ChessNet()

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    buffer = ReplayBuffer(maxlen=100_000)

    if os.path.exists(REPLAY_FILE):
        print(f"Loading replay buffer: {REPLAY_FILE}")
        buffer.load(REPLAY_FILE)

    for iteration in range(1, ITERATIONS + 1):
        print(f"\n=== Iteration {iteration}/{ITERATIONS} ===")

        for g in range(1, GAMES_PER_ITER + 1):
            print(f"  Game {g}/{GAMES_PER_ITER}...", end=" ", flush=True)
            examples = play_game(model, n_simulations=SIMULATIONS)
            for ex in examples:
                buffer.push(*ex)
            print(f"{len(examples)} positions")

        if len(buffer) >= BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            loss, p_loss, v_loss = train_step(model, optimizer, batch)
            print(f"  Loss: {loss:.4f}  (policy={p_loss:.4f}, value={v_loss:.4f})")
        else:
            print(f"  Buffer: {len(buffer)}/{BATCH_SIZE} — skipping train step.")

        save(model, CHECKPOINT)
        buffer.save(REPLAY_FILE)
        print("  Saved.")


if __name__ == "__main__":
    main()
