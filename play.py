import os
import torch
from torch.optim import Adam

from model import ChessNet, save, load
from replay import ReplayBuffer
from selfplay import play_game, train_step

CHECKPOINT = "best_model.pt"
REPLAY_FILE = "replay.pkl"
GAMES_PER_ITER = 5
BATCH_SIZE = 64
SIMULATIONS = 100
ITERATIONS = 50

def main():
    if os.path.exists(CHECKPOINT):
        print("Loading model ...")
        model = load(CHECKPOINT)
    else:
        print("Creating new model ...")
        model = ChessNet()
    model.train()
    optimizer = Adam(model.parameters(), lr=0.001)
    buffer = ReplayBuffer()
    if os.path.exists(REPLAY_FILE):
        print("Loading replay buffer ...")
        buffer.load(REPLAY_FILE)
    for iteration in range(ITERATIONS):
        print(f"Iteration {iteration+1}/{ITERATIONS}")
        for g in range(GAMES_PER_ITER):
            print(f"  Game {g + 1}/{GAMES_PER_ITER}...", end=" ")
            examples = play_game(model, n_simulations=SIMULATIONS)
            for ex in examples:
                buffer.push(*ex)
            print(f"got {len(examples)} positions")

        if len(buffer) >= BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            loss  = train_step(model, optimizer, batch)
            print(f"  Loss: {loss:.4f}")

        save(model, CHECKPOINT)
        buffer.save(REPLAY_FILE)
        print("  Saved.")

if __name__ == "__main__":
    main()