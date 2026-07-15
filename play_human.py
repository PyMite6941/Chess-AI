"""
Play a game against the trained ChessNet.

You play as White. Enter moves in UCI notation (e.g. e2e4, g1f3).

Usage:
    python play_human.py
    python play_human.py --simulations 400   # stronger, slower
"""

import argparse
import os

import chess
import numpy as np

from board import get_legal_moves, move_to_index
from mcts import run_mcts
from model import load


def pick_move(board, model, n_simulations):
    move_probs = run_mcts(board, model, n_simulations)
    legal = get_legal_moves(board)
    legal_idxs = np.array([move_to_index(m) for m in legal])
    legal_probs = move_probs[legal_idxs]
    return legal[int(np.argmax(legal_probs))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="chessnet.pth")
    parser.add_argument("--simulations", type=int, default=200)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"No checkpoint found at {args.checkpoint}.")
        print("Run train_supervised.py first.")
        return

    model = load(args.checkpoint)
    board = chess.Board()

    print("You are White. Enter moves in UCI notation (e2e4, g1f3, ...).")
    print("Type 'quit' to exit, 'undo' to take back your last move.\n")
    print(board)

    while not board.is_game_over():
        print()
        if board.turn == chess.WHITE:
            while True:
                uci = input("Your move: ").strip().lower()
                if uci == "quit":
                    return
                if uci == "undo" and len(board.move_stack) >= 2:
                    board.pop()
                    board.pop()
                    print(board)
                    continue
                try:
                    move = chess.Move.from_uci(uci)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    print("  Illegal move. Try again.")
                except Exception:
                    print("  Invalid format. Use UCI like e2e4.")
        else:
            print(f"Bot thinking ({args.simulations} simulations)...", end=" ", flush=True)
            move = pick_move(board, model, args.simulations)
            board.push(move)
            print(f"Bot plays: {move.uci()}")

        print()
        print(board)

    print("\nGame over:", board.result())


if __name__ == "__main__":
    main()
