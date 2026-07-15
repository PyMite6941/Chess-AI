import chess
import numpy as np
import torch
import torch.nn.functional as F

from board import board_to_tensor, get_legal_moves, move_to_index, is_game_over, get_outcome
from mcts import run_mcts

TEMP_THRESHOLD = 30  # use temperature=1 for first 30 moves, then greedy


def play_game(model, n_simulations=200):
    """Run one self-play game. Returns list of (board_tensor, move_probs, value) tuples."""
    board = chess.Board()
    examples = []
    move_count = 0

    while not is_game_over(board):
        move_probs = run_mcts(board, model, n_simulations)
        legal = get_legal_moves(board)
        legal_idxs = np.array([move_to_index(m) for m in legal])
        legal_probs = move_probs[legal_idxs]

        if legal_probs.sum() == 0:
            legal_probs = np.ones(len(legal)) / len(legal)
        else:
            legal_probs /= legal_probs.sum()

        if move_count < TEMP_THRESHOLD:
            chosen_idx = np.random.choice(len(legal), p=legal_probs)
        else:
            chosen_idx = int(np.argmax(legal_probs))

        examples.append((board_to_tensor(board), move_probs, board.turn))
        board.push(legal[chosen_idx])
        move_count += 1

    outcome = get_outcome(board)
    return [
        (tensor, policy, outcome if turn == chess.WHITE else -outcome)
        for tensor, policy, turn in examples
    ]


def train_step(model, optimizer, batch):
    boards = torch.tensor(np.array([b[0] for b in batch]))
    policies = torch.tensor(np.array([b[1] for b in batch]))
    values = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32).unsqueeze(1)

    model.train()
    pred_policies, pred_values = model(boards)
    policy_loss = torch.mean(
        -torch.sum(policies * torch.log_softmax(pred_policies, dim=1), dim=1)
    )
    value_loss = F.mse_loss(pred_values, values)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), policy_loss.item(), value_loss.item()
