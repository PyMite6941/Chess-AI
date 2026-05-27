import chess
import torch
import numpy as np

from board import board_to_tensor, get_legal_moves, move_to_index, is_game_over, get_outcome
from mcts import run_mcts

def play_game(model, n_simulations=100, temp=1.0):
    board = chess.Board()
    examples = []
    while not is_game_over(board):
        tensor = board_to_tensor(board)
        move_probs = run_mcts(board, model, n_simulations)
        if temp > 0:
            move_probs = move_probs ** (1 / temp)
            move_probs /= move_probs.sum()
        legal = get_legal_moves(board)
        legal_idxs = [move_to_index(m) for m in legal]
        legal_probs = np.array([move_probs[i] for i in legal_idxs])
        if legal_probs.sum() == 0:
            legal_probs = np.ones(len(legal))/len(legal)
        else:
            legal_probs /= legal_probs.sum()
        chosen_idx = np.random.choice(len(legal), p=legal_probs)
        chosen_move = legal[chosen_idx]
        examples.append((tensor, move_probs,board.turn))
        board.push(chosen_move)
    outcome = get_outcome(board)
    training_data = []
    for tensor,policy,turn in examples:
        value = outcome if turn == chess.WHITE else -outcome
        training_data.append((tensor, policy, value))
    return training_data

def train_step(model, optimizer, batch):
    import torch.nn.functional as F
    boards = torch.tensor(np.array(b[0] for b in batch))
    policies = torch.tensor(np.array(b[1] for b in batch))
    values = torch.tensor(np.array(b[2] for b in batch), dtype=torch.float32)
    pred_policies, pred_values = model(boards)
    policy_loss = torch.mean(-torch.sum(policies*torch.log_softmax(pred_policies, dim=1), dim=1))
    value_loss = F.mse_loss(pred_values, values)
    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()