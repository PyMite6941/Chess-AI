import math
import numpy as np
import torch

from board import board_to_tensor,move_to_index,get_legal_moves

C_PUCT = 1.4

class MCTSNode:
    def __init__(self,board,parent=None,prior=0):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visits = 0
        self.value_sum = 0

    def is_leaf(self):
        return len(self.children) == 0
    
    def ucb_score(self):
        q = self.value_sum / (self.visits + 1e-8)
        u = C_PUCT * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q + u

    def best_child(self):
        return max(self.children.items(), key=lambda c: c[1].ucb_score())

def run_mcts(board,model,n_simulations=100):
    root = MCTSNode(board.copy())
    for _ in range(n_simulations):
        node = root
        path = [node]
        while not node.is_leaf():
            move, node = node.best_child()
            path.append(node)
        if not node.board.is_game_over():
            tensor = board_to_tensor(node.board)
            x = torch.tensor(tensor).unsqueeze(0)
            with torch.no_grad():
                policy_logits, value = model(x)
            policy = torch.softmax(policy_logits, dim=1).squeeze().numpy()
            value = value.item()
            legal = get_legal_moves(node.board)
            for move in legal:
                idx = move_to_index(move)
                prior = policy[idx]
                child_board = node.board.copy()
                child_board.push(move)
                node.children[move] = MCTSNode(child_board, parent=node, prior=prior)
        else:
            from board import get_outcome
            value = get_outcome(node.board)
        for n in reversed(path):
            n.visits += 1
            n.value_sum += value
            value = -value
    move_probs = np.zeros(4096)
    for move, child in root.children.items():
        move_probs[move_to_index(move)] = child.visits
    total = move_probs.sum()
    if total > 0:
        move_probs /= total
    return move_probs