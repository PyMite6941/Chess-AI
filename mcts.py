import math
import numpy as np
import chess

from board import board_to_tensor, move_to_index, get_legal_moves, get_outcome

C_PUCT = 1.4


class MCTSNode:
    __slots__ = ("board", "parent", "prior", "children", "visits", "value_sum")

    def __init__(self, board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0

    def is_leaf(self):
        return len(self.children) == 0

    def ucb_score(self):
        q = self.value_sum / (self.visits + 1e-8)
        u = C_PUCT * self.prior * math.sqrt(self.parent.visits + 1) / (1 + self.visits)
        return q + u

    def best_child(self):
        return max(self.children.items(), key=lambda kv: kv[1].ucb_score())


def run_mcts(board, model, n_simulations=200):
    root = MCTSNode(board.copy())

    for _ in range(n_simulations):
        node = root
        path = [node]

        # Selection: walk to a leaf
        while not node.is_leaf():
            _, node = node.best_child()
            path.append(node)

        # Expansion + evaluation
        if node.board.is_game_over():
            raw = get_outcome(node.board)  # White's perspective: +1/-1/0
            # Convert to current player's perspective for backprop
            value = raw if node.board.turn == chess.WHITE else -raw
        else:
            tensor = board_to_tensor(node.board)
            policy, value = model.predict(tensor)
            for move in get_legal_moves(node.board):
                child_board = node.board.copy()
                child_board.push(move)
                node.children[move] = MCTSNode(
                    child_board, parent=node, prior=float(policy[move_to_index(move)])
                )

        # Backpropagation — alternate sign at each level
        for i, n in enumerate(reversed(path)):
            n.visits += 1
            n.value_sum += value if i % 2 == 0 else -value

    move_probs = np.zeros(4096, dtype=np.float32)
    for move, child in root.children.items():
        move_probs[move_to_index(move)] = child.visits
    total = move_probs.sum()
    if total > 0:
        move_probs /= total
    return move_probs
