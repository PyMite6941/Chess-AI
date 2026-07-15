import chess
import numpy as np

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]


def board_to_tensor(board):
    """Encode a chess.Board as a (13, 8, 8) float32 array.

    Planes 0-5:  white pieces  (P N B R Q K)
    Planes 6-11: black pieces  (P N B R Q K)
    Plane 12:    turn          (1.0 = White to move, 0.0 = Black to move)
    """
    tensor = np.zeros((13, 8, 8), dtype=np.float32)
    for i, piece_type in enumerate(PIECE_TYPES):
        for sq in board.pieces(piece_type, chess.WHITE):
            row, col = divmod(sq, 8)
            tensor[i][row][col] = 1.0
        for sq in board.pieces(piece_type, chess.BLACK):
            row, col = divmod(sq, 8)
            tensor[i + 6][row][col] = 1.0
    tensor[12] = 1.0 if board.turn == chess.WHITE else 0.0
    return tensor


def move_to_index(move):
    return move.from_square * 64 + move.to_square


def index_to_move(index, board=None):
    from_sq = index // 64
    to_sq = index % 64
    move = chess.Move(from_sq, to_sq)
    # Auto-promote pawns reaching the back rank to queen
    if board is not None:
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            back_rank = 7 if board.turn == chess.WHITE else 0
            if chess.square_rank(to_sq) == back_rank:
                move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
    return move


def get_legal_moves(board):
    return list(board.legal_moves)


def is_game_over(board):
    return board.is_game_over()


def get_outcome(board):
    result = board.result()
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    return 0.0
