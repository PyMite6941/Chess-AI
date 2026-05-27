import chess
import numpy as np

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

def board_to_tensor(board):
    tensor = np.zeros((12,8,8,),dtype=np.float32)
    for i,piece_type in enumerate(PIECE_TYPES):
        for square in board.pieces(piece_type,chess.WHITE):
            row,col = divmod(square,8)
            tensor[i][row][col] = 1.0
        for square in board.pieces(piece_type,chess.BLACK):
            row,col = divmod(square,8)
            tensor[i+6][row][col] = 1.0
    return tensor

def get_legal_moves(board):
    return list(board.legal_moves)

def move_to_index(move):
    return move.from_square*64+move.to_square

def index_to_move(index):
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)

def is_game_over(board):
    return board.is_game_over()

def get_outcome(board):
    result = board.result()
    if result == "1-0": return 1
    elif result == "0-1": return -1
    return 0