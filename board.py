import chess
import numpy as np

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]

# Number of input planes the encoder produces. The in-browser encoder in
# ai-lab/app/components/ChessDemo.js MUST produce exactly this layout, and
# export_onnx / the demo's ort.Tensor shape must use it. See parity_check.py.
N_PLANES = 19


def board_to_tensor(board):
    """Encode a chess.Board as a (19, 8, 8) float32 array.

    Plane layout (square index = rank*8 + file, a1 = 0):
      0-5    white pieces  (P N B R Q K)
      6-11   black pieces  (P N B R Q K)
      12     side to move  (1.0 = White to move, else 0.0)
      13     White kingside  castling right  (all 1.0 if available)
      14     White queenside castling right
      15     Black kingside  castling right
      16     Black queenside castling right
      17     en-passant target square (1.0 on that square, else 0)
      18     fifty-move progress = halfmove_clock / 100, clamped to 1.0 (broadcast)

    The extra planes (13-18) fix a real blind spot: the 13-plane encoding could
    not tell whether castling was legal or an en-passant capture was available,
    so two genuinely different positions encoded identically.
    """
    tensor = np.zeros((N_PLANES, 8, 8), dtype=np.float32)
    for i, piece_type in enumerate(PIECE_TYPES):
        for sq in board.pieces(piece_type, chess.WHITE):
            row, col = divmod(sq, 8)
            tensor[i][row][col] = 1.0
        for sq in board.pieces(piece_type, chess.BLACK):
            row, col = divmod(sq, 8)
            tensor[i + 6][row][col] = 1.0
    if board.turn == chess.WHITE:
        tensor[12] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16] = 1.0
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        tensor[17][row][col] = 1.0
    tensor[18] = min(board.halfmove_clock, 100) / 100.0
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
