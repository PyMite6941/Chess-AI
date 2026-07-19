"""Lock the tactics engine's behavior. Run: python -m pytest test_tactics.py -q"""
import chess

from tactics import explain_move, find_tactics, generate_puzzles


def _explain(fen, uci):
    return explain_move(chess.Board(fen), chess.Move.from_uci(uci))


def test_checkmate():
    assert "checkmate" in _explain("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1a8")


def test_wins_hanging_queen():
    e = _explain("rnb1kbnr/pppp1ppp/8/4p3/6q1/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 3", "f3g4")
    assert "wins the queen" in e and "undefended" in e


def test_wins_hanging_rook_with_king():
    e = _explain("4k3/8/8/8/8/8/4r3/4K2R w K - 0 1", "e1e2")
    assert "wins the rook" in e


def test_promotion():
    assert "promotes to a queen" in _explain("8/P6k/8/8/8/8/7K/8 w - - 0 1", "a7a8q")


def test_illegal_move_is_flagged_not_described():
    # black queen is blocked by its own d5 pawn — Qxd4 is illegal
    e = _explain("rnbqkbnr/ppp2ppp/8/3p4/3Q4/8/PPPP1PPP/RNB1KBNR b KQkq - 0 3", "d8d4")
    assert "not a legal move" in e


def test_royal_fork_detected():
    # Ne5-f7+ hits the king on h8 (check) AND the queen on d8 — a royal fork
    b = chess.Board("3q3k/8/8/4N3/8/8/8/4K3 w - - 0 1")
    e = explain_move(b, chess.Move.from_uci("e5f7"))
    assert "royal-forks" in e and "queen" in e


def test_solver_finds_mate_first():
    hits = find_tactics(chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"))
    assert hits and hits[0]["motif"] == "mate-in-1"


def test_solver_empty_on_quiet_start():
    # the opening position has no forced tactic
    assert find_tactics(chess.Board()) == []


def test_generate_puzzles_engine_free():
    puzzles = generate_puzzles(count=5, seed=1)
    assert len(puzzles) == 5
    for p in puzzles:
        assert p["fen"] and p["solution"] and p["explanation"]
        # the solution must be legal in its position
        b = chess.Board(p["fen"])
        assert chess.Move.from_uci(p["solution"]) in b.legal_moves
