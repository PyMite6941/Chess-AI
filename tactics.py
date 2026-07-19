"""Portable chess tactics engine — create puzzles and explain tactics anywhere.

Pure python-chess. No neural net, no API keys, no vendor lock-in, $0. Use it as a
library, a CLI, or behind an HTTP endpoint. A matching browser version lives in
ai-lab/app/components/tactics.js (kept in sync with explain_move / find_tactics).

    from tactics import explain_move, find_tactics, generate_puzzles
    import chess
    b = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6q1/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 3")
    print(explain_move(b, chess.Move.from_uci("f3g4")))
    # -> "fxg4 wins the queen — it captures the undefended queen on g4."

    puzzles = generate_puzzles(count=5)          # engine-free, from random play
    puzzles = generate_puzzles(count=5, engine="stockfish")  # engine-verified, harder

CLI:
    python tactics.py explain "<FEN>" <uci>      # explain one move
    python tactics.py solve   "<FEN>"            # find the tactic in a position
    python tactics.py make --count 5             # generate puzzles (add --engine PATH)
"""

import argparse
import json
import random
import sys

import chess

VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
         chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
NAME = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
        chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
MINOR_OR_BETTER = 3  # knight value — "wins material" threshold for a clean puzzle


# ---------------------------------------------------------------- primitives

def _captured_piece(board, move):
    """The piece a move captures (handles en passant), or None."""
    if board.is_en_passant(move):
        # the captured pawn sits behind the destination square
        return board.piece_at(move.to_square + (-8 if board.turn == chess.WHITE else 8))
    return board.piece_at(move.to_square)


def _is_defended(board, square, by_color):
    """Is `square` defended by any of by_color's pieces (as the board stands)?"""
    return bool(board.attackers(by_color, square))


def _fork_targets(board_after, from_square, mover_color):
    """Enemy pieces (knight-or-better, or the king) the piece now on `from_square`
    attacks. Two or more = a fork."""
    targets = []
    for sq in board_after.attacks(from_square):
        p = board_after.piece_at(sq)
        if p and p.color != mover_color and (p.piece_type == chess.KING
                                             or VALUE[p.piece_type] >= MINOR_OR_BETTER):
            targets.append(p.piece_type)
    return targets


# ---------------------------------------------------------------- explanation

def explain_move(board, move):
    """One-sentence, human-readable explanation of what a move does tactically.

    Describes checkmate, check, material won (hanging vs. favourable trade),
    forks (including royal forks), and promotion. Falls back to a plain
    description when there's no tactic.
    """
    mover = board.piece_at(move.from_square)
    if mover is None:
        return "(no piece on the from-square)"
    if move not in board.legal_moves:
        return f"({move.uci()} is not a legal move here)"
    san = board.san(move)
    captured = _captured_piece(board, move)

    after = board.copy(stack=False)
    after.push(move)

    clauses = []

    if after.is_checkmate():
        return f"{san} is checkmate."

    # Material: is the capture free, or a favourable/level trade?
    if captured is not None:
        cap_name = NAME[captured.piece_type]
        # After the move, can the opponent recapture on the destination square?
        recapturers = after.attackers(not mover.color, move.to_square)
        if not recapturers:
            clauses.append(f"wins the {cap_name} - it captures the undefended {cap_name} "
                           f"on {chess.square_name(move.to_square)}")
        elif VALUE[captured.piece_type] > VALUE[mover.piece_type]:
            clauses.append(f"wins material - it takes the {cap_name} with the lower-valued "
                           f"{NAME[mover.piece_type]}")

    # Fork: the moved piece now attacks two or more valuable targets.
    targets = _fork_targets(after, move.to_square, mover.color)
    if len(targets) >= 2:
        names = [NAME[t] for t in targets]
        royal = chess.KING in targets
        label = "royal-forks" if royal else "forks"
        clauses.append(f"{label} the " + " and ".join(dict.fromkeys(names)))

    if after.is_check() and not clauses:
        clauses.append("gives check")

    if move.promotion:
        clauses.append(f"promotes to a {NAME[move.promotion]}")

    if not clauses:
        verb = "captures" if captured else "plays"
        tail = f" the {NAME[captured.piece_type]}" if captured else ""
        return f"{san} {verb}{tail}."

    body = "; ".join(clauses)
    checky = " with check" if after.is_check() and "check" not in body else ""
    return f"{san} {body}{checky}."


# ---------------------------------------------------------------- solving

def find_tactics(board, min_gain=MINOR_OR_BETTER):
    """Engine-free: the clear tactical shots for the side to move, best first.

    Detects, in order of strength: mate in one, a fork of two valuable pieces,
    and capturing an undefended (or favourably traded) piece worth >= min_gain.
    Returns a list of dicts: {move (uci), san, motif, gain, explanation}.
    """
    hits = []
    for move in board.legal_moves:
        captured = _captured_piece(board, move)
        after = board.copy(stack=False)
        after.push(move)

        motif = gain = None
        if after.is_checkmate():
            motif, gain = "mate-in-1", 1000
        else:
            targets = _fork_targets(after, move.to_square, board.turn)
            free = captured is not None and not after.attackers(not board.turn, move.to_square)
            if len(targets) >= 2 and (chess.KING in targets or free):
                motif = "royal-fork" if chess.KING in targets else "fork"
                gain = 900 if chess.KING in targets else sum(
                    VALUE[t] for t in targets if t != chess.KING)
            elif free and VALUE[captured.piece_type] >= min_gain:
                motif, gain = "wins-material", VALUE[captured.piece_type]
            elif (move.promotion == chess.QUEEN
                  and not after.attackers(not board.turn, move.to_square)):
                motif, gain = "promotion", 8  # a new, safe queen

        if motif:
            hits.append({"move": move.uci(), "san": board.san(move), "motif": motif,
                         "gain": gain, "explanation": explain_move(board, move)})
    hits.sort(key=lambda h: h["gain"], reverse=True)
    return hits


# ---------------------------------------------------------------- puzzles

def _random_games(n_games, max_plies=60, seed=0):
    """Yield each random game as a list of its positions. Generating one puzzle per
    game (not per position) avoids emitting the same tactic at consecutive plies."""
    rng = random.Random(seed)
    for _ in range(n_games):
        board = chess.Board()
        positions = []
        for _ in range(rng.randint(4, max_plies)):
            if board.is_game_over():
                break
            positions.append(board.copy(stack=False))
            board.push(rng.choice(list(board.legal_moves)))
        yield positions


def _stockfish_puzzle(board, engine, depth, edge):
    """A position is an engine puzzle if the best move is much better (>= edge cp,
    or forced mate) than the second-best — a real 'only move' tactic."""
    import chess.engine
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=2)
    except Exception:
        return None
    if len(info) < 2:
        return None
    best, second = info[0], info[1]
    s1 = best["score"].pov(board.turn)
    s2 = second["score"].pov(board.turn)
    if s1.is_mate() and s1.mate() and s1.mate() > 0 and not (s2.is_mate() and s2.mate() and s2.mate() > 0):
        gap = "forced mate"
    elif s1.score(mate_score=100000) - s2.score(mate_score=100000) >= edge and s1.score(mate_score=100000) > 100:
        gap = f"+{(s1.score(mate_score=100000) - s2.score(mate_score=100000)) / 100:.1f} over the next-best move"
    else:
        return None
    best_move = best["pv"][0]
    return {"fen": board.fen(), "solution": best_move.uci(), "san": board.san(best_move),
            "motif": "engine-verified", "why": gap, "explanation": explain_move(board, best_move)}


def generate_puzzles(count=10, engine=None, depth=12, edge=250, seed=0, max_games=4000):
    """Create tactic puzzles (at most one per random game, for variety).

    engine=None      -> engine-free: find a mate-in-1, fork, winning capture, or safe
                        promotion (fast, fully portable, no binary needed).
    engine="stockfish" or a path -> engine-verified: keep positions whose best move
                        beats the second-best by `edge` centipawns or forces mate
                        (harder, more varied puzzles).

    Returns a list of {fen, solution, san, motif, explanation, ...}.
    """
    puzzles = []
    eng = None
    if engine:
        import chess.engine
        from stockfish_label import find_stockfish
        eng = chess.engine.SimpleEngine.popen_uci(
            find_stockfish(None if engine == "stockfish" else engine))
    try:
        for positions in _random_games(max_games, seed=seed):
            found = None
            for board in positions:
                if engine:
                    if board.is_check():
                        continue
                    found = _stockfish_puzzle(board, eng, depth, edge)
                else:
                    tac = find_tactics(board)
                    if tac:
                        b = tac[0]
                        found = {"fen": board.fen(), "solution": b["move"], "san": b["san"],
                                 "motif": b["motif"], "explanation": b["explanation"]}
                if found:
                    break
            if found:
                puzzles.append(found)
                if len(puzzles) >= count:
                    break
    finally:
        if eng:
            eng.quit()
    return puzzles


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="Create chess puzzles and explain tactics.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("explain", help="explain one move")
    pe.add_argument("fen")
    pe.add_argument("uci")

    ps = sub.add_parser("solve", help="find the tactic in a position")
    ps.add_argument("fen")

    pm = sub.add_parser("make", help="generate puzzles")
    pm.add_argument("--count", type=int, default=5)
    pm.add_argument("--engine", default=None, help="'stockfish' or a path (engine-verified)")
    pm.add_argument("--seed", type=int, default=0)
    pm.add_argument("--json", action="store_true")

    args = ap.parse_args()

    if args.cmd == "explain":
        board = chess.Board(args.fen)
        print(explain_move(board, chess.Move.from_uci(args.uci)))
    elif args.cmd == "solve":
        board = chess.Board(args.fen)
        hits = find_tactics(board)
        if not hits:
            print("No forced tactic found for the side to move.")
        for h in hits[:5]:
            print(f"  {h['san']:6} [{h['motif']}]  {h['explanation']}")
    elif args.cmd == "make":
        puzzles = generate_puzzles(count=args.count, engine=args.engine, seed=args.seed)
        if args.json:
            print(json.dumps(puzzles, indent=2))
        else:
            for i, p in enumerate(puzzles, 1):
                print(f"\nPuzzle {i}  [{p['motif']}]  ({'White' if ' w ' in p['fen'] else 'Black'} to move)")
                print(f"  FEN: {p['fen']}")
                print(f"  Solution: {p['san']}  —  {p['explanation']}")
        if not puzzles:
            print("No puzzles found in the scan window — try a different --seed or raise --count.")


if __name__ == "__main__":
    main()
