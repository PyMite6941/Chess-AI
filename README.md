# ChessNet

An AlphaZero-style policy/value network for chess, trained by supervised learning on human games and deployed to run **entirely in the browser** — no server, no engine binary, ONNX Runtime Web behind an alpha-beta search.

**Live demo:** https://ai-lab-bice.vercel.app/projects/chess-ai

## Architecture

Two layers, because neither one is sufficient alone:

**The network** (`model.py`) — ResNet-style, 64 channels × 5 residual blocks, with a policy head over all 4096 `from×64+to` moves and a value head predicting win probability through a tanh. Input is a 13×8×8 board encoding (`board.py`). It supplies positional and opening knowledge.

**The search** — policy-guided iterative-deepening alpha-beta minimax with material evaluation. The network orders and prunes candidate moves; the search supplies the tactics the network alone gets wrong — not hanging pieces, taking free material, finding short mates.

The division matters: a supervised policy network trained on human games learns what moves *look* reasonable, which is not the same as what moves survive a forcing sequence. The search is what stops it from playing plausible blunders.

## Repository

| File | Role |
|---|---|
| `model.py` | `ChessNet` — policy + value heads, save/load |
| `board.py` | Board tensor encoding, move ↔ index mapping |
| `train_supervised.py` | Supervised training on the HuggingFace `adamkarvonen/chess_games` dataset (streaming, resumable, CPU/GPU/TPU) |
| `evaluate.py` | Held-out evaluation — policy CE, value MSE, top-1/top-5 accuracy, tactics suite |
| `export_onnx.py` | `chessnet.pth` → single-file `chessnet.onnx` for the browser |
| `mcts.py`, `selfplay.py` | MCTS self-play stage (built, not yet used in the deployed model) |
| `tactics.py` | Standalone puzzle generation + plain-English tactic explanations (pure `python-chess`, no engine required) |
| `stockfish_label.py` | Stockfish-labelled training data experiment |
| `parity_check.py` / `.mjs` | Proves the Python and JavaScript implementations agree, case for case |

Detailed docs: **`CHESSNET.md`** (how it trains, exports, and deploys), **`NEXT_STEPS.md`** (roadmap and experiment verdicts), **`KAGGLE.md`** (running training on Kaggle's free accelerators).

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate     # source .venv/bin/activate elsewhere
pip install -r requirements.txt
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1          # required on Windows

python train_supervised.py --samples 200000 --epochs 8 --min-elo 1600 --save-every 100
python evaluate.py                                  # held-out comparison
python play_human.py                                # play it locally
```

## How experiments are run here

Every change to the model is treated as a hypothesis that has to beat the deployed model on a **held-out set built the same way as the deployed model's original evaluation**, and it only ships if it wins. The experiment log in `NEXT_STEPS.md` is mostly a record of things that didn't:

- **Stockfish-labelled training data** — lost the held-out comparison. Not deployed.
- **19-plane encoding** (added castling, en-passant, and fifty-move planes to the original 13) — trained with the deployed recipe held constant to isolate the encoding. Policy cross-entropy and value MSE improved fractionally; top-1 accuracy got slightly *worse*; tactics unchanged at 3/5. A wash. Not deployed.
- **Minimum-ELO 2000 + value discounting** — mixed. Not deployed.
- **A test-set leak** in the Stockfish training path, found and fixed (`--skip-games` had to be 0 for that data source, and wasn't).

Three of those are negative results and one is a bug I introduced myself. They're documented here because a training-loss improvement that doesn't survive a held-out comparison isn't an improvement, and the only way to know which is which is to write down the ones that failed.

## Next

Replacing the hand-tuned material evaluation in the search with a trained **NNUE** network, and measuring the ELO difference against the current engine on a fixed test suite — so "the network is better" is a number rather than an impression. The MCTS self-play stage in `mcts.py` is also built but not yet feeding the deployed model.
