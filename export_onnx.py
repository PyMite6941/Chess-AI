"""
Export the trained ChessNet to a single self-contained ONNX file.

The new PyTorch ONNX exporter splits weights into a separate .data file
by default. This script consolidates everything back into one file so you
only need to upload and host a single chessnet.onnx.

Usage:
    python export_onnx.py
    python export_onnx.py --checkpoint chessnet.pth --output chessnet.onnx
"""

import argparse
import os

import onnx
import torch

from model import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="chessnet.pth")
    parser.add_argument("--output", default="chessnet.onnx")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Run train_supervised.py or play.py first.")
        return

    model = load(args.checkpoint)
    model.eval()

    dummy = torch.zeros(1, 13, 8, 8)

    # Export — may produce chessnet.onnx + chessnet.onnx.data
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={
            "board":  {0: "batch"},
            "policy": {0: "batch"},
            "value":  {0: "batch"},
        },
        opset_version=17,
    )

    # Consolidate external .data file back into one inline file
    data_file = args.output + ".data"
    if os.path.exists(data_file):
        print("Consolidating external .data file into single ONNX...")
        proto = onnx.load(args.output, load_external_data=True)
        onnx.save_model(proto, args.output, save_as_external_data=False)
        os.remove(data_file)
        print("  .data file merged and removed.")

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nExported: {args.output}  ({size_mb:.1f} MB)")
    print("Single file — safe to upload directly to HuggingFace.")


if __name__ == "__main__":
    main()
