############################################################
# run_comparison.py
#
# Small runner for training/evaluating the current comparison
# models and saving the outputs in one place.
#
# Current model slots:
#   1. Classical ANN        (ann_baseline.py)
#   2. Digital SNN          (poc_snntorch.py, rram_enabled=False)
#   3. Hardware-aware SNN   (snn_aihwkit.py)
#
# Usage:
#   python run_comparison.py
#   python run_comparison.py --epochs 3
#   python run_comparison.py --skip-ann
#
# Outputs:
#   reports/summary.txt
#   reports/full_results.json
#
# Notes:
#   - This is meant as a simple integration script for the repo.
#   - Some model interfaces may still need adjustment as the team
#     finishes alignment.
############################################################

import argparse
import os
import json
import sys

# Ensure src/ is on the path so imports work whether or not the package
# is pip-installed. Run this script from the repo root:
#   python run_comparison.py
_SRC = os.path.join(os.path.dirname(__file__), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import torch.nn as nn
import snntorch.functional as SF

# Import model modules.
# This runner assumes a similar train/eval interface across files.
# If one module changes, this script may need a small update too.

try:
    from neurovector_sim import ann_baseline
except ImportError:
    ann_baseline = None
    print(
        "NOTE: ann_baseline module not found. ANN run will be skipped.\n"
        "      The ANN section in this file is still a placeholder.\n"
    )

from neurovector_sim import poc_snntorch as digital_snn
from neurovector_sim import snn_aihwkit as hw_snn


# ============================================================
# Config
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Shared settings used for the comparison runs below.
# These may still change if the team adjusts alignment.
BATCH_SIZE = 128
NUM_STEPS = 25     # SNN time steps (not used by ANN)
HIDDEN_SIZE = 500
LR = 2e-3

REPORT_DIR = "./reports"


# ============================================================
# Runner Helpers
# ============================================================

def run_ann(epochs: int):
    """
    Train and evaluate the ANN baseline.

    This section is still a placeholder until ann_baseline.py is
    settled in the repo. The expected calls here may need to be
    adjusted depending on how the ANN file ends up being structured.
    """
    if ann_baseline is None:
        print("\n  [ANN] Skipped - ann_baseline module not available.")
        return None

    print("\n" + "=" * 60)
    print(" MODEL 1: Classical ANN")
    print("=" * 60)

    train_loader, test_loader = ann_baseline.get_mnist_loaders(BATCH_SIZE)

    model = ann_baseline.ClassicalANN(hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_acc": [], "train_loss": [], "test_acc": [], "test_loss": []}

    for epoch in range(1, epochs + 1):
        print(f"  Epoch {epoch}/{epochs}", end=" - ")

        train_acc, train_loss = ann_baseline.train_epoch(
            model, train_loader, DEVICE, optimizer, loss_fn
        )
        test_acc, test_loss, _, _ = ann_baseline.eval_epoch(
            model, test_loader, DEVICE, loss_fn
        )

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)

        print(f"train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")

    print(f"\n  [ANN] Final test accuracy: {history['test_acc'][-1]:.3f}")
    return history


def run_digital_snn(epochs: int):
    """Train and evaluate the current digital SNN baseline."""
    print("\n" + "=" * 60)
    print(" MODEL 2: Digital SNN")
    print("=" * 60)

    train_loader, test_loader = digital_snn.get_mnist_loaders(BATCH_SIZE)

    # rram_enabled=False keeps this as the digital baseline
    model = digital_snn.SNNNet(
        num_steps=NUM_STEPS,
        hidden_size=HIDDEN_SIZE,
        rram_enabled=False,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    history = {
        "train_acc": [],
        "train_loss": [],
        "test_acc": [],
        "test_loss": [],
        "energy": [],
        "latency": [],
    }

    for epoch in range(1, epochs + 1):
        print(f"  Epoch {epoch}/{epochs}", end=" - ")

        model.step_drift()  # keeps the call pattern consistent across runs

        train_acc, train_loss = digital_snn.train_epoch(
            model, train_loader, DEVICE, optimizer, loss_fn
        )
        test_acc, test_loss, avg_energy, avg_latency = digital_snn.eval_epoch(
            model, test_loader, DEVICE, loss_fn
        )

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)
        history["energy"].append(avg_energy)
        history["latency"].append(avg_latency)

        print(
            f"train_acc={train_acc:.3f}, "
            f"test_acc={test_acc:.3f}, "
            f"energy={avg_energy:.1f}, "
            f"latency={avg_latency:.1f}"
        )

    print(f"\n  [Digital SNN] Final test accuracy: {history['test_acc'][-1]:.3f}")
    return history


def run_hw_snn(epochs: int):
    """Train and evaluate the hardware-aware SNN using AIHWKIT."""
    print("\n" + "=" * 60)
    print(" MODEL 3: Hardware-Aware SNN")
    print("=" * 60)

    train_loader, test_loader = hw_snn.get_mnist_loaders(BATCH_SIZE)

    rpu = hw_snn.build_rpu_config()
    model = hw_snn.HardwareAwareSNN(
        num_steps=NUM_STEPS,
        hidden_size=HIDDEN_SIZE,
        rpu_config=rpu,
    ).to(hw_snn.DEVICE)  # AIHWKIT device handling is defined in snn_aihwkit.py

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    history = {
        "train_acc": [],
        "train_loss": [],
        "test_acc": [],
        "test_loss": [],
        "energy": [],
        "latency": [],
    }

    for epoch in range(1, epochs + 1):
        print(f"  Epoch {epoch}/{epochs}", end=" - ")

        train_acc, train_loss = hw_snn.train_epoch(
            model, train_loader, hw_snn.DEVICE, optimizer, loss_fn
        )
        test_acc, test_loss, avg_energy, avg_latency = hw_snn.eval_epoch(
            model, test_loader, hw_snn.DEVICE, loss_fn
        )

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)
        history["energy"].append(avg_energy)
        history["latency"].append(avg_latency)

        print(
            f"train_acc={train_acc:.3f}, "
            f"test_acc={test_acc:.3f}, "
            f"energy={avg_energy:.1f}, "
            f"latency={avg_latency:.1f}"
        )

    print(f"\n  [HW-Aware SNN] Final test accuracy: {history['test_acc'][-1]:.3f}")
    return history


# ============================================================
# Summary Table
# ============================================================

def print_summary(results: dict):
    """Print a formatted results table to stdout and save to file."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append(" COMPARISON SUMMARY")
    lines.append("=" * 70)
    lines.append(f"{'Model':<28} {'Test Acc':>10} {'Test Loss':>12} {'Energy':>10} {'Latency':>10}")
    lines.append("-" * 70)

    for label, key, has_proxy in [
        ("Classical ANN", "ann", False),
        ("Digital SNN", "digital_snn", True),
        ("Hardware-Aware SNN", "hw_snn", True),
    ]:
        h = results.get(key)
        if h is None:
            lines.append(f"  {label:<26} {'(skipped)':>10}")
            continue

        acc = f"{h['test_acc'][-1]:.4f}"
        loss = f"{h['test_loss'][-1]:.4f}"
        eng = f"{h['energy'][-1]:.2f}" if has_proxy and h.get("energy") else "N/A"
        lat = f"{h['latency'][-1]:.2f}" if has_proxy and h.get("latency") else "N/A"

        lines.append(f"  {label:<26} {acc:>10} {loss:>12} {eng:>10} {lat:>10}")

    lines.append("=" * 70)
    lines.append("Energy = avg spikes/sample (lower = more efficient)")
    lines.append("Latency = avg first-spike time step (lower = faster)")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    report_path = os.path.join(REPORT_DIR, "summary.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Summary saved to {report_path}")

    json_path = os.path.join(REPORT_DIR, "full_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {json_path}")


# ============================================================
# Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run the current comparison models")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--skip-ann", action="store_true", help="Skip Classical ANN")
    parser.add_argument("--skip-dsnn", action="store_true", help="Skip Digital SNN")
    parser.add_argument("--skip-hwsnn", action="store_true", help="Skip Hardware-Aware SNN")
    args = parser.parse_args()

    print("\nRunning comparison script")
    print(f"Device: {DEVICE} | Epochs: {args.epochs} | Hidden: {HIDDEN_SIZE} | Steps: {NUM_STEPS}\n")

    results = {}

    if not args.skip_ann:
        results["ann"] = run_ann(args.epochs)

    if not args.skip_dsnn:
        results["digital_snn"] = run_digital_snn(args.epochs)

    if not args.skip_hwsnn:
        results["hw_snn"] = run_hw_snn(args.epochs)

    # Filter out models that returned None
    active = {k: v for k, v in results.items() if v is not None}

    if active:
        print_summary(results)
    else:
        print("No models were run. Use --help to see options.")


if __name__ == "__main__":
    main()
