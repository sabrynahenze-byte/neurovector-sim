import argparse
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn_mnist import CNNModel, train_save_cnn_model
from snn_mnist import SNNFCModel, SNNConvModel, train_save_snn_model, train_save_snn_cnn_model
from aihwkit_mnist import AIHWKITSNNFC, AIHWKITSNNConv, train_save_aihwkit_fc, train_save_aihwkit_cnn
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice

# fixes imports one level up
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_compare import ModelEvaluator, TorchAdapter, SnnTorchAdapter, AIHWKITAdapter, compare_models

VALID_MODELS = ["cnn", "snn_fc", "snn_conv", "aihwkit_fc", "aihwkit_conv"]

TRAIN_FN = {
    "cnn":          train_save_cnn_model,
    "snn_fc":       train_save_snn_model,
    "snn_conv":     train_save_snn_cnn_model,
    "aihwkit_fc":   train_save_aihwkit_fc,
    "aihwkit_conv": train_save_aihwkit_cnn,
}

WEIGHTS = {
    "cnn":          "pytorch_conv_model.pt",
    "snn_fc":       "snntorch_fc_model.pt",
    "snn_conv":     "snntorch_conv_model.pt",
    "aihwkit_fc":   "aihwkit_fc_model.pt",
    "aihwkit_conv": "aihwkit_conv_model.pt",
}

IS_SNN = {
    "cnn": False,
    "snn_fc": True,
    "snn_conv": True,
    "aihwkit_fc": True,
    "aihwkit_conv": True,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and/or evaluate ANN vs SNN models on MNIST."
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train selected models before evaluating."
    )
    parser.add_argument(
        "--train-models", default="all",
        dest="train_models",
        help=f"Comma-separated models to train (only used with --train). Options: {', '.join(VALID_MODELS)}. Default: all."
    )
    parser.add_argument(
        "--models", default="all",
        help=f"Comma-separated models to evaluate. Options: {', '.join(VALID_MODELS)}. Default: all."
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="Device to use for evaluation. Default: auto-detect."
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override the epoch count for all models being trained."
    )
    return parser.parse_args()


def resolve_models(models_arg):
    if models_arg == "all":
        return list(VALID_MODELS)
    selected = [m.strip() for m in models_arg.split(",")]
    invalid = [m for m in selected if m not in VALID_MODELS]
    if invalid:
        raise SystemExit(f"Unknown model(s): {invalid}. Valid options: {VALID_MODELS}")
    return selected


def main():
    args = parse_args()
    eval_models = resolve_models(args.models)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.train:
        train_models = resolve_models(args.train_models)
        for name in train_models:
            print(f"\n--- Training {name} ---")
            kwargs = {"num_epochs": args.epochs} if args.epochs is not None else {}
            TRAIN_FN[name](**kwargs)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=transform),
        batch_size=128, shuffle=False
    )

    # Build only the models that were selected
    # ConstantStepDevice matches training. The model was trained with this device
    # config so weights are already robust to its noise characteristics.
    rpu_config = SingleRPUConfig(device=ConstantStepDevice())

    def build_model(name):
        if name == "cnn":
            return CNNModel()
        if name == "snn_fc":
            return SNNFCModel(num_steps=25)
        if name == "snn_conv":
            return SNNConvModel(num_steps=25)
        if name == "aihwkit_fc":
            # num_steps=25 matches training. Rate encoding needs enough timesteps
            # to produce dense enough spike trains for reliable classification.
            return AIHWKITSNNFC(num_steps=25, rpu_config=rpu_config)
        if name == "aihwkit_conv":
            return AIHWKITSNNConv(num_steps=25, rpu_config=rpu_config)

    def build_adapter(name):
        if name == "cnn":
            return TorchAdapter()
        if name in ("snn_fc", "snn_conv"):
            return SnnTorchAdapter()
        return AIHWKITAdapter()

    criterion = nn.CrossEntropyLoss()
    loss_fn = lambda out, target: criterion(
        out.sum(dim=0) if out.dim() == 3 else out, target
    )

    evaluators = {}
    for name in eval_models:
        model = build_model(name).to(device)
        path = WEIGHTS[name]
        if os.path.exists(path):
            print(f"Loading weights from {path}...")
            model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
            model.eval()
        else:
            print(f"Warning: {path} not found. Running with random weights (expect ~10% accuracy).")
        evaluators[name] = ModelEvaluator(
            model=model,
            adapter=build_adapter(name),
            name=name,
            loss_fn=loss_fn,
            is_snn=IS_SNN[name],
        )

    print(f"\nRunning comparison on {device}...")
    results = compare_models(evaluators, test_loader, device)

    print("\n" + "=" * 110)
    print(f"{'Model':<20} | {'Acc':<8} | {'Loss':<8} | {'ECE':<8} | {'SOPs/Sample':<12} | {'Sparsity':<8} | {'Energy':<14} | {'Latency':<10}")
    print("-" * 110)
    for _, res in results.items():
        acc_str = f"{res.acc * 100:.2f}%"
        loss_str = f"{res.loss:4.3f}" if res.loss is not None else "N/A"
        ece_str = f"{res.ece:.4f}" if res.ece else "N/A"
        sops = sparsity = energy = latency = "N/A"
        if res.extra:
            sops = f"{res.extra.get('avg_sops_per_sample', 0):,.0f}"
            if "activation_sparsity_pct" in res.extra:
                sparsity = f"{res.extra['activation_sparsity_pct']:.2f}%"
                energy = f"{res.extra['energy_proxy_spks']:.2f} spikes"
                latency = f"{res.extra['latency_proxy_steps']:.2f} steps"
        print(f"{res.name:<20} | {acc_str:<8} | {loss_str:<8} | {ece_str:<8} | {sops:<12} | {sparsity:<8} | {energy:<14} | {latency:<10}")
    print("=" * 110)


if __name__ == "__main__":
    main()
