import argparse
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ann_mnist import (
    ANNFCModel,
    CNNModel,
    train_save_ann_fc_model,
    train_save_cnn_model,
)
from snn_mnist import (
    SNNFCModel,
    SNNConvModel,
    train_save_snn_model,
    train_save_snn_cnn_model,
)
from aihwkit_mnist import (
    AIHWKITSNNFC,
    AIHWKITSNNConv,
    train_save_aihwkit_fc,
    train_save_aihwkit_cnn,
)
from aihwkit_mnist_digital import (
    AIHWKITDigitalFC,
    AIHWKITDigitalConv,
    train_save_digital_fc,
    train_save_digital_cnn,
)
from aihwkit.simulator.configs import InferenceRPUConfig, SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.nn import AnalogLinear, AnalogConv2d

# fixes imports one level up
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_compare import (
    ModelEvaluator,
    adapt_torch,
    adapt_snn,
)

MODELS = {
    "pytorch_fc": {"train": train_save_ann_fc_model, "weights": "pytorch_fc_model.pt", "is_snn": False},
    "pytorch_conv": {"train": train_save_cnn_model, "weights": "pytorch_conv_model.pt", "is_snn": False},
    "snn_fc": {"train": train_save_snn_model, "weights": "snntorch_fc_model.pt", "is_snn": True},
    "snn_conv": {"train": train_save_snn_cnn_model, "weights": "snntorch_conv_model.pt", "is_snn": True},
    "aihwkit_digital_fc": {"train": train_save_digital_fc, "weights": "aihwkit_digital_fc_model.pt", "is_snn": True},
    "aihwkit_digital_conv": {"train": train_save_digital_cnn, "weights": "aihwkit_digital_conv_model.pt", "is_snn": True},
    "aihwkit_fc": {"train": train_save_aihwkit_fc, "weights": "aihwkit_fc_model.pt", "is_snn": True},
    "aihwkit_conv": {"train": train_save_aihwkit_cnn, "weights": "aihwkit_conv_model.pt", "is_snn": True},
}
VALID_MODELS = list(MODELS)
AIHWKIT_ANALOG_MODELS = {"aihwkit_fc", "aihwkit_conv"}
AIHWKIT_DIGITAL_MODELS = {"aihwkit_digital_fc", "aihwkit_digital_conv"}
AIHWKIT_MODELS = AIHWKIT_ANALOG_MODELS | AIHWKIT_DIGITAL_MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and/or evaluate ANN vs SNN models on MNIST."
    )
    parser.add_argument(
        "--train", action="store_true", help="Train selected models before evaluating."
    )
    parser.add_argument(
        "--train-models",
        default="all",
        dest="train_models",
        help=f"Comma-separated models to train (only used with --train). Options: {', '.join(VALID_MODELS)}. Default: all.",
    )
    parser.add_argument(
        "--models",
        default="all",
        help=f"Comma-separated models to evaluate. Options: {', '.join(VALID_MODELS)}. Default: all.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use for evaluation. Default: auto-detect.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the epoch count for all models being trained.",
    )
    parser.add_argument(
        "--drift-time",
        type=float,
        default=None,
        dest="drift_time",
        help=(
            "Seconds since weight programming to simulate conductance drift on "
            "AIHWKIT models (e.g. 1, 100, 1e6). Omit to skip drift simulation."
        ),
    )
    parser.add_argument(
        "--fail-prob",
        type=float,
        default=None,
        dest="fail_prob",
        help=(
            "Fraction of analog cells to permanently kill on AIHWKIT models, "
            "simulating memristor endurance failure (e.g. 0.01 = 1%%). "
            "Omit to skip endurance simulation."
        ),
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


def transfer_aihwkit_weights(src_model, dst_model):
    """Copy analog weights between models with different RPU configs, layer by layer."""
    src_layers = [m for m in src_model.modules() if isinstance(m, (AnalogLinear, AnalogConv2d))]
    dst_layers = [m for m in dst_model.modules() if isinstance(m, (AnalogLinear, AnalogConv2d))]
    for src, dst in zip(src_layers, dst_layers):
        weights, biases = src.get_weights()
        dst.set_weights(weights, biases)


@torch.no_grad()
def apply_inference_effects(model, drift_time, fail_prob):
    """
    Apply hardware degradation to a trained AIHWKIT model (in-memory only).
    drift_time: seconds since programming (conductance drift).
    fail_prob: fraction of cells to zero out (endurance failure).
    """
    if drift_time is not None:
        model.drift_analog_weights(drift_time)
        print(f"  Drift applied at t={drift_time:.2e}s")

    if fail_prob is not None and fail_prob > 0.0:
        for layer in model.modules():
            if isinstance(layer, (AnalogLinear, AnalogConv2d)):
                weights, biases = layer.get_weights()
                mask = (torch.rand_like(weights) > fail_prob).float()
                layer.set_weights(weights * mask, biases)
        print(f"  Endurance mask applied at fail_prob={fail_prob:.2%}")


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
            MODELS[name]["train"](**kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=transform),
        batch_size=128, shuffle=False,
    )

    # Use InferenceRPUConfig only when drift/fail_prob are requested.
    # Its PCMLikeNoiseModel adds programming noise that tanks accuracy (~40%)
    # even with no drift, so baseline eval keeps the training SingleRPUConfig.
    needs_inference_rpu = args.drift_time is not None or args.fail_prob is not None
    rpu_config = InferenceRPUConfig() if needs_inference_rpu else SingleRPUConfig(device=ConstantStepDevice())

    def build_model(name):
        if name == "pytorch_fc":
            return ANNFCModel()
        if name == "pytorch_conv":
            return CNNModel()
        if name == "snn_fc":
            return SNNFCModel(num_steps=15)
        if name == "snn_conv":
            return SNNConvModel(num_steps=15)
        if name == "aihwkit_fc":
            return AIHWKITSNNFC(num_steps=15, rpu_config=rpu_config)
        if name == "aihwkit_conv":
            return AIHWKITSNNConv(num_steps=15, rpu_config=rpu_config)
        if name == "aihwkit_digital_fc":
            return AIHWKITDigitalFC(num_steps=15)
        if name == "aihwkit_digital_conv":
            return AIHWKITDigitalConv(num_steps=15)

    def build_adapter(name):
        if name in ("pytorch_fc", "pytorch_conv"):
            return adapt_torch
        return adapt_snn

    criterion = nn.CrossEntropyLoss()
    loss_fn = lambda out, target: criterion(
        out.sum(dim=0) if out.dim() == 3 else out, target
    )

    # load_state_dict can't load AnalogTile state into InferenceTile slots,
    # so we load into a SingleRPUConfig model first then transfer weights.
    train_rpu_config = SingleRPUConfig(device=ConstantStepDevice())

    evaluators = {}
    for name in eval_models:
        model = build_model(name).to(device)
        path = MODELS[name]["weights"]
        if os.path.exists(path):
            print(f"Loading weights from {path}...")
            if name in AIHWKIT_DIGITAL_MODELS:
                # Tile types match, load directly.
                model.load_state_dict(
                    torch.load(path, map_location=device, weights_only=False)
                )
            elif name in AIHWKIT_ANALOG_MODELS:
                if needs_inference_rpu:
                    # Tile types differ, load into training config then transfer.
                    if name == "aihwkit_fc":
                        train_model = AIHWKITSNNFC(num_steps=15, rpu_config=train_rpu_config).to(device)
                    else:
                        train_model = AIHWKITSNNConv(num_steps=15, rpu_config=train_rpu_config).to(device)
                    train_model.load_state_dict(
                        torch.load(path, map_location=device, weights_only=False)
                    )
                    transfer_aihwkit_weights(train_model, model)
                else:
                    # Same config as training, load directly.
                    model.load_state_dict(
                        torch.load(path, map_location=device, weights_only=False)
                    )
            else:
                model.load_state_dict(
                    torch.load(path, map_location=device, weights_only=False)
                )
            model.eval()
        else:
            print(
                f"Warning: {path} not found. Running with random weights (expect ~10% accuracy)."
            )

        if name in AIHWKIT_MODELS and (
            args.drift_time is not None or args.fail_prob is not None
        ):
            print(f"Applying inference effects to {name}...")
            apply_inference_effects(model, args.drift_time, args.fail_prob)

        evaluators[name] = ModelEvaluator(
            model=model,
            adapter=build_adapter(name),
            name=name,
            loss_fn=loss_fn,
            is_snn=MODELS[name]["is_snn"],
        )

    print(f"\nRunning comparison on {device}...")
    results = {
        name: ev.evaluate(test_loader, device)
        for name, ev in evaluators.items()
    }

    print("\n" + "=" * 110)
    print(
        f"{'Model':<20} | {'Acc':<8} | {'Loss':<8} | {'ECE':<8} | {'SOPs/Sample':<12} | {'Sparsity':<8} | {'Energy':<14} | {'Latency':<10}"
    )
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
        print(
            f"{res.name:<20} | {acc_str:<8} | {loss_str:<8} | {ece_str:<8} | {sops:<12} | {sparsity:<8} | {energy:<14} | {latency:<10}"
        )
    print("=" * 110)


if __name__ == "__main__":
    main()
