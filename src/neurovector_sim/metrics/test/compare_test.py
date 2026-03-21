import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn_mnist import CNNModel
from snn_mnist import SNNFCModel, SNNConvModel
from aihwkit_mnist import AIHWKITSNNFC, AIHWKITSNNConv
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice
import os

from pathlib import Path
import sys

# Fixes imports not found one level up
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, parent_dir)

from model_compare import ModelEvaluator, TorchAdapter, SnnTorchAdapter, AIHWKITAdapter, compare_models

## Setup Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

## Initialize Models
rpu_config = SingleRPUConfig(device=ConstantStepDevice())

pytorch_conv_model = CNNModel().to(device)
snntorch_fc_model = SNNFCModel(num_steps=25).to(device)
snntorch_conv_model = SNNConvModel(num_steps=25).to(device)
aihwkit_fc_model = AIHWKITSNNFC(num_steps=5, rpu_config=rpu_config).to(device)
aihwkit_conv_model = AIHWKITSNNConv(num_steps=5, rpu_config=rpu_config).to(device)

pytorch_conv_weights_path = "pytorch_conv_model.pt"
snntorch_fc_weights_path = "snntorch_fc_model.pt"
snntorch_conv_weights_path = "snntorch_conv_model.pt"
aihwkit_fc_weights_path = "aihwkit_fc_model.pt"
aihwkit_conv_weights_path="aihwkit_conv_model.pt"

for model, path in {
    pytorch_conv_model: pytorch_conv_weights_path,
    snntorch_fc_model: snntorch_fc_weights_path,
    snntorch_conv_model: snntorch_conv_weights_path,
    aihwkit_fc_model: aihwkit_fc_weights_path,
}.items():
    if os.path.exists(path):
        print(f"Loading parameters from {path}...")
        state_dict = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        print(
            f"Warning: {path} not found. Running with random weights (expect ~10% accuracy)."
        )

## Setup comparison evaluators

# Loss function adjusted based on ANN vs SNN dimensionality
criterion = nn.CrossEntropyLoss()
shared_loss_fn = lambda out, target: criterion(
    out.sum(dim=0) if out.dim() == 3 else out, target
)

evaluators = {
    "PyTorch_ANN_Conv": ModelEvaluator(
        model=pytorch_conv_model,
        adapter=TorchAdapter(),  # Handles standard [B, C] output
        name="PyTorch_ANN_Conv",
        loss_fn=shared_loss_fn,
        is_snn=False,  # No SOP counting for standard CNN
    ),
    "snnTorch_SNN_FC": ModelEvaluator(
        model=snntorch_fc_model,
        adapter=SnnTorchAdapter(),  # Handles [T, B, C] spiking output
        name="SNN_FC_Model",
        loss_fn=shared_loss_fn,
        is_snn=True,  # Enables SOPs, Sparsity, and Latency metrics
    ),
    "snnTorch_SNN_Conv": ModelEvaluator(
        model=snntorch_conv_model,
        adapter=SnnTorchAdapter(),  # Handles [T, B, C] spiking output
        name="SNN_Conv_Model",
        loss_fn=shared_loss_fn,
        is_snn=True,  # Enables SOPs, Sparsity, and Latency metrics
    ),
    "aihwkit_SNN_FC": ModelEvaluator(
        model=aihwkit_fc_model,
        adapter=AIHWKITAdapter(),
        name="aihwkit_FC_Model",
        loss_fn=shared_loss_fn,
        is_snn=True,
    ),
    "aihwkit_SNN_Conv": ModelEvaluator(
        model=aihwkit_conv_model,
        adapter=AIHWKITAdapter(),
        name="aihwkit_Conv_model",
        loss_fn=shared_loss_fn,
        is_snn=True,
    )
}

## Run evaluators and print results
print(f"Running comparison on {device}...")
results = compare_models(evaluators, test_loader, device)

print("\n" + "=" * 110)
print(
    f"{'Model':<20} | {'Acc':<8} | {'Loss':<8} | {'ECE':<8} | {'SOPs/Sample':<12} | {'Sparsity':<8} | {'Energy':<14} | {'Latency':<10}"
)
print("-" * 110)

for _, res in results.items():
    acc_str = f"{res.acc*100:.2f}%"
    loss_str = f"{res.loss:4.3f}" if res.loss is not None else "N/A"
    ece_str = f"{res.ece:.4f}" if res.ece else "N/A"
    sops = sparsity = energy = latency = "N/A"

    if res.extra:
        sops = f"{res.extra.get('avg_sops_per_sample', 0):,.0f}" if res.extra else "N/A"

        if "activation_sparsity_pct" in res.extra:
            sparsity = f"{res.extra['activation_sparsity_pct']:.2f}%"
            energy = f"{res.extra['energy_proxy_spks']:.2f} spikes"
            latency = f"{res.extra['latency_proxy_steps']:.2f} steps"

    print(
        f"{res.name:<20} | {acc_str:<8} | {loss_str:<8} | {ece_str:<8} | {sops:<12} | {sparsity:<8} | {energy:<14} | {latency:<10}"
    )
print("=" * 110)
