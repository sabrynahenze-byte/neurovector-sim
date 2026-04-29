#!/usr/bin/env python3

import io
import base64
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import snntorch as snn
from snntorch import surrogate

try:
    from aihwkit.nn import AnalogLinear, AnalogConv2d
    from aihwkit.simulator.configs import SingleRPUConfig, InferenceRPUConfig
    from aihwkit.simulator.configs.devices import ConstantStepDevice
    from aihwkit.simulator.presets.devices import ReRamArrayOMPresetDevice

    HAS_AIHWKIT = True
except ImportError:
    HAS_AIHWKIT = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# Model classes are copied from src/neurovector_sim/metrics/test/ rather
# than imported, to keep this file runnable without putting the metrics
# package on sys.path.


class ANNFCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        return self.fc1(x)


class SNNFCModel(nn.Module):
    def __init__(self, beta=0.95, num_steps=15):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(784, 256)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(256, 10)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = []
        for _ in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec)


class SNNConvModel(nn.Module):
    def __init__(self, beta=0.95, num_steps=15):
        super().__init__()
        self.num_steps = num_steps
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.pool = nn.MaxPool2d(2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(9216, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_rec = []
        for _ in range(self.num_steps):
            spk1, mem1 = self.lif1(self.conv1(x), mem1)
            spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
            flat = self.flat(self.pool(spk2))
            spk3, mem3 = self.lif3(self.fc1(flat), mem3)
            spk3_rec.append(spk3)
        return torch.stack(spk3_rec)


if HAS_AIHWKIT:

    def _analog_train_rpu():
        return SingleRPUConfig(device=ConstantStepDevice())

    def _digital_infer_rpu():
        cfg = InferenceRPUConfig()
        cfg.device = ReRamArrayOMPresetDevice()
        return cfg

    class AIHWKITSNNFC(nn.Module):
        def __init__(self, num_steps=15, beta=0.95, rpu_config=None):
            super().__init__()
            self.num_steps = num_steps
            if rpu_config is None:
                rpu_config = _analog_train_rpu()
            self.fc1 = AnalogLinear(784, 256, rpu_config=rpu_config)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
            self.fc2 = AnalogLinear(256, 10, rpu_config=rpu_config)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        def forward(self, x):
            x_flat = x.view(x.size(0), -1)
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            spk_out = []
            for _ in range(self.num_steps):
                spk1, mem1 = self.lif1(self.fc1(x_flat), mem1)
                spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
                spk_out.append(spk2)
            return torch.stack(spk_out)

    class AIHWKITSNNConv(nn.Module):
        def __init__(self, num_steps=15, beta=0.95, rpu_config=None):
            super().__init__()
            self.num_steps = num_steps
            if rpu_config is None:
                rpu_config = _analog_train_rpu()
            self.conv1 = AnalogConv2d(1, 32, kernel_size=3, rpu_config=rpu_config)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
            self.conv2 = AnalogConv2d(32, 64, kernel_size=3, rpu_config=rpu_config)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
            self.pool = nn.MaxPool2d(2)
            self.flat = nn.Flatten()
            self.fc1 = AnalogLinear(9216, 10, rpu_config=rpu_config)
            self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
            spk_out = []
            for _ in range(self.num_steps):
                spk1, mem1 = self.lif1(self.conv1(x), mem1)
                spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
                flat = self.flat(self.pool(spk2))
                spk3, mem3 = self.lif3(self.fc1(flat), mem3)
                spk_out.append(spk3)
            return torch.stack(spk_out)

    class AIHWKITDigitalFC(nn.Module):
        def __init__(self, num_steps=15, beta=0.95, rpu_config=None):
            super().__init__()
            self.num_steps = num_steps
            if rpu_config is None:
                rpu_config = _digital_infer_rpu()
            self.fc1 = AnalogLinear(784, 256, rpu_config=rpu_config)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
            self.fc2 = AnalogLinear(256, 10, rpu_config=rpu_config)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        def forward(self, x):
            x_flat = x.view(x.size(0), -1)
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            spk_out = []
            for _ in range(self.num_steps):
                spk1, mem1 = self.lif1(self.fc1(x_flat), mem1)
                spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
                spk_out.append(spk2)
            return torch.stack(spk_out)

    class AIHWKITDigitalConv(nn.Module):
        def __init__(self, num_steps=15, beta=0.95, rpu_config=None):
            super().__init__()
            self.num_steps = num_steps
            if rpu_config is None:
                rpu_config = _digital_infer_rpu()
            self.conv1 = AnalogConv2d(1, 32, kernel_size=3, rpu_config=rpu_config)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
            self.conv2 = AnalogConv2d(32, 64, kernel_size=3, rpu_config=rpu_config)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
            self.pool = nn.MaxPool2d(2)
            self.flat = nn.Flatten()
            self.fc1 = AnalogLinear(9216, 10, rpu_config=rpu_config)
            self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem3 = self.lif3.init_leaky()
            spk_out = []
            for _ in range(self.num_steps):
                spk1, mem1 = self.lif1(self.conv1(x), mem1)
                spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
                flat = self.flat(self.pool(spk2))
                spk3, mem3 = self.lif3(self.fc1(flat), mem3)
                spk_out.append(spk3)
            return torch.stack(spk_out)


MODEL_REGISTRY = {
    "pytorch_fc": {
        "label": "ANN FC",
        "family": "ANN",
        "variant": "FC",
        "weights": "pytorch_fc_model.pt",
        "is_snn": False,
        "is_aihwkit": False,
        "builder": lambda: ANNFCModel(),
    },
    "pytorch_conv": {
        "label": "ANN Conv",
        "family": "ANN",
        "variant": "Conv",
        "weights": "pytorch_conv_model.pt",
        "is_snn": False,
        "is_aihwkit": False,
        "builder": lambda: CNNModel(),
    },
    "snn_fc": {
        "label": "SNN FC",
        "family": "SNN",
        "variant": "FC",
        "weights": "snntorch_fc_model.pt",
        "is_snn": True,
        "is_aihwkit": False,
        "builder": lambda: SNNFCModel(num_steps=15),
    },
    "snn_conv": {
        "label": "SNN Conv",
        "family": "SNN",
        "variant": "Conv",
        "weights": "snntorch_conv_model.pt",
        "is_snn": True,
        "is_aihwkit": False,
        "builder": lambda: SNNConvModel(num_steps=15),
    },
}

if HAS_AIHWKIT:
    MODEL_REGISTRY.update(
        {
            "aihwkit_digital_fc": {
                "label": "AIHWKIT Digital FC",
                "family": "AIHWKIT Digital",
                "variant": "FC",
                "weights": "aihwkit_digital_fc_model.pt",
                "is_snn": True,
                "is_aihwkit": True,
                "builder": lambda: AIHWKITDigitalFC(num_steps=15),
            },
            "aihwkit_digital_conv": {
                "label": "AIHWKIT Digital Conv",
                "family": "AIHWKIT Digital",
                "variant": "Conv",
                "weights": "aihwkit_digital_conv_model.pt",
                "is_snn": True,
                "is_aihwkit": True,
                "builder": lambda: AIHWKITDigitalConv(num_steps=15),
            },
            "aihwkit_fc": {
                "label": "AIHWKIT Analog FC",
                "family": "AIHWKIT Analog",
                "variant": "FC",
                "weights": "aihwkit_fc_model.pt",
                "is_snn": True,
                "is_aihwkit": True,
                "builder": lambda: AIHWKITSNNFC(num_steps=15),
            },
            "aihwkit_conv": {
                "label": "AIHWKIT Analog Conv",
                "family": "AIHWKIT Analog",
                "variant": "Conv",
                "weights": "aihwkit_conv_model.pt",
                "is_snn": True,
                "is_aihwkit": True,
                "builder": lambda: AIHWKITSNNConv(num_steps=15),
            },
        }
    )


# A CPU-only aihwkit build can't move tile state to CUDA, so pin those
# to CPU even when CUDA is available.
default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")
loaded_models = {}
model_devices = {}


def device_for(entry):
    return cpu_device if entry["is_aihwkit"] else default_device


for name, entry in MODEL_REGISTRY.items():
    weight_path = PROJECT_ROOT / entry["weights"]
    if not weight_path.exists():
        print(f"[skip] {name}: {weight_path} not found")
        continue
    try:
        dev = device_for(entry)
        model = entry["builder"]().to(dev)
        state = torch.load(str(weight_path), map_location=dev, weights_only=False)
        model.load_state_dict(state)
        model.eval()
        loaded_models[name] = model
        model_devices[name] = dev
        print(f"[ok]   {name}: loaded from {entry['weights']} on {dev}")
    except Exception as e:
        print(f"[fail] {name}: {e}")

print(
    f"\n{len(loaded_models)}/{len(MODEL_REGISTRY)} models loaded "
    f"(default device: {default_device})\n"
)


transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "L":
        img = img.convert("L")
    tensor = transform(img).unsqueeze(0)
    return tensor, img


def run_model(name, img_tensor):
    model = loaded_models[name]
    entry = MODEL_REGISTRY[name]
    dev = model_devices[name]

    with torch.no_grad():
        out = model(img_tensor.to(dev))

    # SNNs return [T, B, 10] spike trains, sum over time to get logits.
    if entry["is_snn"]:
        logits = out.sum(dim=0)
    else:
        logits = out

    probs = F.softmax(logits, dim=1).squeeze().tolist()
    prediction = int(logits.argmax(dim=1).item())
    confidence = max(probs)

    return {
        "model": name,
        "label": entry["label"],
        "family": entry["family"],
        "variant": entry["variant"],
        "prediction": prediction,
        "confidence": f"{confidence:.2%}",
        "probabilities": {str(i): round(p, 4) for i, p in enumerate(probs)},
    }


def _render_family_plot(group):
    n = len(group)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)

    for ax, res in zip(axes[0], group):
        probs = [res["probabilities"][str(d)] for d in range(10)]
        pred = res["prediction"]
        colors = ["#667eea" if d == pred else "#ccc" for d in range(10)]
        ax.bar(range(10), probs, color=colors)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(10))
        ax.set_title(f"{res['label']}\npred={pred} ({res['confidence']})", fontsize=10)
        ax.set_xlabel("Digit")

    axes[0][0].set_ylabel("Probability")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    buf.close()
    return b64


def make_family_plots(results):
    grouped = {}
    for res in results:
        grouped.setdefault(res["family"], []).append(res)

    return [
        {"family": family, "image": _render_family_plot(group)}
        for family, group in grouped.items()
    ]


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/models", methods=["GET"])
def list_models():
    out = []
    for name, entry in MODEL_REGISTRY.items():
        out.append(
            {
                "id": name,
                "label": entry["label"],
                "family": entry["family"],
                "variant": entry["variant"],
                "loaded": name in loaded_models,
            }
        )
    return jsonify(out)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    selected = request.form.get("models", "all")
    if selected == "all":
        model_names = list(loaded_models.keys())
    else:
        model_names = [m.strip() for m in selected.split(",")]
        bad = [m for m in model_names if m not in loaded_models]
        if bad:
            return jsonify({"error": f"Models not loaded: {bad}"}), 400

    try:
        img_tensor, _ = preprocess(file.read())
        results = [run_model(name, img_tensor) for name in model_names]
        return jsonify(
            {
                "success": True,
                "results": results,
                "plots": make_family_plots(results),
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
