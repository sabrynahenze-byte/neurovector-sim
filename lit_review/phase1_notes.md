# Phase 1 Notes (Shared)

This mirrors Bryan’s outline so we can drop in quick notes/links.

## Problem Definition
- Potential applications: drone fire detection, aircraft damage detection, road inspection, license plate recognition, etc.
- Open: which one do we want to test as our benchmark?

## Emerging Memory Technologies
- Explore RRAM, MRAM, PCM, etc. with pros/cons (power, latency, endurance).
- Figure out how each could interface with PyTorch.

## Algorithm Selection
- Which NN type is best? SNN seems aligned with Prof. Ahmed’s push.
- PyTorch tools (Torch, snnTorch, IBM AIHWKit) worth exploring.

All algorithms for neuromorphic computing are based on SNNs. However, the model setup can vary depending on the application requirements. Below are tables of the applicable training algorithms and neuron models:

### Training Algorithms

| Algorithm | Description | Tools |
| --------- | ----------- | ----- |
| [ANN-SNN Conversion](ding_yu_tian_huang_ann-snn_conversion.pdf) | Start with a pre-trained standard ANN using backpropagation and then apply a conversion algorithm to obtain an SNN with comparable performance. The advantage of converting to an SNN is multifold: (1) Leverage pre-trained ANN to overcome difficulty of training SNN with time-consuming SNN-specific training methods (e.g. surrogate gradient descent); (2) lower energy consumption due to the event-driven nature of SNNs (discrete spikes for computation rather than continuous-valued activations); and (3) lower computational load by replacing mutliply-accumulate (MAC) operations with accumulate-only (AC) operations*. | [SpikingJelly ANN2SNN](https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based_en/ann2snn.html), [sinabs](https://sinabs.readthedocs.io/v3.0.4/), PyTorch + snnTorch |
| [SNN via Surrogate Gradients](jiang_chang_surrogate_gradient.pdf) | Start with an untrained SNN and apply surrogate gradient descent to train the neural network during backpropagation. The advantage of this method is to enable direct training of the SNN via gradient-based learning to overcome its inability to be trained using derivation (during backpropagation). Since the spike activation function of SNNs is non-differentiable**, it is replaced with a surrogate function that approximates a continuous and smooth gradient. | snntorch
| [Reservoir Computing (ESN/LSM)](dettori_matino_colla_speets.pdf) | A computational framework that maps input into higher-dimensional computational spaces through the dynamics of a fixed, non-linear system called a reservoir (from [Wikipedia](https://en.wikipedia.org/wiki/Reservoir_computing)), whose dynamic state at each time step is passed to the readout layer. The general flow is: Input signal -> input layer -> reservoir -> readout layer -> output signal -> training. The simple, linear readout layer is the only part of the network that is trained. | reservoirPy |

*In traditional ANNs [image](images/ann_diagram.png), there are two stages that utilize MAC operations: **forward propagation** (used in inference to predict data) and [**backpropagation**](https://en.wikipedia.org/wiki/Backpropagation) (used in training to reduce prediction). During forward propagation, input data moves forward through the network's layers from the beginning (input layer -> hidden layers -> output layer), where each neuron in a layer performs a weighted sum of all its inputs $\sum(x_i \cdot w_i)$. This sum is then passed through an activation function to produce the neuron's output. During backward propagation, input data (the error) moves backward from the end (output layer -> hidden layers -> input layer), where operations are performed on entire layers at once using the chain rule, which is efficiently implemented by vectorization with basic matrix and vector multiplication. Training SNNs is more complex because the discrete spikes are not differentiable, so different backpropagation algorithms need to be used. For computation, the computation savings primarily come from forward propagation, where the resource-intensive multiplication of MAC operations are replaced with fast pure addition of AC operations.

**The spiking function represents the core, non-differentiable spiking behavior and is often the Heaviside step function $$\begin{equation*}
H(t) =
\begin{cases}
    1, & t \ge 0, \\
    0, & t < 0
\end{cases}
\end{equation*}$$

### Neuron Models

| Model | Biological Realism | Computational Cost | Use Case | Tools |
| ----- | ------------------ | ------------------ | -------- | ----- |
| Leaky Integrate-and-Fire (LIF) | Low-Medium | Very low | Widely used in neuromorphic hardware (e.g. Loihi, TrueNorth) due to simplicity and ease of digital/analog implementation. Ideal for low-power simulation. | snnTorch, Norse, Brian2, SpikingJelly, NEST |
| Izhikevich | High (simple models) | Medium | Great trade-off between biological realism (when simulation size is manageable at < 10k neurons) and speed. Captures bursting, spiking, and adaptation. Used when dynamics like bursting/frequency adaptation matter. | Brian2, BindsNET, SpikingJelly |
| Hodgkin-Huxley | Very high | Very high | Accurate at the ion channel level, but computationally expensive. Not used in large-scale simulations; better for neuron-level or small networks. Use when modeling at the ion-channel level or simulating biophysical dynamics but not energy-efficient hardware emulation. | Brian2, NEURON |
| Spike Response Model (SRM) | Medium-high | Medium | Flexible mathematical model that uses kernels to define spike effects. Good for theoretical modeling or spike-timing studies. Less common in hardware. Use as an analytical model to explain post-synaptic potentials or train using timing-based learning (e.g., STDP) | Brian2, NEST |

| Application | Recommended Algorithm(s) | Supporting Literature |
|-------------|---------------------------|------------------------|
| **Drone Fire Detection / Road Inspection** | Hybrid: Spiking Encoder ➜ CNN Detector, or ANN→SNN Conversion | [Gao et al. (2023)](gao_et_al_ann-snn_conversion.pdf); Ding et al. (2021) |
| **Predictive Maintenance** | Reservoir Computing (ESN / Deep ESN); Native SNN (stretch) | Dettori et al. (2020); SNN review (2022) |
| **License Plate / Damage Detection** | CNN baseline ➜ SNN Conversion | Gao et al. (2023); Ding et al. (2021) |

## Device Evaluation and Testing
- Accuracy loss: simulate with/without non-idealities.
- Power and speed comparisons.
- Maybe price, if we find data.

---

Notes:
- I set this up in GitHub so it’s easier for everyone to add references as we go.
- When we split topics (tomorrow night or in the meeting next week), this doc will already give us a structure to plug into.

