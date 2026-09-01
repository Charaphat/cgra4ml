"""Flattens the trained FashionMNISTModel (fashion_model.py) into individual
top-level children in forward-execution order, then runs it through
quantized_model (ptq.py) for PTQ with int8 weights and 16-bit activations.

quantized_model walks net.named_children() linearly, pairing each compute
layer with the activation/pool/flatten/softmax modules that follow it in that
same flat list - it has no notion of nn.Sequential or a nested submodule like
ResidualBlock, both of which fashion_model.py uses for training-time
readability. FlatFashionMNISTModel exposes the exact same computation as
FashionMNISTModel, but as one flat list of children, with weights copied over
from a trained FashionMNISTModel checkpoint (load_flat_from_nested).

    python -m deepsocflow.py.brevitas.fashion_quantize

Residual topology: r1's skip source is stem (already pooled by the time
residual1 sees it - MaxPool2d), r2's skip source is conv2 (already pooled by
AvgPool2d). This shape - a residual sourced from a bundle that itself pools -
was unverified anywhere in this backend until StageM (deepsocflow/py/
brevitas/conv.py) proved it via a real RTL simulation (`Bundle 0/1/2/3,
Error: 0`) that tile_write's add_buffers write already uses the POOLED value
for a bundle that pools (it's a shared subroutine called from the pooling
call site with the already-reduced result) - no runtime.h change was needed.
"""
import os

import torch
import torch.nn as nn

from deepsocflow.py.brevitas.fashion_model import FashionMNISTModel

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "fashion_cnn.pt")
GRAPH_JSON_PATH = os.path.join(MODEL_DIR, "fashion_graph.json")


class FlatFashionMNISTModel(nn.Module):
    """Same computation as FashionMNISTModel, flattened - see module docstring."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(32)
        self.stem_act = nn.ReLU()
        self.stem_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.r1_conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.r1_bn1 = nn.BatchNorm2d(32)
        self.r1_act1 = nn.LeakyReLU(negative_slope=0.125)
        self.r1_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.r1_bn2 = nn.BatchNorm2d(32)
        self.r1_act_pre_add = nn.ReLU()          # matches stem_act's type
        self.r1_act_post_add = nn.LeakyReLU(negative_slope=0.125)

        self.conv2_conv = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv2_act = nn.LeakyReLU(negative_slope=0.125)
        self.conv2_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.r2_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.r2_bn1 = nn.BatchNorm2d(64)
        self.r2_act1 = nn.ReLU()
        self.r2_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.r2_bn2 = nn.BatchNorm2d(64)
        self.r2_act_pre_add = nn.LeakyReLU(negative_slope=0.125)  # matches conv2_act's type
        self.r2_act_post_add = nn.ReLU()

        self.gap = nn.AvgPool2d(kernel_size=7)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 32)
        self.fc1_act = nn.LeakyReLU(negative_slope=0.125)
        self.fc2 = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.stem_pool(self.stem_act(self.stem_bn(self.stem_conv(x))))
        skip = x
        x = self.r1_act1(self.r1_bn1(self.r1_conv1(x)))
        x = self.r1_act_pre_add(self.r1_bn2(self.r1_conv2(x)))
        x = self.r1_act_post_add(x + skip)

        x = self.conv2_pool(self.conv2_act(self.conv2_bn(self.conv2_conv(x))))
        skip = x
        x = self.r2_act1(self.r2_bn1(self.r2_conv1(x)))
        x = self.r2_act_pre_add(self.r2_bn2(self.r2_conv2(x)))
        x = self.r2_act_post_add(x + skip)

        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc1_act(self.fc1(x))
        logits = self.fc2(x)
        return logits

    def predict_proba(self, x):
        return self.softmax(self.forward(x))


# {flat attribute name: nested checkpoint's own state_dict key prefix} - only
# parametrized layers (conv/bn/linear) need an entry; activations/pools carry
# no weights.
_WEIGHT_NAME_MAP = {
    'stem_conv': 'stem.0', 'stem_bn': 'stem.1',
    'r1_conv1': 'residual1.conv1', 'r1_bn1': 'residual1.bn1',
    'r1_conv2': 'residual1.conv2', 'r1_bn2': 'residual1.bn2',
    'conv2_conv': 'conv2.0', 'conv2_bn': 'conv2.1',
    'r2_conv1': 'residual2.conv1', 'r2_bn1': 'residual2.bn1',
    'r2_conv2': 'residual2.conv2', 'r2_bn2': 'residual2.bn2',
    'fc1': 'classifier.1', 'fc2': 'classifier.3',
}


def load_flat_from_nested(nested: FashionMNISTModel) -> FlatFashionMNISTModel:
    """Copies a trained FashionMNISTModel's weights into a
    FlatFashionMNISTModel with the same architecture, so the flat model
    computes byte-for-byte the same function the checkpoint was trained as."""
    num_classes = nested.classifier[-1].out_features
    flat = FlatFashionMNISTModel(num_classes=num_classes)
    nested_sd = nested.state_dict()
    flat_sd = flat.state_dict()
    for flat_prefix, nested_prefix in _WEIGHT_NAME_MAP.items():
        for suffix in ('weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked'):
            key = f'{nested_prefix}.{suffix}'
            if key in nested_sd:
                flat_sd[f'{flat_prefix}.{suffix}'] = nested_sd[key]
    flat.load_state_dict(flat_sd)
    return flat


def _verify_flat_matches_nested(nested, flat, x):
    """The flattening is a pure refactor - the two models must agree exactly
    on real input, not just have the right shapes, before anything downstream
    (quantization, calibration) can be trusted."""
    nested.eval()
    flat.eval()
    with torch.no_grad():
        out_nested = nested(x)
        out_flat = flat(x)
    max_err = (out_nested - out_flat).abs().max().item()
    assert max_err == 0.0, (
        f"FlatFashionMNISTModel disagrees with the trained FashionMNISTModel "
        f"(max abs err {max_err}) - the flattening changed the computation, "
        f"it should only have changed the module tree shape")


def main(act_bits=16, bias_bits=32, graph_json_path=GRAPH_JSON_PATH):
    """act_bits/bias_bits default to this backend's original 16-bit-activation
    target (bias_bits=32 needed there - see the comment below). Pass
    act_bits=8, bias_bits=16 for a variant that fits the already-deployed
    ZCU102 bitstream (X_BITS=8) - see fashion_quantize_int8.py."""
    from deepsocflow.py.brevitas.quantization.ptq import quantized_model
    from deepsocflow.py.brevitas.fashion_mnist import load

    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    nested = FashionMNISTModel(num_classes=10)
    nested.load_state_dict(checkpoint['model_state_dict'])
    nested.eval()

    flat = load_flat_from_nested(nested)
    flat.eval()

    X, Y = load()
    mean, std = checkpoint['mean'], checkpoint['std']

    # Calibration set: a random (class-varied) 512-sample slice, not the
    # first 64 - calibration_mode collects per-tensor activation statistics,
    # and a bigger, better-mixed sample gives steadier scales.
    rng = torch.Generator().manual_seed(0)
    calib_idx = torch.randperm(len(X), generator=rng)[:512].numpy()
    x_sample = torch.from_numpy(
        (X[calib_idx].astype('float32') / 255.0 - mean) / std).unsqueeze(1)

    _verify_flat_matches_nested(nested, flat, x_sample)
    print("Flattened model matches the trained nested model exactly on "
          f"{x_sample.shape[0]} real samples.")

    residuals = {'r1_conv2': 'stem_conv', 'r2_conv2': 'conv2_conv'}

    # bias_bits needs to scale with act_bits: bias_frac is derived
    # (input_frac + weight_frac), not chosen independently, and a wider
    # act_bits pushes input_frac up - at act_bits=16 that pushed bias_frac to
    # 19-21, which silently saturates every real trained bias at this
    # backend's usual bias_bits=16 (representable magnitude ~0.06), hence
    # bias_bits=32 there. At act_bits=8, bias_frac lands back around 10-13,
    # comfortably fitting bias_bits=16 - verified directly against the
    # exported graph JSON before choosing each. Hardware allows bits_bias in
    # {8,16,32} either way (hardware.py's own assert), so neither needs an
    # RTL/firmware change.
    qm = quantized_model(flat, weight_bits=8, bias_bits=bias_bits, act_bits=act_bits,
                         residuals=residuals)
    qm.quantization(x_sample)
    qm.eval()

    os.makedirs(MODEL_DIR, exist_ok=True)
    qm.export_graph_json(x_sample[:4], graph_json_path)
    print(f"exported graph json to {graph_json_path}")
    qm.print_graph()

    with torch.no_grad():
        preds = qm(x_sample).argmax(dim=-1)
    acc = (preds == torch.from_numpy(Y[calib_idx].astype('int64'))).float().mean().item()
    print(f"quantized model accuracy on {x_sample.shape[0]} calibration "
          f"samples: {acc:.4f}")

    # A separate, disjoint sample - measures generalization, not just how
    # well scales fit the data they were calibrated on.
    eval_idx = torch.randperm(len(X), generator=rng)[512:2512].numpy()
    x_eval = torch.from_numpy(
        (X[eval_idx].astype('float32') / 255.0 - mean) / std).unsqueeze(1)
    y_eval = torch.from_numpy(Y[eval_idx].astype('int64'))
    with torch.no_grad():
        preds_eval = qm(x_eval).argmax(dim=-1)
        preds_float = flat(x_eval).argmax(dim=-1)
    acc_eval = (preds_eval == y_eval).float().mean().item()
    acc_float = (preds_float == y_eval).float().mean().item()
    n_flips = int((preds_eval != preds_float).sum())
    print(f"quantized model accuracy on {x_eval.shape[0]} held-out samples: "
          f"{acc_eval:.4f}  (float model: {acc_float:.4f}, "
          f"{n_flips}/{x_eval.shape[0]} predictions differ from float)")

    # --- sim.py cross-check: the integer FixedPointModel must reproduce
    # brevitas's own quantized forward pass exactly, not approximately - the
    # same bar every stage in conv.py is held to (see conv_main.py's "vs
    # brevitas" check). Confirms this real, deeper, residual+pool+16-bit
    # model doesn't expose a new sim.py/adapter gap the smaller synthetic
    # bring-up stages didn't happen to exercise.
    from deepsocflow.py.brevitas.simulation.sim import FixedPointModel

    x_check = x_eval[:8]
    fp = FixedPointModel(graph_json_path)
    fp.load_int_weights(graph_json_path)
    x_int = fp.quantize_input(x_check)
    fp.forward(x_int)

    with torch.no_grad():
        qm_out = qm(x_check)
    qm_ref = (qm_out.value if hasattr(qm_out, 'value') else qm_out).detach().numpy()

    import numpy as np
    sim_deq = np.asarray(fp.softmax_out, dtype=np.float64)
    max_err = float(abs(sim_deq - qm_ref).max())
    n_pred_flips = int((sim_deq.argmax(-1) != qm_ref.argmax(-1)).sum())
    print(f"sim.py vs brevitas: max abs err {max_err:.3e} (softmax, float64 "
          f"vs float32), {n_pred_flips}/{len(sim_deq)} predictions differ")
    assert n_pred_flips == 0, f"{n_pred_flips} predictions disagree with brevitas"
    assert max_err < 1e-5, (
        f"softmax max abs err {max_err} is larger than float-rounding noise")
    print("PASSED: sim.py's integer FixedPointModel matches brevitas's own "
          "quantized forward pass exactly.")

    return qm, flat, nested


if __name__ == '__main__':
    main()
