"""Exports the quantized FashionMNISTModel to ONNX and executes the exported
graph with onnxruntime (not brevitas), reporting how it compares to the
original float model and to brevitas's own in-memory forward pass - a check
on the EXPORT itself, not just brevitas's Python-side fake quantization
(already cross-checked against sim.py separately).

Exports via a PLAIN torch.onnx.export rather than brevitas's export_qonnx,
not brevitas's compact QONNX custom-`Quant`-node format - see the module-level
note below for why, and what that trade-off is.

    python -m deepsocflow.py.brevitas.fashion_qonnx_test
"""
import os

import numpy as np
import torch
from torch.onnx import register_custom_op_symbolic

from deepsocflow.py.brevitas.fashion_mnist import load
from deepsocflow.py.brevitas.fashion_model import FashionMNISTModel
from deepsocflow.py.brevitas.fashion_quantize import load_flat_from_nested
from deepsocflow.py.brevitas.quantization.ptq import quantized_model

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "fashion_cnn.pt")
ONNX_PATH = os.path.join(MODEL_DIR, "fashion.onnx")

OPSET = 18


# brevitas's own round-half-to-even implementation (RescalingIntQuant's
# int_quant, brevitas/core/quant/int_base.py) uses trunc + bitwise ops on
# integer tensors to compute the tie-break - the exact same kind of bit
# trick this project's own C `shift_round` macro uses for the same purpose.
# torch 2.12.1's legacy TorchScript ONNX exporter has no symbolic for
# aten::trunc at any opset, and its built-in bitwise_or/and/not symbolics
# explicitly refuse non-boolean (integer) operands - both real gaps in this
# torch version, not anything wrong in this project's own code. Registering
# straightforward symbolics (trunc via sign*floor(abs(x)); the bitwise ops
# via ONNX's own BitwiseOr/And/Not, native since opset 18) is the standard,
# safe way to bridge an ONNX-export gap like this - it does not touch torch
# or brevitas's installed packages, and CPython/ORT still compute the exact
# op behavior either way.
#
# QONNX vs plain ONNX: brevitas's own export_qonnx (QONNXManager) hits these
# same gaps INSIDE its custom Quant-node export handler and fails even with
# these symbolics patched (its handler emits extra integer bitwise ops of its
# own on top). A plain torch.onnx.export on the quantized model bypasses that
# handler entirely, tracing straight through brevitas's real (decomposed)
# round/clip/mul arithmetic instead of collapsing it into a compact `Quant`
# custom op - the resulting graph is standard ONNX (executable by plain
# onnxruntime, no `qonnx` package needed) rather than QONNX's interchange
# format, but computes the identical fake-quantized forward pass.
def _register_onnx_export_workarounds():
    def trunc_symbolic(g, self):
        sign = g.op('Sign', self)
        absval = g.op('Abs', self)
        floored = g.op('Floor', absval)
        return g.op('Mul', sign, floored)

    def bitwise_or_symbolic(g, self, other):
        return g.op('BitwiseOr', self, other)

    def bitwise_and_symbolic(g, self, other):
        return g.op('BitwiseAnd', self, other)

    def bitwise_not_symbolic(g, self):
        return g.op('BitwiseNot', self)

    register_custom_op_symbolic('aten::trunc', trunc_symbolic, OPSET)
    register_custom_op_symbolic('aten::bitwise_or', bitwise_or_symbolic, OPSET)
    register_custom_op_symbolic('aten::bitwise_and', bitwise_and_symbolic, OPSET)
    register_custom_op_symbolic('aten::bitwise_not', bitwise_not_symbolic, OPSET)


class _LogitsOnly(torch.nn.Module):
    """qm(x) returns an IntQuantTensor/QuantTensor wrapper, not a plain
    Tensor - torch.onnx.export needs a plain-Tensor-returning module."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        return out.value if hasattr(out, 'value') else out


def main():
    import onnxruntime as ort

    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    nested = FashionMNISTModel(num_classes=10)
    nested.load_state_dict(checkpoint['model_state_dict'])
    nested.eval()
    flat = load_flat_from_nested(nested)
    flat.eval()

    X, Y = load()
    mean, std = checkpoint['mean'], checkpoint['std']

    rng = torch.Generator().manual_seed(0)
    calib_idx = torch.randperm(len(X), generator=rng)[:512].numpy()
    x_calib = torch.from_numpy(
        (X[calib_idx].astype('float32') / 255.0 - mean) / std).unsqueeze(1)

    residuals = {'r1_conv2': 'stem_conv', 'r2_conv2': 'conv2_conv'}
    qm = quantized_model(flat, weight_bits=8, bias_bits=32, act_bits=16,
                         residuals=residuals)
    qm.quantization(x_calib)
    qm.eval()

    # --- export --------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    _register_onnx_export_workarounds()
    wrapper = _LogitsOnly(qm)
    torch.onnx.export(
        wrapper, x_calib[:1], ONNX_PATH, opset_version=OPSET,
        input_names=['input'], output_names=['output'],
        dynamo=False,
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
    print(f"exported ONNX model to {ONNX_PATH}")

    sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    print(f"ONNX graph: input='{input_name}', "
          f"{len(sess.get_inputs())} input(s), {len(sess.get_outputs())} output(s)")

    # --- evaluate on a held-out sample -----------------------------------
    eval_idx = torch.randperm(len(X), generator=rng)[512:2512].numpy()
    x_eval = ((X[eval_idx].astype('float32') / 255.0 - mean) / std)[:, None, :, :]
    y_eval = Y[eval_idx].astype('int64')

    with torch.no_grad():
        logits_float = nested(torch.from_numpy(x_eval)).numpy()
        logits_brevitas = qm(torch.from_numpy(x_eval))
        logits_brevitas = (logits_brevitas.value if hasattr(logits_brevitas, 'value')
                            else logits_brevitas).numpy()

    (logits_onnx,) = sess.run(None, {input_name: x_eval})

    preds_float = logits_float.argmax(-1)
    preds_brevitas = logits_brevitas.argmax(-1)
    preds_onnx = logits_onnx.argmax(-1)

    acc_float = float((preds_float == y_eval).mean())
    acc_brevitas = float((preds_brevitas == y_eval).mean())
    acc_onnx = float((preds_onnx == y_eval).mean())

    max_err_onnx_vs_brevitas = float(np.abs(logits_onnx - logits_brevitas).max())
    flips_onnx_vs_brevitas = int((preds_onnx != preds_brevitas).sum())
    flips_onnx_vs_float = int((preds_onnx != preds_float).sum())

    print(f"\n=== ONNX vs float model report ({len(x_eval)} held-out samples) ===")
    print(f"float model accuracy:        {acc_float:.4f}")
    print(f"brevitas quantized accuracy: {acc_brevitas:.4f}")
    print(f"ONNX-executed accuracy:      {acc_onnx:.4f}")
    print(f"ONNX vs brevitas: max abs logit diff = {max_err_onnx_vs_brevitas:.3e}, "
          f"{flips_onnx_vs_brevitas}/{len(x_eval)} predictions differ")
    print(f"ONNX vs float:    {flips_onnx_vs_float}/{len(x_eval)} predictions differ")

    assert flips_onnx_vs_brevitas == 0, (
        f"{flips_onnx_vs_brevitas} predictions disagree between the ONNX "
        f"export and brevitas's own forward pass - the export changed the "
        f"model's decisions, not just introduced float noise")
    assert max_err_onnx_vs_brevitas < 1e-3, (
        f"ONNX vs brevitas max abs logit diff {max_err_onnx_vs_brevitas} is "
        f"larger than float32 rounding noise")
    print("\nPASSED: the exported ONNX graph reproduces brevitas's own "
          "quantized forward pass (0 prediction flips).")


if __name__ == '__main__':
    main()
