"""FashionMNISTModel quantized at act_bits=8 (bits_bias=16) instead of the
16-bit-activation target (fashion_quantize.py, act_bits=16/bias_bits=32).

The point of this variant: X_BITS=8 is what the already-deployed ZCU102
bitstream (run/work_pynq/pynq_deploy/design_1.bit) was actually synthesized
for - the 16-bit version needs a brand new Vivado build (see
run/work_pynq/mnist_fashion_config/README.md) before it can run on real
hardware at all. This one can, in principle, run on the bitstream that's
already there.

    python -m deepsocflow.py.brevitas.fashion_quantize_int8
"""
import os

from deepsocflow.py.brevitas.fashion_quantize import MODEL_DIR, main

GRAPH_JSON_PATH = os.path.join(MODEL_DIR, "fashion_graph_int8.json")


if __name__ == '__main__':
    main(act_bits=8, bias_bits=16, graph_json_path=GRAPH_JSON_PATH)
