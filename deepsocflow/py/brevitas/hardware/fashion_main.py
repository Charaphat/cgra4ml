"""Drives the real, trained FashionMNISTModel all the way to RTL simulation.

    python deepsocflow/py/brevitas/hardware/fashion_main.py --sim xsim

Mirrors conv_main.py's structure. Everything up to this point (fashion_quantize.py)
only validated the model in Python (PTQ + sim.py cross-check); this is the first
time it goes through the actual engine-layout export and a real RTL simulation.
"""
import argparse
import os

import numpy as np
import torch

BREV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--sim', default='xsim', choices=['xsim', 'verilator', 'none'])
    parser.add_argument('--sim-path', default='/home/software/Xilinx/Vivado/2023.2/bin/')
    args = parser.parse_args()

    from deepsocflow.py.brevitas.fashion_mnist import load
    from deepsocflow.py.brevitas.fashion_quantize import GRAPH_JSON_PATH
    from deepsocflow.py.brevitas.hardware.hardware import Hardware
    from deepsocflow.py.brevitas.simulation.sim import FixedPointModel

    assert os.path.exists(GRAPH_JSON_PATH), (
        f"{GRAPH_JSON_PATH} not found - run "
        f"`python -m deepsocflow.py.brevitas.fashion_quantize` first")

    X, Y = load()
    checkpoint = torch.load(
        os.path.join(BREV_DIR, 'model', 'fashion_cnn.pt'), map_location='cpu')
    mean, std = checkpoint['mean'], checkpoint['std']
    x_rtl = ((X[:args.batch].astype('float32') / 255.0 - mean) / std)[:, None, :, :]

    fp = FixedPointModel(GRAPH_JSON_PATH)
    fp.load_int_weights(GRAPH_JSON_PATH)
    x_int = fp.quantize_input(x_rtl)
    out_int = fp.forward(x_int)
    print(f"int output shape: {out_int.shape}")
    fp.print_graph()

    # ACC_WIDTH = K_BITS + X_BITS + clog2(KH*KW*CM), CM padded to the engine's
    # RAM_WEIGHTS_DEPTH capacity (512 by default) - at X_BITS=16 with 3x3
    # kernels that's 8+16+clog2(9*512)=8+16+13=37, well over the 32-bit
    # default (same class of bound StageL's STAGE_BITS_SUM tuned around, just
    # bigger here because of the 3x3 kernels). 64 gives comfortable margin.
    #
    # ram_edges_depth: holds the row overlap a KH>1 kernel needs between
    # blocks - conv.py's stages needed 3584 at 8x8; FashionMNIST's 28x28
    # images and 3x3 kernels need this recomputed, not assumed - EDGES =
    # cm_max * XW must stay <= this, so size it from the real channel/width
    # this model actually uses (64 channels * 28 width, with margin).
    hw = Hardware(
        processing_elements=(8, 24),
        bits_input=16, bits_weights=8, bits_bias=32, bits_sum=64,
        ram_edges_depth=32768,
        axi_width=128,
        valid_prob=1, ready_prob=1,
        data_dir=os.path.relpath(os.path.join(BREV_DIR, 'vectors_fashion')))

    from deepsocflow.py.brevitas.export.export import export_rtl
    from deepsocflow.py.brevitas.export.rtl_export import verify_inference

    result = export_rtl(fp, hw, x_rtl, batch_size=x_rtl.shape[0])
    print(f"exported {len(result['files'])} RTL files to {hw.DATA_DIR}")

    hw.export_json()
    hw.export()

    if args.sim == 'none':
        print("skipping simulation (--sim none)")
        return

    verify_inference(None, hw, SIM=args.sim, SIM_PATH=args.sim_path)
    print("FashionMNISTModel: RTL simulation PASSED")


if __name__ == '__main__':
    main()
