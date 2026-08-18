"""Drives the act_bits=8 FashionMNISTModel variant (fashion_quantize_int8.py)
to RTL simulation, at the SAME hardware config as the already-deployed ZCU102
bitstream (bits_input=8, bits_bias=16, bits_sum=32) - unlike fashion_main.py's
16-bit-activation version, this one needs no new Vivado build to actually run
on real hardware.

    python deepsocflow/py/brevitas/hardware/fashion_main_int8.py --sim xsim
"""
import argparse
import os

import torch

BREV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--sim', default='xsim', choices=['xsim', 'verilator', 'none'])
    parser.add_argument('--sim-path', default='/home/software/Xilinx/Vivado/2023.2/bin/')
    args = parser.parse_args()

    from deepsocflow.py.brevitas.fashion_mnist import load
    from deepsocflow.py.brevitas.fashion_quantize_int8 import GRAPH_JSON_PATH
    from deepsocflow.py.brevitas.hardware.hardware import Hardware
    from deepsocflow.py.brevitas.simulation.sim import FixedPointModel

    assert os.path.exists(GRAPH_JSON_PATH), (
        f"{GRAPH_JSON_PATH} not found - run "
        f"`python -m deepsocflow.py.brevitas.fashion_quantize_int8` first")

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

    # ACC_WIDTH = K_BITS + X_BITS + clog2(KH*KW*CM), CM padded to
    # RAM_WEIGHTS_DEPTH (512) - at X_BITS=8 that's 8+8+clog2(9*512)=29,
    # comfortably under the project's usual bits_sum=32 default (unlike the
    # act_bits=16 variant, which needed 64 - see fashion_main.py). bits_bias=16
    # is likewise this backend's usual default, not the =32 the 16-bit
    # variant needed - see fashion_quantize.py's comment on why bias_bits
    # scales with act_bits.
    #
    # ram_edges_depth: EDGES = cm_max * XW = 512 * 28 = 14336 for this image
    # size - reused fashion_main.py's 32768 for margin (image-size-driven,
    # not activation-bit-width-driven, so it doesn't shrink at X_BITS=8).
    hw = Hardware(
        processing_elements=(8, 24),
        bits_input=8, bits_weights=8, bits_bias=16, bits_sum=32,
        ram_edges_depth=32768,
        axi_width=128,
        valid_prob=1, ready_prob=1,
        data_dir=os.path.relpath(os.path.join(BREV_DIR, 'vectors_fashion_int8')))

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
    print("FashionMNISTModel (int8): RTL simulation PASSED")


if __name__ == '__main__':
    main()
