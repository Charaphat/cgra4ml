"""Tests for the branching-main-path fix: a bundle whose output feeds two
downstream bundles as MAIN input (not a residual add), each needing different
engine tiling - the ResNet downsample-block shape (conv.py::StageJ). Before
this fix, deepsocflow/c/runtime.h's tile_write wrote a producer's output into
exactly one buffer, tiled for exactly one consumer; the second consumer
silently read a buffer laid out for the first.

These tests drive the real export path (build_bundles -> _export_bundles),
the same functions the RTL testbench and PYNQ driver consume, rather than
re-deriving the allocator's behaviour - see test_brevitas_adapter.py's own
tests for why: several past defects here were only visible once the real
config_fw.h writer ran, not from reading the adapter's attributes alone.
"""
import json

import numpy as np
import pytest


def _stage_j_fixedpointmodel(tmp_path):
    """Builds conv.py::StageJ (the branching bring-up stage) through the real
    quantization -> export_graph_json -> FixedPointModel path, forward()ed so
    .trace is populated - the same sequence conv_main.py uses."""
    pytest.importorskip("torch")
    from deepsocflow.py.brevitas.conv import (
        build_model, stage_data, prime_batchnorm, STAGE_RESIDUALS, STAGE_BRANCHES)
    from deepsocflow.py.brevitas.quantization.ptq import quantized_model
    from deepsocflow.py.brevitas.simulation.sim import FixedPointModel

    X, _, x_rtl = stage_data('j')
    model = build_model('j')
    prime_batchnorm(model, X)
    qm = quantized_model(model, weight_bits=8, bias_bits=16,
                         residuals=STAGE_RESIDUALS.get('j'),
                         branches=STAGE_BRANCHES.get('j'))
    qm.quantization(X)
    qm.eval()

    graph_json = tmp_path / 'stage_j_graph.json'
    qm.export_graph_json(x_rtl, str(graph_json))

    fp = FixedPointModel(str(graph_json))
    fp.load_int_weights(str(graph_json))
    x_int = fp.quantize_input(x_rtl)
    fp.forward(x_int)
    return fp


def test_branches_kwarg_produces_a_two_main_consumer_topology(tmp_path):
    """Confirms the topology this whole fix targets actually gets built:
    bundle0 (stem) must be the declared main input of BOTH bundle1
    (conv_shortcut, default predecessor) and bundle2 (conv_main, via
    branches={'conv_main': 'stem'}) - not a residual add, an actual second
    MAIN consumer. If this is wrong, every other test in this file is
    checking a topology that doesn't exercise the bug at all."""
    fp = _stage_j_fixedpointmodel(tmp_path)
    assert fp.bundles['bundle1']['input'] == 'bundle0'
    assert fp.bundles['bundle2']['input'] == 'bundle0'
    assert fp.bundles['bundle2']['skip_from'] == 'bundle1'


def test_stage_j_bundle0_gets_two_distinct_output_buffers(tmp_path, monkeypatch):
    """The core regression: stem's two consumers (1x1 shortcut, 3x3 main
    path) need genuinely different engine tilings (CM/X_PAD differ - a 1x1
    kernel needs no row padding, a 3x3 one does), so the allocator must give
    them two SEPARATE buffer slots, not the pre-fix single shared one."""
    from deepsocflow.py.brevitas.export.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware.hardware import Hardware
    from deepsocflow.py.brevitas.export.rtl_export import _export_bundles

    fp = _stage_j_fixedpointmodel(tmp_path)
    data_dir = tmp_path / 'vectors'
    data_dir.mkdir(parents=True, exist_ok=True)
    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(data_dir))
    bundles = build_bundles(fp, hw)

    monkeypatch.chdir(tmp_path)
    _export_bundles(hw, None)

    stem = bundles[0]
    assert stem.ib_out != -1 and stem.ib_out2 != -1, (
        "stem must have both a primary and a second main consumer group")
    assert stem.out_buffer_idx != stem.out_buffer_idx2, (
        "the two consumer groups must land in DIFFERENT buffer slots")
    assert {stem.ib_out, stem.ib_out2} == {1, 2}, (
        "the two consumer groups' representatives must be bundle 1 "
        "(conv_shortcut) and bundle 2 (conv_main)")


def test_stage_j_consumers_read_their_own_matching_buffer(tmp_path, monkeypatch):
    """Each consumer's in_buffer_idx must resolve to the SPECIFIC buffer
    slot the producer tiled for IT, not just 'the producer's buffer' as if
    there could only ever be one - the exact mechanism the pre-fix
    `BUNDLES[prev_ib].out_buffer_idx` always got wrong for a branching
    producer (it always pointed both consumers at the same, singular slot)."""
    from deepsocflow.py.brevitas.export.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware.hardware import Hardware
    from deepsocflow.py.brevitas.export.rtl_export import _export_bundles

    fp = _stage_j_fixedpointmodel(tmp_path)
    data_dir = tmp_path / 'vectors'
    data_dir.mkdir(parents=True, exist_ok=True)
    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(data_dir))
    bundles = build_bundles(fp, hw)

    monkeypatch.chdir(tmp_path)
    _export_bundles(hw, None)

    # in_buffer_idx is resolved only in the header-writing loop, not stored
    # back on the Python bundle object - read it the way the real firmware
    # and PYNQ driver do, from the emitted config.json.
    stem = bundles[0]
    config = json.loads((tmp_path / 'config.json').read_text())
    conv_shortcut_json, conv_main_json = config['bundles'][1], config['bundles'][2]

    assert conv_shortcut_json['in_buffer_idx'] != conv_main_json['in_buffer_idx'], (
        "the two consumers of the same producer must read DIFFERENT buffers")
    assert {conv_shortcut_json['in_buffer_idx'], conv_main_json['in_buffer_idx']} == \
        {stem.out_buffer_idx, stem.out_buffer_idx2}, (
        "each consumer must read exactly the buffer slot its own producer "
        "allocated for it")


def test_n_branch_bundles_counts_the_branching_bundle_exactly(tmp_path, monkeypatch):
    """N_BRANCH_BUNDLES guards every read of ib_out2/out_buffer_idx2/o_bytes2/
    o_words2 in runtime.h - a legacy-exporter build never defines it at all,
    which is what makes those fields' C zero-default safe there (same
    pattern as N_LUTS/ca_lut_idx). It must equal exactly the number of
    bundles that actually got a second consumer group - 1 for StageJ (only
    stem branches), not 0 (which would compile the fix out and silently
    revert to single-buffer behaviour) and not more."""
    from deepsocflow.py.brevitas.export.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware.hardware import Hardware
    from deepsocflow.py.brevitas.export.rtl_export import _export_bundles

    fp = _stage_j_fixedpointmodel(tmp_path)
    data_dir = tmp_path / 'vectors'
    data_dir.mkdir(parents=True, exist_ok=True)
    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(data_dir))
    build_bundles(fp, hw)

    monkeypatch.chdir(tmp_path)
    _export_bundles(hw, None)

    fw_text = (tmp_path / 'config_fw.h').read_text()
    assert '#define N_BRANCH_BUNDLES 1' in fw_text

    config = json.loads((tmp_path / 'config.json').read_text())
    assert config['defines']['N_BRANCH_BUNDLES'] == 1
    branching = [b for b in config['bundles'] if b['ib_out2'] != -1]
    assert len(branching) == 1, (
        f"exactly one bundle (stem) should have ib_out2 set, got {len(branching)}")


def test_o_bytes2_matches_the_second_branchs_own_consumer_not_the_first(tmp_path, monkeypatch):
    """o_bytes/o_bytes2 must each be sized from THEIR OWN representative
    consumer's transfer geometry, not from BUNDLES[ib+1] regardless of which
    branch that happens to be (the pre-fix formula, which silently mis-sized
    the buffer whenever the branch it was sizing wasn't literally the next
    bundle in array order)."""
    from deepsocflow.py.brevitas.export.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware.hardware import Hardware
    from deepsocflow.py.brevitas.export.rtl_export import _export_bundles

    fp = _stage_j_fixedpointmodel(tmp_path)
    data_dir = tmp_path / 'vectors'
    data_dir.mkdir(parents=True, exist_ok=True)
    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(data_dir))
    bundles = build_bundles(fp, hw)

    monkeypatch.chdir(tmp_path)
    _export_bundles(hw, None)

    config = json.loads((tmp_path / 'config.json').read_text())
    stem_json = config['bundles'][0]

    assert stem_json['o_bytes'] > 0    # sanity: sized, not zero/unset
    assert stem_json['o_bytes2'] > 0
    # The two groups need genuinely different tiling (that's the whole point
    # of this fix), so their transfer sizes must differ too.
    assert stem_json['o_bytes'] != stem_json['o_bytes2'], (
        "primary and second branch buffers were sized identically - "
        "suspicious for two consumers with genuinely different kernel sizes")


def test_non_branching_stage_keeps_ib_out2_unset(tmp_path, monkeypatch):
    """Regression guard in the other direction: an ordinary linear-chain
    stage (no branches=) must never end up with ib_out2/out_buffer_idx2 set
    on any bundle - the grouping logic must not invent a second group where
    there is only ever one consumer per producer."""
    pytest.importorskip("torch")
    from deepsocflow.py.brevitas.conv import build_model, stage_data, prime_batchnorm
    from deepsocflow.py.brevitas.quantization.ptq import quantized_model
    from deepsocflow.py.brevitas.simulation.sim import FixedPointModel
    from deepsocflow.py.brevitas.export.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware.hardware import Hardware
    from deepsocflow.py.brevitas.export.rtl_export import _export_bundles

    X, _, x_rtl = stage_data('b')
    model = build_model('b')
    prime_batchnorm(model, X)
    qm = quantized_model(model, weight_bits=8, bias_bits=16)
    qm.quantization(X)
    qm.eval()

    graph_json = tmp_path / 'stage_b_graph.json'
    qm.export_graph_json(x_rtl, str(graph_json))
    fp = FixedPointModel(str(graph_json))
    fp.load_int_weights(str(graph_json))
    fp.forward(fp.quantize_input(x_rtl))

    data_dir = tmp_path / 'vectors'
    data_dir.mkdir(parents=True, exist_ok=True)
    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(data_dir))
    bundles = build_bundles(fp, hw)

    monkeypatch.chdir(tmp_path)
    _export_bundles(hw, None)

    for b in bundles:
        assert b.ib_out2 == -1
        assert b.out_buffer_idx2 == -1

    config = json.loads((tmp_path / 'config.json').read_text())
    assert config['defines']['N_BRANCH_BUNDLES'] == 0
    for b in config['bundles']:
        assert b['ib_out2'] == -1
        assert b['out_buffer_idx2'] == -1
