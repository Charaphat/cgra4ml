"""Synthetic conv models for bringing Conv2d support up on the brevitas backend.

The dataset is a deliberately trivial one - 8x8 single-channel images holding
either a horizontal or a vertical bar - generated in-process so nothing has to
be downloaded, and small enough that a whole batch fits comfortably in one RTL
simulation.

Five models, one per bring-up stage, each adding exactly one new thing to the
datapath so a failure points at what was just introduced:

    stage_a   1x1 conv                    - a 1x1 conv is a per-pixel matmul, so
                                            this exercises only the NCHW->NHWC
                                            layout plumbing, with arithmetic
                                            simple enough to check by hand
    stage_b   3x3 'same' conv             - real conv math (padding, X_PAD)
    stage_c   3x3 stride-2 conv           - CSH/CSW striding (dropped on the CPU
                                            side; the engine stays stride-1)
    stage_d   + MaxPool2d                 - pooling
    stage_e   + flatten -> dense -> softmax  the full classifier

Only stage_e is trained. Stages a-d exist to prove the *datapath* is bit-exact,
which is a property of the export/simulate path rather than of the weights - so
they use seeded random weights and are never trained. Calibration still runs on
real bar images, so their quant scales come from realistic activation statistics
rather than from brevitas's fresh-init defaults.
"""
import os

import torch
import torch.nn as nn

from deepsocflow.py.brevitas.quantization.ptq import tf_same_padding

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "conv.pt")
QONNX_PATH = os.path.join(MODEL_DIR, "conv.onnx")
GRAPH_JSON_PATH = os.path.join(MODEL_DIR, "conv_graph.json")

IMG_SIZE = 8

# Same reasoning as xor.py's DEFAULT_SEED: nn.Conv2d/nn.Linear draw their initial
# weights at construction time, so the seed has to be set before the model is
# built, not inside train().
DEFAULT_SEED = 0


CHANNELS = 1


def make_bars(n_samples, size=IMG_SIZE, channels=CHANNELS, seed=DEFAULT_SEED):
    """(n,channels,size,size) float images and (n,) int labels.

    Label 0 is a horizontal bar (one full row set), label 1 is a vertical bar
    (one full column set); the bar is drawn into one randomly chosen channel,
    the way a coloured bar would appear in an RGB image (with the default of one
    channel there is only ever channel 0). Alternating labels
    rather than random ones keeps the two classes exactly balanced, which
    matters at the tiny batch sizes the RTL export uses - a 4-image export that
    happened to draw one class only would still pass while testing half the
    network's behaviour.

    A single channel is the natural shape for these images and is deliberately
    kept: CM_0 == 1 is the case that breaks the engine's C_CI counter, and
    adapter.py::pad_single_input_channel is what makes it work. Running every
    bring-up stage on grayscale therefore exercises that workaround, while the
    second layer of each model (CI = 4 or 8) covers the ordinary path.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n_samples, channels, size, size)
    y = torch.zeros(n_samples, dtype=torch.long)
    idx = torch.randint(0, size, (n_samples,), generator=g)
    ch = torch.randint(0, channels, (n_samples,), generator=g)
    for i in range(n_samples):
        if i % 2 == 0:
            x[i, ch[i], idx[i], :] = 1.0
        else:
            x[i, ch[i], :, idx[i]] = 1.0
            y[i] = 1
    return x, y


# Stage c is the exception: a stride-2 3x3 conv over an EVEN axis needs
# asymmetric padding under the engine's TF-'same' semantics, which torch's
# Conv2d cannot express - the two then disagree by one pixel, silently
# (ptq.py::_assert_stride_matches_engine refuses it). At an odd size TF's total
# padding comes out to k-1, exactly what torch's symmetric padding=1 gives, and
# the two agree. 7 rather than 8 for that stage only.
STAGE_IMG_SIZE = {'c': 7}


def stage_data(stage, n_samples=256, rtl_batch=4):
    """(X, Y, X_RTL) for a stage, at whatever input size that stage needs.

    X_RTL is the batch actually pushed through RTL simulation - four images,
    which is two of each class since make_bars alternates, so a passing run
    cannot have exercised only half the network's behaviour.
    """
    size = STAGE_IMG_SIZE.get(stage, IMG_SIZE)
    x, y = make_bars(n_samples, size=size)
    return x, y, x[:rtl_batch]


X, Y = make_bars(256)
X_RTL = X[:4]


class StageA(nn.Module):
    """1x1 conv only - no pool, no flatten, no softmax.

    Ending on a conv rather than a softmax is deliberate: the final bundle's
    check is then a genuine integer comparison. xmodel.py's softmax check
    compares probabilities in [0,1] with atol=0.5, which cannot fail.
    """

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(CHANNELS, 4, kernel_size=1, bias=True)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(4, 2, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        return x


class StageB(nn.Module):
    """3x3 'same' conv - real conv arithmetic, still stride 1."""

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(CHANNELS, 4, kernel_size=3, padding='same', bias=True)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(4, 2, kernel_size=3, padding='same', bias=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        return x


class StageC(nn.Module):
    """Adds a stride-2 conv.

    torch rejects padding='same' on a strided conv, so the padding is given
    explicitly as kernel_size//2 - which is what 'same' resolves to for an odd
    kernel, and what dataflow.py's CSH_SHIFT math assumes.
    """

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(CHANNELS, 4, kernel_size=3, stride=2, padding=1, bias=True)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(4, 2, kernel_size=3, padding='same', bias=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        return x


class StageD(nn.Module):
    """Adds max pooling.

    Max rather than average on purpose: averaging accumulates, so it has to
    agree with runtime.h's div_round and its widened accumulator
    (xlayers.py:342), whereas max just selects an existing value and has no
    rounding behaviour to match. Average pooling is tracked separately.
    """

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(CHANNELS, 4, kernel_size=3, padding='same', bias=True)
        self.relu_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2 = nn.Conv2d(4, 2, kernel_size=3, padding='same', bias=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        return x


class StageE(nn.Module):
    """The full classifier: conv -> pool -> conv -> flatten -> dense -> softmax.

    8x8 -> (pool) 4x4, then 8 channels of 4x4 flatten to 128 features.
    """

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(CHANNELS, 4, kernel_size=3, padding='same', bias=True)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(4, 8, kernel_size=3, padding='same', bias=True)
        self.relu_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_3 = nn.Conv2d(8, 8, kernel_size=1, bias=True)
        self.relu_3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.out = nn.Linear(8 * 4 * 4, 2, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.pool_2(x)
        x = self.conv_3(x)
        x = self.relu_3(x)
        x = self.flatten(x)
        x = self.out(x)
        x = self.softmax(x)
        return x


class StageF(nn.Module):
    """Adds a residual (skip) connection.

    conv_2's bundle adds conv_1's bundle output. Both use ReLU, which is required
    rather than incidental: the two bundles share one activation quant proxy so
    they land on the same fractional grid (the hardware adds the operands raw),
    and that proxy owns the nonlinearity as well as the scale.

    Ends on a conv with no softmax so the final RTL comparison is a genuine
    integer one.
    """

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(CHANNELS, 8, kernel_size=3, padding='same', bias=True)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(8, 8, kernel_size=3, padding='same', bias=True)
        self.relu_2 = nn.ReLU()
        self.conv_3 = nn.Conv2d(8, 2, kernel_size=1, bias=True)

    def forward(self, x):
        x = skip = self.relu_1(self.conv_1(x))
        x = self.relu_2(self.conv_2(x)) + skip
        return self.conv_3(x)


class StageG(nn.Module):
    """Average pooling instead of max.

    The engine sums the window and divides with runtime.h's div_round, which is
    not ordinary rounding - see QuantAvgPool2dDivRound in xlayer/quantPooling.py. Only 'valid'
    padding (torch padding=0) is implemented, which keeps the divisor constant.
    """

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(CHANNELS, 8, kernel_size=3, padding='same', bias=True)
        self.relu_1 = nn.ReLU()
        self.pool_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_2 = nn.Conv2d(8, 2, kernel_size=3, padding='same', bias=True)

    def forward(self, x):
        x = self.pool_1(self.relu_1(self.conv_1(x)))
        return self.conv_2(x)


class StageH(nn.Module):
    """conv + BatchNorm, the shape almost every torch CNN is written in.

    Both convs are bias=False, as they normally are when a BN follows - the fold
    is what creates the bias, so this exercises that path rather than assuming
    a bias was already there. Ends on a conv with no softmax so the final RTL
    comparison is a genuine integer one.
    """

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(CHANNELS, 8, kernel_size=3, padding='same', bias=False)
        self.bn_1 = nn.BatchNorm2d(8)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(8, 8, kernel_size=3, padding='same', bias=False)
        self.bn_2 = nn.BatchNorm2d(8)
        self.relu_2 = nn.ReLU()
        self.conv_3 = nn.Conv2d(8, 2, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.relu_1(self.bn_1(self.conv_1(x)))
        x = self.relu_2(self.bn_2(self.conv_2(x)))
        return self.conv_3(x)


class StageI(nn.Module):
    """Strided conv on an EVEN input, via explicit TF-'same' padding.

    Stage c gets a strided conv to work by choosing an odd input size, where
    torch's symmetric padding happens to coincide with the engine's. That is a
    workaround, not a solution - ResNet's strided convs all sit on even sizes.
    Here the padding is written out explicitly with tf_same_padding, which
    expresses the asymmetric split torch's Conv2d(padding=) cannot, and matches
    the engine at any size.
    """

    def __init__(self):
        super().__init__()
        lo, hi = tf_same_padding(IMG_SIZE, 3, 2)
        self.pad_1 = nn.ZeroPad2d((lo, hi, lo, hi))
        self.conv_1 = nn.Conv2d(CHANNELS, 8, kernel_size=3, stride=2, padding=0, bias=True)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(8, 2, kernel_size=3, padding='same', bias=True)

    def forward(self, x):
        return self.conv_2(self.relu_1(self.conv_1(self.pad_1(x))))


class StageJ(nn.Module):
    """Branching main path: stem's output feeds TWO downstream bundles as MAIN
    input (not a residual add) - conv_shortcut (1x1) and conv_main (3x3) - the
    ResNet downsample-block shape. Their outputs are then combined the same way
    StageF's residual add already is, giving the real block: 1x1 shortcut +
    3x3 main path, summed.

    stem's output channel count (16) is capped by check_hardware's ACC_WIDTH
    bound (K_BITS + X_BITS + clog2(KH*KW*CI) <= 24, the float32 mantissa
    _conv2d_same's golden per-pass sums are exact within) - a 3x3 kernel at
    K_BITS=X_BITS=8 caps CI at 28, well under the 64 the docs/superpowers/
    specs/2026-08-14-resnet-remaining-gaps-design.md measurement used. What
    actually matters for exercising the bug does not need that literal channel
    count, though: X_PAD is driven by kernel HEIGHT, not channel count, and is
    0 for the 1x1 shortcut but nonzero for the 3x3 main path regardless of CI -
    that alone gives the two consumers genuinely different engine tilings of
    the same producer tensor, which is exactly what runtime.h's tile_write
    could not route to more than one consumer before this fix.

    conv_shortcut needs no `branches` entry: it is literally the next bundle
    after stem in the flat child list, so it already gets stem's output as its
    default main input. conv_main DOES need one - without it, its default main
    input would be conv_shortcut's output (the wrong, but plausible-looking,
    literal predecessor) instead of stem's.
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(CHANNELS, 16, kernel_size=3, padding='same', bias=True)
        self.relu_stem = nn.ReLU()
        self.conv_shortcut = nn.Conv2d(16, 8, kernel_size=1, bias=True)
        self.relu_shortcut = nn.ReLU()
        self.conv_main = nn.Conv2d(16, 8, kernel_size=3, padding='same', bias=True)
        self.relu_main = nn.ReLU()
        self.conv_out = nn.Conv2d(8, 2, kernel_size=1, bias=True)

    def forward(self, x):
        x = stem = self.relu_stem(self.stem(x))
        shortcut = self.relu_shortcut(self.conv_shortcut(stem))
        x = self.relu_main(self.conv_main(stem)) + shortcut
        return self.conv_out(x)


class StageK(nn.Module):
    """Real post-add activation - the standard ResNet BasicBlock shape
    (activation AFTER the sum), unlike StageF which lets add_act default to
    Identity. conv_2's own pre-add activation (relu_2) still has to match
    skip's type for the raw-add scale-sharing to work (unchanged requirement,
    same as StageF) - lrelu_post is the NEW thing: a genuinely different
    activation TYPE applied to the sum, with its own independently
    calibrated scale, detected because it is the next real activation module
    in the flat child list past conv_2's own activation. Using a different
    type than relu_2 (not another ReLU) is deliberate - it is what would
    expose a bug where the post-add detection accidentally reused conv_2's
    activation instead of building a fresh one.

    Ends on a conv with no softmax so the final RTL comparison is a genuine
    integer one.
    """

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(CHANNELS, 8, kernel_size=3, padding='same', bias=True)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(8, 8, kernel_size=3, padding='same', bias=True)
        self.relu_2 = nn.ReLU()
        self.lrelu_post = nn.LeakyReLU(0.125)
        self.conv_4 = nn.Conv2d(8, 2, kernel_size=1, bias=True)

    def forward(self, x):
        x = skip = self.relu_1(self.conv_1(x))
        x = self.relu_2(self.conv_2(x)) + skip
        x = self.lrelu_post(x)
        return self.conv_4(x)


class StageL(nn.Module):
    """Same shape as StageA (1x1 conv only, no pool/flatten/softmax - the
    final bundle's check stays a genuine integer comparison) but built with
    16-bit activations rather than the default 8 - see conv_main.py's
    STAGE_ACT_BITS. The point isn't conv geometry (already proven by StageA);
    it's exercising the widened write_x/pack_words_into_bytes/dnn_engine.v
    tkeep paths that only fire once a word needs more than one byte.
    """

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(CHANNELS, 4, kernel_size=1, bias=True)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(4, 2, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        return x


class StageM(nn.Module):
    """Residual sourced from a POOLED bundle - stem ends in MaxPool2d, and
    conv_2's residual add uses stem's (pooled) output as its skip identity.
    Mirrors FashionMNISTModel's real shape (stem -> pool -> residual block,
    whose `identity` is the already-pooled tensor) - a combination no prior
    stage exercises (StageF/K's residual sources have no pool; StageD/G's
    pooling isn't used as a residual source). Settles empirically whether
    tile_write's add_buffers write (deepsocflow/c/runtime.h) captures the
    pre-pool or post-pool value for a bundle that pools: it's a shared
    subroutine called from two sites - once passing the raw pre-pool value
    (no pooling), once passing the already-reduced `result` (pooling present)
    - so which one actually executes depends on the CALL SITE, not the
    function's own definition order in the file.

    Ends on a conv with no softmax so the final RTL comparison is a genuine
    integer one.
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(CHANNELS, 8, kernel_size=3, padding='same', bias=True)
        self.relu_stem = nn.ReLU()
        self.pool_stem = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1 = nn.Conv2d(8, 8, kernel_size=3, padding='same', bias=True)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(8, 8, kernel_size=3, padding='same', bias=True)
        self.relu_2 = nn.ReLU()  # pre-add, must match relu_stem's type
        self.conv_out = nn.Conv2d(8, 2, kernel_size=1, bias=True)

    def forward(self, x):
        x = skip = self.pool_stem(self.relu_stem(self.stem(x)))
        x = self.relu_1(self.conv_1(x))
        x = self.relu_2(self.conv_2(x)) + skip
        return self.conv_out(x)


# {consumer_attr: source_attr} handed to quantized_model as `residuals`.
STAGE_RESIDUALS = {'f': {'conv_2': 'conv_1'},
                    'j': {'conv_main': 'conv_shortcut'},
                    'k': {'conv_2': 'conv_1'},
                    'm': {'conv_2': 'stem'}}

# {consumer_attr: source_attr} handed to quantized_model as `branches` - a
# consumer's MAIN input, not its residual add. See StageJ.
STAGE_BRANCHES = {'j': {'conv_main': 'stem'}}

# Per-stage activation bit width, handed to both Hardware(bits_input=...) and
# quantized_model(act_bits=...) - every other stage stays at the default (8).
STAGE_ACT_BITS = {'l': 16}

# Per-stage accumulator bit width override, handed to Hardware(bits_sum=...).
# rtl_export.py::export_bundle's own ACC_WIDTH check (distinct from
# check_hardware's, which uses the model's real in_features) is
# K_BITS + X_BITS + clog2(KH*KW*CM) where CM is padded up to the engine's
# RAM_WEIGHTS_DEPTH capacity (512 by default) rather than the real channel
# count - at X_BITS=16 that's 8+16+clog2(512)=33, already over the default
# bits_sum=32. Every X_BITS<=8 stage stays comfortably under 32
# (8+8+9=25), so only 'l' needs the override.
STAGE_BITS_SUM = {'l': 40}

STAGES = {'a': StageA, 'b': StageB, 'c': StageC, 'd': StageD, 'e': StageE,
          'f': StageF, 'g': StageG, 'h': StageH, 'i': StageI, 'j': StageJ,
          'k': StageK, 'l': StageL, 'm': StageM}


def prime_batchnorm(model, x, batches=8):
    """Runs data through the model in train mode so its BatchNorms accumulate
    real running statistics, then returns it in eval mode.

    Without this the fold is a no-op and proves nothing: a freshly constructed
    BatchNorm has running_mean=0, running_var=1, gamma=1, beta=0, so
    W*gamma/sqrt(var+eps) is W and the folded bias is zero. A stage that
    exercises folding has to have statistics worth folding.

    These stages are otherwise untrained on purpose (see the module docstring);
    priming touches only the BN buffers, not the weights.
    """
    model.train()
    n = max(1, len(x) // batches)
    with torch.no_grad():
        for i in range(batches):
            chunk = x[i * n:(i + 1) * n]
            if len(chunk):
                model(chunk)
    return model.eval()


def build_model(stage='e', seed=DEFAULT_SEED):
    """Constructs a stage's model with the seed applied first, so the initial
    weights are reproducible (see DEFAULT_SEED)."""
    torch.manual_seed(seed)
    return STAGES[stage]()


def train(model, x=X, y=Y, epochs=400, lr=0.01, seed=DEFAULT_SEED, patience=60):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()

    best_loss = float("inf")
    epochs_without_improvement = 0

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(torch.log(out.clamp_min(1e-9)), y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"epoch {epoch + 1:4d}  loss {loss.item():.4f}")

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"epoch {epoch + 1:4d}  loss {loss.item():.4f}  "
                      f"(early stop: no improvement for {patience} epochs)")
                break

    return model


def load(model, path=MODEL_PATH):
    model.load_state_dict(torch.load(path))
    return model


if __name__ == "__main__":
    model = build_model('e')
    train(model)

    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=-1)
    acc = (preds == Y).float().mean().item()
    print(f"train accuracy: {acc:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved model to {MODEL_PATH}")
