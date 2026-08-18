import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """activation is split into three independent roles, declared in the
    order forward() actually uses them - quantized_model (ptq.py) walks
    named_children() linearly and pairs each compute layer with the next
    activation module in that order, so a declaration order that doesn't
    match execution order (the original model's single `self.activation`,
    reused for both roles and declared before conv1) would make the auto-
    mapper try to pair conv1 with the module meant for the sum instead.
    Reusing one module instance for two roles is also unsafe on its own:
    named_children() dedupes by identity, so the second bundle would
    silently consume the wrong child.

    act_pre_add (immediately after conv2/bn2, right before the add) is not a
    free choice: the hardware adds its two operands raw, so both sides must
    already sit on the same fractional grid, which the pipeline can only
    guarantee by having the two ends share one activation quant proxy - and
    sharing a proxy shares its nonlinearity too. It must therefore be the
    same activation TYPE as whatever produced `identity` (this block's own
    skip source), not this block's own preferred flavor - hence it's a
    separate constructor argument, set by the caller to match.
    """

    def __init__(self, channels, activation="relu", pre_add_activation="relu"):
        super().__init__()

        def _act(kind):
            if kind == "leaky_relu":
                # 0.125 (not 0.1): quant_lrelu (runtime.h) implements the
                # negative-side scaling as a left-shift, so it can only
                # realize slopes that are exact negative powers of two.
                return nn.LeakyReLU(negative_slope=0.125)
            return nn.ReLU()

        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = _act(activation)

        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.act_pre_add = _act(pre_add_activation)

        self.act_post_add = _act(activation)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_pre_add(out)

        # Residual connection
        out = out + identity
        out = self.act_post_add(out)

        return out


class FashionMNISTModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # CNN stem - ends on ReLU, so residual1's pre_add_activation below is
        # fixed to "relu" to match it (see ResidualBlock's docstring).
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        # Residual block: 32 channels. activation="leaky_relu" is this
        # block's own flavor (act1/act_post_add); pre_add_activation="relu"
        # is mandatory - it must match the stem's own ReLU above.
        self.residual1 = ResidualBlock(
            channels=32,
            activation="leaky_relu",
            pre_add_activation="relu"
        )

        # Increase channels - ends on LeakyReLU, so residual2's
        # pre_add_activation below is fixed to "leaky_relu" to match it.
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.125),

            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            )
        )

        # Residual block: 64 channels. activation="relu" is this block's own
        # flavor; pre_add_activation="leaky_relu" is mandatory - it must
        # match the conv2 block's own LeakyReLU above.
        self.residual2 = ResidualBlock(
            channels=64,
            activation="relu",
            pre_add_activation="leaky_relu"
        )

        # AdaptiveAvgPool2d((1,1)) has no engine equivalent - dataflow.py
        # only ever reads a fixed 2-D pool_size/stride, and unlike AvgPool2d
        # it can't be sized ahead of export time from kernel_size alone.
        # AvgPool2d(kernel_size=7) is exactly equivalent here because a
        # 28x28 input has shrunk to exactly 7x7 by this point (28 -> stem's
        # stride-2 maxpool -> 14 -> conv2's stride-2 avgpool -> 7).
        self.global_avg_pool = nn.AvgPool2d(kernel_size=7)

        # Dense layers. Dropout removed: it's a training-only regularizer
        # (already a no-op under eval()) with no engine equivalent and no
        # case in quantized_model - keeping it here would just be dead
        # config once the model is quantized and deployed.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.125),
            nn.Linear(32, num_classes)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.residual1(x)

        x = self.conv2(x)
        x = self.residual2(x)

        x = self.global_avg_pool(x)
        logits = self.classifier(x)

        # logits out, for use with CrossEntropyLoss
        return logits

    def predict_proba(self, x):
        logits = self.forward(x)
        return self.softmax(logits)
