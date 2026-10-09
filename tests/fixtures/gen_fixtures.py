"""Generate the PyTorch parity fixtures used by `tests/pytorch_parity.rs`.

Each fixture is a small, randomly initialized network saved the way the official
Real-ESRGAN checkpoints are (`{"params": state_dict}` or `{"params_ema": ...}`), plus a JSON
file with an input and the output PyTorch computes for it.

The networks below are re-implementations of the BasicSR / Real-ESRGAN architectures
(`RRDBNet`, `SRVGGNetCompact`, `UNetDiscriminatorSN`) with identical parameter names.

Usage (needs only `torch`):

    python tests/fixtures/gen_fixtures.py
"""

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

OUT = Path(__file__).resolve().parent


def pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    h, w = hh // scale, hw // scale
    x = x.view(b, c, h, scale, w, scale)
    return x.permute(0, 1, 3, 5, 2, 4).reshape(b, c * scale * scale, h, w)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat, num_grow_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, scale, num_feat, num_block, num_grow_ch, num_in_ch=3, num_out_ch=3):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch *= 4
        elif scale == 1:
            num_in_ch *= 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_feat, num_conv, upscale, num_in_ch=3, num_out_ch=3):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


class UNetDiscriminatorSN(nn.Module):
    def __init__(self, num_in_ch, num_feat, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        lrelu = lambda t: F.leaky_relu(t, negative_slope=0.2)  # noqa: E731
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)  # noqa: E731
        x0 = lrelu(self.conv0(x))
        x1 = lrelu(self.conv1(x0))
        x2 = lrelu(self.conv2(x1))
        x3 = lrelu(self.conv3(x2))
        x4 = lrelu(self.conv4(up(x3)))
        if self.skip_connection:
            x4 = x4 + x2
        x5 = lrelu(self.conv5(up(x4)))
        if self.skip_connection:
            x5 = x5 + x1
        x6 = lrelu(self.conv6(up(x5)))
        if self.skip_connection:
            x6 = x6 + x0
        out = lrelu(self.conv7(x6))
        out = lrelu(self.conv8(out))
        return self.conv9(out)


def randomize_(model):
    """Replace the default initialization so every parameter (PReLU slopes included) matters."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            fan = param[0].numel() if param.dim() > 1 else param.numel()
            param.copy_(torch.randn_like(param) / fan**0.5)


def save(name, model, key, input_shape, legacy=False):
    model.eval()
    x = torch.rand(*input_shape)
    with torch.no_grad():
        y = model(x)
    torch.save(
        {key: model.state_dict()},
        OUT / f"{name}.pth",
        _use_new_zipfile_serialization=not legacy,
    )
    reference = {
        "input_shape": list(x.shape),
        "input": x.flatten().tolist(),
        "output_shape": list(y.shape),
        "output": y.flatten().tolist(),
    }
    (OUT / f"{name}.json").write_text(json.dumps(reference))
    print(f"{name}: {list(x.shape)} -> {list(y.shape)}")


def main():
    torch.manual_seed(0)

    rrdb = RRDBNet(scale=2, num_feat=8, num_block=2, num_grow_ch=4)
    randomize_(rrdb)
    save("rrdb_x2", rrdb, "params_ema", (1, 3, 6, 8))

    # The official ESRGAN checkpoint uses the pre-1.6 (non-zip) serialization.
    rrdb = RRDBNet(scale=4, num_feat=8, num_block=1, num_grow_ch=4)
    randomize_(rrdb)
    save("rrdb_x4_legacy", rrdb, "params", (1, 3, 5, 7), legacy=True)

    srvgg = SRVGGNetCompact(num_feat=8, num_conv=3, upscale=4)
    randomize_(srvgg)
    save("srvgg_x4", srvgg, "params", (1, 3, 6, 5))

    disc = UNetDiscriminatorSN(num_in_ch=3, num_feat=4)
    randomize_(disc)
    # A few training-mode passes move the power-iteration vectors `weight_u` / `weight_v`
    # away from their initial values, as in a trained checkpoint.
    disc.train()
    with torch.no_grad():
        for _ in range(3):
            disc(torch.rand(1, 3, 16, 16))
    save("unet_disc", disc, "params", (1, 3, 16, 24))


if __name__ == "__main__":
    main()
