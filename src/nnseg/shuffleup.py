"""ShuffleUp3d: exact replacement for nnU-Net's ConvTranspose3d(kernel_size == stride, padding 0).

A transposed conv whose kernel equals its stride maps every input voxel to one s_z*s_y*s_x
block of output voxels with no overlap, so it is exactly a 1x1x1 Conv3d to
out_ch * prod(stride) channels followed by depth-to-space. Same MACs, bit-exact in fp32
(measured: max |diff| 0.0 on TS models), -11 % even in fp32 on MPS, and - the point -
runs in fp16 / bf16 on MPS, which refuses ConvTranspose3d in half. Headed for nnseg's
torch adapter as load-time network surgery.
"""
from __future__ import annotations

import torch


class ShuffleUp3d(torch.nn.Module):
    def __init__(self, t: torch.nn.ConvTranspose3d):
        super().__init__()
        k, s = tuple(t.kernel_size), tuple(t.stride)
        if not (k == s and tuple(t.padding) == (0, 0, 0) and tuple(t.output_padding) == (0, 0, 0)
                and t.groups == 1 and tuple(t.dilation) == (1, 1, 1)):
            raise ValueError(f"ShuffleUp3d needs kernel == stride, no padding, groups 1: got k={k} s={s} "
                             f"p={t.padding} op={t.output_padding} g={t.groups} d={t.dilation}")
        ic, oc = t.in_channels, t.out_channels
        self.s, self.oc = s, oc
        n_taps = s[0] * s[1] * s[2]
        self.conv = torch.nn.Conv3d(ic, oc * n_taps, kernel_size=1, bias=t.bias is not None,
                                    device=t.weight.device, dtype=t.weight.dtype)
        with torch.no_grad():
            w = t.weight.detach()                                               # (ic, oc, kz, ky, kx)
            self.conv.weight.copy_(w.permute(1, 2, 3, 4, 0).reshape(oc * n_taps, ic, 1, 1, 1))
            if t.bias is not None:
                self.conv.bias.copy_(t.bias.detach().repeat_interleave(n_taps))

    def forward(self, x):
        n, _, Z, Y, X = x.shape
        sz, sy, sx = self.s
        u = self.conv(x).view(n, self.oc, sz, sy, sx, Z, Y, X)
        u = u.permute(0, 1, 5, 2, 6, 3, 7, 4)                                   # (n, oc, Z, sz, Y, sy, X, sx)
        return u.reshape(n, self.oc, Z * sz, Y * sy, X * sx)


def swap_transposed(module: torch.nn.Module) -> int:
    """Replace every ConvTranspose3d under `module` in place. Returns the count."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.ConvTranspose3d):
            setattr(module, name, ShuffleUp3d(child)); n += 1
        else:
            n += swap_transposed(child)
    return n
