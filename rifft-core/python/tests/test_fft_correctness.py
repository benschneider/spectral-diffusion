from __future__ import annotations

import torch

from riff_core import fft2, ifft2


def test_fft2_matches_torch():
    torch.manual_seed(0)
    data = torch.randn((256, 256), dtype=torch.complex64)
    ref = torch.fft.fft2(data)
    out = fft2(data)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_ifft2_roundtrip():
    torch.manual_seed(1)
    data = torch.randn((128, 128), dtype=torch.complex64)
    freq = fft2(data)
    recon = ifft2(freq) / (128 * 128)
    torch.testing.assert_close(recon, data, atol=1e-4, rtol=1e-4)
