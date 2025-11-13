from __future__ import annotations

import torch

from riff_core import batched_fft2, batched_ifft2


def test_batched_fft_matches_torch():
    torch.manual_seed(2)
    data = torch.randn((4, 64, 64), dtype=torch.complex64)
    ref = torch.fft.fft2(data)
    out = batched_fft2(data)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_batched_ifft_roundtrip():
    torch.manual_seed(3)
    data = torch.randn((2, 32, 32), dtype=torch.complex64)
    freq = batched_fft2(data)
    recon = batched_ifft2(freq) / (32 * 32)
    torch.testing.assert_close(recon, data, atol=1e-4, rtol=1e-4)
