from __future__ import annotations

import torch

from riff_core import fft_filter_ifft


def test_fused_matches_torch():
    torch.manual_seed(4)
    data = torch.randn((128, 128), dtype=torch.complex64)
    filt = torch.randn((128, 128), dtype=torch.complex64)
    out, restored_filter = fft_filter_ifft(data, filt)
    ref = torch.fft.ifft2(torch.fft.fft2(data) * torch.fft.fft2(filt))
    ref = ref / (128 * 128)
    torch.testing.assert_close(out / (128 * 128), ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(restored_filter, filt, atol=1e-6, rtol=1e-6)
