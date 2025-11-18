import torch

from src.spectral import fast_fft


class _DummyBridge:
    """Mimic RIFFT semantics via torch.fft for unit testing."""

    def __init__(self) -> None:
        self.forward_calls = 0
        self.inverse_calls = 0

    def fft2(self, tensor, *, column_major: bool = False, copy_input: bool = True):
        self.forward_calls += 1
        return torch.fft.fft2(tensor, norm=None)

    def ifft2(self, tensor, *, copy_input: bool = True):
        self.inverse_calls += 1
        return torch.fft.ifft2(tensor, norm=None)


def test_fast_fft_matches_torch_when_rifft_forced(monkeypatch):
    tensor = torch.randn(2, 3, 32, 32, dtype=torch.float32)
    freq = torch.randn(2, 3, 32, 32, dtype=torch.complex64)

    bridge = _DummyBridge()
    monkeypatch.setattr(fast_fft, "_rifft_bridge", bridge)
    monkeypatch.setattr(fast_fft, "_RIFFT_AVAILABLE", True)
    monkeypatch.setattr(fast_fft, "_RIFFT_MIN_DIM", 1)

    accelerated = fast_fft.fftn(tensor, dim=(-2, -1), norm="ortho")
    reference = torch.fft.fftn(tensor, dim=(-2, -1), norm="ortho")
    assert torch.allclose(accelerated, reference, atol=1e-6, rtol=1e-6)
    accelerated_ifft = fast_fft.ifftn(freq, dim=(-2, -1), norm="ortho")
    reference_ifft = torch.fft.ifftn(freq, dim=(-2, -1), norm="ortho")
    assert torch.allclose(accelerated_ifft, reference_ifft, atol=1e-6, rtol=1e-6)
    assert bridge.forward_calls == 1
    assert bridge.inverse_calls == 1


def test_fast_fft_scales_forward_norm(monkeypatch):
    tensor = torch.randn(2, 3, 64, 64, dtype=torch.float32)
    bridge = _DummyBridge()
    monkeypatch.setattr(fast_fft, "_rifft_bridge", bridge)
    monkeypatch.setattr(fast_fft, "_RIFFT_AVAILABLE", True)
    monkeypatch.setattr(fast_fft, "_RIFFT_MIN_DIM", 1)

    accelerated = fast_fft.fftn(tensor, dim=(-2, -1), norm="forward")
    reference = torch.fft.fftn(tensor, dim=(-2, -1), norm="forward")
    assert torch.allclose(accelerated, reference, atol=1e-6, rtol=1e-6)
    assert bridge.forward_calls == 1


def test_fast_fft_prefers_torch_for_small_shapes(monkeypatch):
    tensor = torch.randn(1, 1, 8, 8)
    freq = torch.randn(1, 1, 8, 8, dtype=torch.complex64)

    bridge = _DummyBridge()
    monkeypatch.setattr(fast_fft, "_rifft_bridge", bridge)
    monkeypatch.setattr(fast_fft, "_RIFFT_AVAILABLE", True)
    monkeypatch.setattr(fast_fft, "_RIFFT_MIN_DIM", 128)

    # Fallback path should match direct torch results.
    out_fft = fast_fft.fftn(tensor, dim=(-2, -1), norm=None)
    ref_fft = torch.fft.fftn(tensor, dim=(-2, -1), norm=None)
    assert torch.allclose(out_fft, ref_fft, atol=1e-6, rtol=1e-6)

    out_ifft = fast_fft.ifftn(freq, dim=(-2, -1), norm=None)
    ref_ifft = torch.fft.ifftn(freq, dim=(-2, -1), norm=None)
    assert torch.allclose(out_ifft, ref_ifft, atol=1e-6, rtol=1e-6)
    assert bridge.forward_calls == 0
    assert bridge.inverse_calls == 0


def test_fast_fft_respects_requires_grad(monkeypatch):
    tensor = torch.randn(1, 1, 32, 32, dtype=torch.float32, requires_grad=True)

    def boom(*args, **kwargs):
        raise AssertionError("RIFFT path should be disabled for autograd tensors")

    monkeypatch.setattr(fast_fft, "_run_rifft_fft", boom)
    result = fast_fft.fftn(tensor, dim=(-2, -1), norm=None)
    reference = torch.fft.fftn(tensor, dim=(-2, -1), norm=None)
    assert torch.allclose(result, reference, atol=1e-6, rtol=1e-6)


def test_fast_fft_only_handles_last_two_dims(monkeypatch):
    tensor = torch.randn(2, 3, 16, 16, dtype=torch.float32)
    bridge = _DummyBridge()
    monkeypatch.setattr(fast_fft, "_rifft_bridge", bridge)
    monkeypatch.setattr(fast_fft, "_RIFFT_AVAILABLE", True)
    monkeypatch.setattr(fast_fft, "_RIFFT_MIN_DIM", 1)

    out_fft = fast_fft.fftn(tensor, dim=(0, 1), norm=None)
    ref_fft = torch.fft.fftn(tensor, dim=(0, 1), norm=None)
    assert torch.allclose(out_fft, ref_fft, atol=1e-6, rtol=1e-6)
    assert bridge.forward_calls == 0


def test_prefer_rifft_helper(monkeypatch):
    monkeypatch.setattr(fast_fft, "_RIFFT_MIN_DIM", 256)
    monkeypatch.setattr(fast_fft, "_RIFFT_AVAILABLE", True)
    assert not fast_fft.prefer_rifft(64, 64)
    assert fast_fft.prefer_rifft(512, 300)
    assert not fast_fft.prefer_rifft(512, 300, requires_grad=True)
