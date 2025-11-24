import torch

from src.spectral.adapter import SpectralAdapter


def test_learnable_adapter_backward():
    adapter = SpectralAdapter(
        enabled=True,
        weighting="bandpass",
        normalize=True,
        bandpass_inner=0.1,
        bandpass_outer=0.6,
        learnable=True,
        condition_dim=4,
        mlp_hidden_dim=8,
    )
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    cond = torch.randn(2, 4)

    out = adapter(x, cond)
    loss = out.mean()
    loss.backward()

    assert adapter.base_params.grad is not None
    if adapter.mlp is not None:
        grads = [p.grad for p in adapter.mlp.parameters()]
        assert any(g is not None for g in grads)


def test_non_learnable_adapter_matches_identity_when_disabled():
    adapter = SpectralAdapter(enabled=False)
    x = torch.randn(2, 3, 8, 8)
    y = adapter(x)
    assert torch.allclose(x, y)


def test_learnable_adapter_handles_missing_condition():
    adapter = SpectralAdapter(
        enabled=True,
        weighting="bandpass",
        learnable=True,
        mlp_hidden_dim=8,
        condition_dim=0,
    )
    x = torch.randn(1, 3, 8, 8, requires_grad=True)
    out = adapter(x)
    out.sum().backward()
    assert adapter.base_params.grad is not None
