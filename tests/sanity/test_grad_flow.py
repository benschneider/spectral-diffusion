import pytest
import torch


def test_gradient_flow_not_vanishing(spectral_model, cifar_batch):
    """Single backward pass should produce finite, non-trivial gradients."""
    spectral_model.train()
    x = cifar_batch.clone().requires_grad_(True)
    output = spectral_model(x)
    loss = output.abs().mean()
    loss.backward()

    grads = [p.grad for p in spectral_model.parameters() if p.grad is not None]
    assert grads, "No gradients were produced during backward pass."

    mean_grad = torch.stack([g.abs().mean() for g in grads]).mean().item()
    assert mean_grad > 1e-8, f"Gradient vanished (mean={mean_grad:.2e})."
    assert mean_grad < 1e2, f"Gradient exploded (mean={mean_grad:.2e})."


def test_gradient_variance_stable(spectral_model, cifar_batch):
    """
    Gradients over successive steps should neither collapse nor blow up wildly.
    Detects issues that emerge only after a few updates.
    """
    spectral_model.train()
    opt = torch.optim.Adam(spectral_model.parameters(), lr=1e-3)
    variances = []
    for _ in range(5):
        x = cifar_batch.clone()
        loss = spectral_model(x).abs().mean()
        opt.zero_grad()
        loss.backward()
        grads = [p.grad.flatten() for p in spectral_model.parameters() if p.grad is not None]
        if not grads:
            pytest.fail("No gradients recorded during variance check.")
        stacked = torch.cat(grads)
        var = float(torch.var(stacked).cpu())
        variances.append(max(var, 1e-12))
        opt.step()
    ratio = max(variances) / max(min(variances), 1e-12)
    assert ratio < 100.0, f"Gradient variance instability detected (ratio={ratio:.1f})."
