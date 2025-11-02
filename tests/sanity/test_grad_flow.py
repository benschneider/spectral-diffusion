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
