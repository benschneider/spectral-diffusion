import torch


def test_loss_decreases_over_steps(spectral_model, cifar_batch):
    """
    Run a handful of optimisation steps to ensure loss trends downward.
    Catches early training stalls that a single-step test would miss.
    """
    model = spectral_model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    prev_loss = None
    for step in range(10):
        x = cifar_batch.clone()
        loss = model(x).abs().mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        loss_value = loss.item()
        if prev_loss is not None:
            assert loss_value <= prev_loss * 1.05, (
                f"Loss failed to decrease at step {step}: {loss_value:.4f} vs {prev_loss:.4f}"
            )
        prev_loss = loss_value
