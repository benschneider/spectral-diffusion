import torch

from src.training.sampling import sample_ddpm
from src.training.scheduler import build_diffusion


class DummyModel(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def test_sample_ddpm_shapes():
    coeffs = build_diffusion(T=10, kind="linear")
    model = DummyModel()
    samples = sample_ddpm(
        model=model,
        coeffs=coeffs,
        num_samples=4,
        shape=(3, 8, 8),
        num_steps=5,
        device=torch.device("cpu"),
    )
    assert samples.shape == (4, 3, 8, 8)
    assert samples.min() >= -1.0
    assert samples.max() <= 1.0
