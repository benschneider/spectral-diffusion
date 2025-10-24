import torch

from src.training.sampling import build_sampler, sample_ddpm
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


def test_sampler_registry_returns_ddpm():
    coeffs = build_diffusion(T=10, kind="linear")
    model = DummyModel()
    sampler = build_sampler("ddpm", model=model, coeffs=coeffs)
    samples = sampler.sample(
        num_samples=2,
        shape=(1, 4, 4),
        num_steps=3,
        device=torch.device("cpu"),
    )
    assert samples.shape == (2, 1, 4, 4)
    assert samples.mean().abs() <= 1.0
