import pytest
import torch

from src.training.sampling import (
    SAMPLER_REGISTRY,
    DDPMSampler,
    build_sampler,
    register_sampler,
    sample_ddpm,
)
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


def test_sampler_registry_includes_ddim():
    coeffs = build_diffusion(T=10, kind="linear")
    model = DummyModel()
    sampler = build_sampler("ddim", model=model, coeffs=coeffs)
    samples = sampler.sample(
        num_samples=2,
        shape=(1, 4, 4),
        num_steps=3,
        device=torch.device("cpu"),
    )
    assert samples.shape == (2, 1, 4, 4)
    assert samples.abs().max() <= 1.0


def test_build_sampler_unknown_raises():
    coeffs = build_diffusion(T=4, kind="linear")
    model = DummyModel()
    with pytest.raises(ValueError):
        build_sampler("not-a-sampler", model=model, coeffs=coeffs)


def test_register_sampler_allows_custom_sampler():
    coeffs = build_diffusion(T=4, kind="linear")
    model = DummyModel()

    class ZeroSampler(DDPMSampler):
        @torch.no_grad()
        def sample(self, num_samples, shape, num_steps, device):
            return torch.zeros(num_samples, *shape, device=device)

    register_sampler("zero", ZeroSampler)
    try:
        sampler = build_sampler("zero", model=model, coeffs=coeffs)
        samples = sampler.sample(
            num_samples=1,
            shape=(1, 2, 2),
            num_steps=2,
            device=torch.device("cpu"),
        )
        assert torch.allclose(samples, torch.zeros_like(samples))
    finally:
        SAMPLER_REGISTRY.pop("zero", None)


def test_sampler_registry_dpm_solver_pp():
    coeffs = build_diffusion(T=10, kind="linear")
    model = DummyModel()
    sampler = build_sampler("dpm_solver++", model=model, coeffs=coeffs)
    samples = sampler.sample(
        num_samples=2,
        shape=(1, 4, 4),
        num_steps=3,
        device=torch.device("cpu"),
    )
    assert samples.shape == (2, 1, 4, 4)
    assert samples.abs().max() <= 1.0
