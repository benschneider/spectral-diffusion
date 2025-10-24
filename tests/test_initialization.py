import torch
from torch import nn

from src.core.initialization import apply_initialization


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)


def test_apply_initialization_zeros():
    model = DummyModel()
    for param in model.parameters():
        param.data.normal_()
    apply_initialization(model, {"strategy": "zeros"})
    for param in model.parameters():
        assert torch.count_nonzero(param) == 0


def test_apply_initialization_constant_vector():
    model = DummyModel()

    apply_initialization(
        model,
        {
            "strategy": "cross_domain_flat",
            "source": {
                "type": "constant",
                "values": [0.0, 1.0, -1.0],
                "length": model.linear.weight.numel(),
            },
            "scale": 0.25,
            "recycle": True,
        },
    )
    base = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)
    expected = base.repeat(8)[: model.linear.weight.numel()] * 0.25
    expected = expected.view_as(model.linear.weight)
    assert torch.allclose(model.linear.weight, expected, atol=1e-6)
