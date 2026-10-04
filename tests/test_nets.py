import pytest
import torch

from pmaint.models.nets import build_model


@pytest.mark.parametrize("name", ["lstm", "cnn"])
def test_forward_shape_and_backward(name):
    model = build_model(name, n_features=15)
    x = torch.randn(8, 30, 15)
    y = model(x)
    assert y.shape == (8,)
    y.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
