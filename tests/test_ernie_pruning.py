from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from dsai6707.ernie_pruning import ErnieExpertMaskController, ernie_moe_layers

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _toy_model():
    import torch

    class DenseMLP(torch.nn.Module):
        def forward(self, values):
            return values

    class ToyMoeMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.k = 2
            self.gate = torch.nn.Linear(4, 4, bias=False, dtype=torch.float32)
            self.experts = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
            self.num_local_experts = 4

        def forward_experts(self, dispatched_input):
            dispatched_input = dispatched_input.reshape(
                1, self.num_local_experts, -1, dispatched_input.shape[-1]
            )
            chunks = dispatched_input.permute(1, 0, 2, 3).contiguous().unbind(0)
            return torch.stack(
                [expert(chunk) for expert, chunk in zip(self.experts, chunks, strict=True)],
                dim=1,
            )

        def route(self, values):
            logits = self.gate(values.float())
            probabilities = torch.sigmoid(logits)
            topk_probabilities, topk_indices = torch.topk(probabilities, self.k, dim=-1)
            weights = topk_probabilities / topk_probabilities.sum(dim=-1, keepdim=True)
            return logits, probabilities, topk_indices, weights

    class ToyLayer(torch.nn.Module):
        def __init__(self, mlp):
            super().__init__()
            self.mlp = mlp

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [ToyLayer(DenseMLP()), ToyLayer(ToyMoeMLP()), ToyLayer(ToyMoeMLP())]
            )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()

    model = Model().eval()
    with torch.no_grad():
        for mlp in ernie_moe_layers(model):
            mlp.gate.weight.copy_(torch.eye(4))
    return model


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch integration test")
def test_runtime_mask_matches_sliced_sigmoid_topk_and_close_restores_gate() -> None:
    import torch

    model = _toy_model()
    moe_layers = ernie_moe_layers(model)
    assert len(moe_layers) == 2
    identity = np.ones((2, 4), dtype=np.bool_)
    values = torch.tensor([[10.0, 9.0, 8.0, 7.0]])

    with torch.inference_mode():
        expected_identity = moe_layers[0].route(values)
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    parameter_values = tuple(parameter.detach().clone() for parameter in model.parameters())

    controller = ErnieExpertMaskController(model, identity)
    with torch.inference_mode():
        actual_identity = moe_layers[0].route(values)
    for actual, expected in zip(actual_identity, expected_identity, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    candidate = np.asarray(
        [[False, False, True, True], [True, False, True, False]], dtype=np.bool_
    )
    controller.apply(candidate)
    logits, probabilities, indices, weights = moe_layers[0].route(values)

    torch.testing.assert_close(logits, torch.tensor([[8.0, 7.0]]))
    assert set(indices[0].tolist()) == {0, 1}
    expected_probabilities = torch.sigmoid(torch.tensor([8.0, 7.0]))
    torch.testing.assert_close(probabilities[0], expected_probabilities)
    torch.testing.assert_close(weights[0], expected_probabilities / expected_probabilities.sum())
    np.testing.assert_array_equal(controller.mask, candidate)

    controller.close()
    restored_logits = moe_layers[0].gate(values)
    torch.testing.assert_close(restored_logits, values)
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    for actual, expected in zip(model.parameters(), parameter_values, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="closed"):
        controller.apply(identity)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch integration test")
def test_mask_validation_and_duplicate_controller_protection() -> None:
    import torch

    model = _toy_model()
    identity = np.ones((2, 4), dtype=np.bool_)

    with pytest.raises(ValueError, match="NumPy boolean"):
        ErnieExpertMaskController(model, identity.astype(np.int8))
    with pytest.raises(ValueError, match="shape"):
        ErnieExpertMaskController(model, np.ones((1, 4), dtype=np.bool_))

    controller = ErnieExpertMaskController(model, identity)
    uneven = identity.copy()
    uneven[0, 0] = False
    controller.apply(uneven)
    np.testing.assert_array_equal(controller.mask, uneven)
    first, second = ernie_moe_layers(model)
    values = torch.tensor([[10.0, 9.0, 8.0, 7.0]])
    with torch.inference_mode():
        assert first.gate(values).shape == (1, 3)
        assert second.gate(values).shape == (1, 4)
        compact_output = first.forward_experts(torch.arange(24).reshape(1, 6, 4))
        passthrough_output = second.forward_experts(torch.arange(32).reshape(1, 8, 4))
    assert compact_output.shape == (1, 3, 2, 4)
    assert passthrough_output.shape == (1, 4, 2, 4)

    second_uneven = np.asarray(
        [[True, False, True, False], [True, True, True, False]], dtype=np.bool_
    )
    controller.apply(second_uneven)
    with torch.inference_mode():
        assert first.gate(values).shape == (1, 2)
        assert second.gate(values).shape == (1, 3)

    too_small = np.zeros((2, 4), dtype=np.bool_)
    too_small[:, 0] = True
    with pytest.raises(ValueError, match="top-k is 2"):
        controller.apply(too_small)
    with pytest.raises(RuntimeError, match="already has"):
        ErnieExpertMaskController(model, identity)
    controller.close()

