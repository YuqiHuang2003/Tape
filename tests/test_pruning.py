import importlib.util
import unittest

from dsai6707.pruning import ExpertSelectionPlan

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class ExpertSelectionPlanTests(unittest.TestCase):
    def test_plan_is_canonical_and_round_trips(self):
        plan = ExpertSelectionPlan.from_retained(
            [[3, 1], [2, 0]],
            num_experts=4,
        )
        self.assertEqual(plan.layers, ((1, 3), (0, 2)))
        self.assertEqual(plan.retained_counts, (2, 2))
        self.assertFalse(plan.is_identity)
        self.assertEqual(ExpertSelectionPlan.from_dict(plan.to_dict()), plan)

    def test_all_experts_builds_identity_plan(self):
        plan = ExpertSelectionPlan.all_experts(num_layers=2, num_experts=4)
        self.assertEqual(plan.layers, ((0, 1, 2, 3), (0, 1, 2, 3)))
        self.assertTrue(plan.is_identity)

    def test_plan_rejects_invalid_sets(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ExpertSelectionPlan.from_retained([[0, 0]], num_experts=4)
        with self.assertRaisesRegex(ValueError, r"\[0, 4\)"):
            ExpertSelectionPlan.from_retained([[0, 4]], num_experts=4)
        with self.assertRaisesRegex(ValueError, "at least one"):
            ExpertSelectionPlan.from_retained([[]], num_experts=4)


def _toy_model(*, num_layers=1):
    import torch

    class ToyRouter(torch.nn.Module):
        """The Transformers 4.46.3 GraniteMoeTopKGating algorithm."""

        def __init__(self):
            super().__init__()
            self.num_experts = 4
            self.input_size = 4
            self.top_k = 2
            self.layer = torch.nn.Linear(4, 4, bias=False)

        def forward(self, hidden_states):
            logits = self.layer(hidden_states).float()
            top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
            top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)
            zeros = torch.zeros(
                [top_k_gates.size(0), self.num_experts],
                dtype=top_k_gates.dtype,
                device=top_k_gates.device,
            )
            gates = zeros.scatter(1, top_k_indices, 1)
            expert_size = gates.long().sum(0).tolist()
            top_k_experts = top_k_indices.flatten()
            _, index_sorted_experts = top_k_experts.sort(0)
            batch_index = index_sorted_experts.div(self.top_k, rounding_mode="trunc")
            flat_gates = top_k_gates.flatten()
            batch_gates = flat_gates[index_sorted_experts]
            return index_sorted_experts, batch_index, batch_gates, expert_size, logits

    class ToyParallelExperts(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, output_size, input_size))

        def forward(self, inputs, expert_size):
            chunks = inputs.split(expert_size, dim=0)
            outputs = [
                torch.nn.functional.linear(chunks[index], self.weight[index])
                for index in range(4)
            ]
            return torch.cat(outputs, dim=0)

    class ToyMoe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_size = 4
            self.hidden_size = 3
            self.activation = torch.nn.functional.silu
            self.router = ToyRouter()
            self.input_linear = ToyParallelExperts(4, 6)
            self.output_linear = ToyParallelExperts(3, 4)

        def forward(self, layer_input):
            batch, length, width = layer_input.shape
            hidden = layer_input.reshape(-1, width)
            _, batch_index, batch_gates, expert_size, logits = self.router(hidden)
            expert_inputs = hidden[batch_index]
            expert_hidden = self.input_linear(expert_inputs, expert_size)
            gate, up = expert_hidden.chunk(2, dim=-1)
            expert_hidden = self.activation(gate) * up
            expert_outputs = self.output_linear(expert_hidden, expert_size)
            expert_outputs = expert_outputs * batch_gates[:, None]
            output = torch.zeros_like(hidden).index_add(0, batch_index, expert_outputs)
            return output.view(batch, length, width), logits

    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block_sparse_moe = ToyMoe()

    class ToyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([ToyLayer() for _ in range(num_layers)])

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = ToyBackbone()

        def forward(self, values):
            return self.model.layers[0].block_sparse_moe(values)

    torch.manual_seed(7)
    model = ToyModel().eval()
    with torch.no_grad():
        for layer in model.model.layers:
            layer.block_sparse_moe.router.layer.weight.copy_(torch.eye(4))
    return model


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch integration test")
class GranitePruningControllerTests(unittest.TestCase):
    def test_mask_is_applied_before_original_topk_and_softmax(self):
        import torch

        from dsai6707.pruning import GranitePruningController

        model = _toy_model()
        router = model.model.layers[0].block_sparse_moe.router
        plan = ExpertSelectionPlan.from_retained([[2, 3]], num_experts=4)
        hidden = torch.tensor([[10.0, 9.0, 8.0, 7.0]])

        with GranitePruningController(model, plan):
            _, _, batch_gates, expert_size, logits = router(hidden)

        self.assertEqual(expert_size, [0, 0, 1, 1])
        self.assertTrue(torch.isneginf(logits[0, :2]).all())
        torch.testing.assert_close(logits[0, 2:], torch.tensor([8.0, 7.0]))
        torch.testing.assert_close(
            batch_gates,
            torch.softmax(torch.tensor([8.0, 7.0]), dim=0),
        )

    def test_all_retained_path_is_bit_exact_and_does_not_touch_parameters(self):
        import torch

        from dsai6707.pruning import GranitePruningController

        model = _toy_model()
        inputs = torch.randn(2, 3, 4)
        with torch.inference_mode():
            expected_output, expected_logits = model(inputs)
        parameter_ids = tuple(id(parameter) for parameter in model.parameters())
        parameter_values = tuple(parameter.detach().clone() for parameter in model.parameters())
        state_keys = tuple(model.state_dict())
        plan = ExpertSelectionPlan.all_experts(num_layers=1, num_experts=4)

        with GranitePruningController(model, plan):
            with torch.inference_mode():
                actual_output, actual_logits = model(inputs)

        torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
        self.assertEqual(tuple(id(parameter) for parameter in model.parameters()), parameter_ids)
        self.assertEqual(tuple(model.state_dict()), state_keys)
        for actual, expected in zip(model.parameters(), parameter_values, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_apply_changes_candidate_and_close_restores_unpruned_router(self):
        import torch

        from dsai6707.pruning import GranitePruningController

        model = _toy_model()
        router = model.model.layers[0].block_sparse_moe.router
        hidden = torch.tensor([[10.0, 9.0, 8.0, 7.0]])
        identity = ExpertSelectionPlan.all_experts(num_layers=1, num_experts=4)
        controller = GranitePruningController(model, identity)
        controller.apply(ExpertSelectionPlan.from_retained([[2, 3]], num_experts=4))
        pruned_logits = router(hidden)[-1]
        controller.close()
        restored_logits = router(hidden)[-1]

        self.assertTrue(torch.isneginf(pruned_logits[0, :2]).all())
        torch.testing.assert_close(restored_logits, hidden)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            controller.apply(identity)

    def test_controller_rejects_fewer_retained_experts_than_topk(self):
        from dsai6707.pruning import GranitePruningController

        model = _toy_model()
        plan = ExpertSelectionPlan.from_retained([[0]], num_experts=4)
        with self.assertRaisesRegex(ValueError, "top_k is 2"):
            GranitePruningController(model, plan)

    def test_heterogeneous_layer_capacities_mask_without_mutating_parameters(self):
        import torch

        from dsai6707.pruning import GranitePruningController

        model = _toy_model(num_layers=2)
        routers = tuple(
            layer.block_sparse_moe.router for layer in model.model.layers
        )
        hidden = torch.tensor([[10.0, 9.0, 8.0, 7.0]])
        plan = ExpertSelectionPlan.from_retained(
            [[0, 2], [1, 2, 3]],
            num_experts=4,
        )
        parameter_ids = tuple(id(parameter) for parameter in model.parameters())
        parameter_values = tuple(parameter.detach().clone() for parameter in model.parameters())
        state_keys = tuple(model.state_dict())
        with torch.inference_mode():
            original_logits = tuple(router(hidden)[-1].clone() for router in routers)

        controller = GranitePruningController(model, plan)
        self.addCleanup(controller.close)
        with torch.inference_mode():
            first_result = routers[0](hidden)
            second_result = routers[1](hidden)

        self.assertEqual(plan.retained_counts, (2, 3))
        self.assertEqual(first_result[3], [1, 0, 1, 0])
        self.assertEqual(second_result[3], [0, 1, 1, 0])
        torch.testing.assert_close(
            first_result[-1],
            torch.tensor([[10.0, float("-inf"), 8.0, float("-inf")]]),
        )
        torch.testing.assert_close(
            second_result[-1],
            torch.tensor([[float("-inf"), 9.0, 8.0, 7.0]]),
        )

        self.assertEqual(tuple(id(parameter) for parameter in model.parameters()), parameter_ids)
        self.assertEqual(tuple(model.state_dict()), state_keys)
        for actual, expected in zip(model.parameters(), parameter_values, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        controller.close()
        with torch.inference_mode():
            restored_logits = tuple(router(hidden)[-1] for router in routers)
        for actual, expected in zip(restored_logits, original_logits, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
