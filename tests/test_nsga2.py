import unittest

import numpy as np

from dsai6707.pruning_nsga2 import (
    balanced_set_crossover,
    crowded_tournament,
    crowding_distances,
    mask_sha256,
    mutate_distinct_layer_swaps,
    nondominated_sort,
    rank_and_crowding,
    select_survivors,
)


class PruningNSGA2Tests(unittest.TestCase):
    def setUp(self):
        self.mask = np.ones((24, 32), dtype=np.bool_)
        self.mask[:, 28:] = False

    def test_mask_hash_is_shape_aware_and_independent_of_memory_layout(self):
        fortran_order = np.asfortranarray(self.mask)
        self.assertEqual(mask_sha256(self.mask), mask_sha256(fortran_order))
        self.assertEqual(mask_sha256(self.mask), mask_sha256(self.mask.copy()))

        changed = self.mask.copy()
        changed[0, 0] = False
        changed[0, 28] = True
        self.assertNotEqual(mask_sha256(self.mask), mask_sha256(changed))

        reshaped = self.mask.reshape(12, 64)
        self.assertNotEqual(mask_sha256(self.mask), mask_sha256(reshaped))

    def test_balanced_crossover_splits_each_parental_difference(self):
        first = self.mask.copy()
        second = np.ones_like(first)
        for layer in range(second.shape[0]):
            second[layer, [2, 3, 28 + layer % 2, 30 + layer % 2]] = False

        first_child, second_child = balanced_set_crossover(
            first, second, rng=np.random.default_rng(6707)
        )
        repeated = balanced_set_crossover(
            first, second, rng=np.random.default_rng(6707)
        )
        self.assertTrue(np.array_equal(first_child, repeated[0]))
        self.assertTrue(np.array_equal(second_child, repeated[1]))
        self.assertTrue(np.all(np.count_nonzero(first_child, axis=1) == 28))
        self.assertTrue(np.all(np.count_nonzero(second_child, axis=1) == 28))

        for layer in range(first.shape[0]):
            first_pruned = ~first[layer]
            second_pruned = ~second[layer]
            common = first_pruned & second_pruned
            difference = first_pruned ^ second_pruned
            outside_union = ~(first_pruned | second_pruned)
            self.assertTrue(np.all(~first_child[layer, common]))
            self.assertTrue(np.all(~second_child[layer, common]))
            self.assertTrue(
                np.all(
                    (~first_child[layer, difference])
                    ^ (~second_child[layer, difference])
                )
            )
            self.assertTrue(np.all(first_child[layer, outside_union]))
            self.assertTrue(np.all(second_child[layer, outside_union]))

    def test_mutation_changes_one_pair_in_one_to_three_distinct_layers(self):
        mutation = mutate_distinct_layer_swaps(
            self.mask, rng=np.random.default_rng(17)
        )
        self.assertGreaterEqual(len(mutation.swaps), 1)
        self.assertLessEqual(len(mutation.swaps), 3)
        self.assertEqual(len(mutation.layers), len(set(mutation.layers)))
        self.assertEqual(mutation.layers, tuple(sorted(mutation.layers)))
        self.assertTrue(np.all(np.count_nonzero(mutation.mask, axis=1) == 28))

        changed = mutation.mask != self.mask
        self.assertEqual(int(np.count_nonzero(changed)), 2 * len(mutation.swaps))
        self.assertEqual(set(np.flatnonzero(np.any(changed, axis=1))), set(mutation.layers))
        for swap in mutation.swaps:
            self.assertTrue(self.mask[swap.layer, swap.removed_expert])
            self.assertFalse(mutation.mask[swap.layer, swap.removed_expert])
            self.assertFalse(self.mask[swap.layer, swap.added_expert])
            self.assertTrue(mutation.mask[swap.layer, swap.added_expert])

    def test_nondominated_sort_minimizes_and_orders_fronts_by_candidate_id(self):
        objectives = np.asarray(
            [
                [1.0, 4.0],
                [2.0, 2.0],
                [4.0, 1.0],
                [3.0, 4.0],
                [4.0, 3.0],
                [5.0, 5.0],
            ]
        )
        candidate_ids = np.asarray([40, 10, 30, 20, 50, 60])
        self.assertEqual(
            nondominated_sort(objectives, candidate_ids=candidate_ids),
            ((1, 2, 0), (3, 4), (5,)),
        )

    def test_crowding_distance_is_normalized_and_skips_constant_axes(self):
        objectives = np.asarray(
            [
                [1.0, 7.0, 4.0],
                [2.0, 7.0, 2.0],
                [4.0, 7.0, 1.0],
            ]
        )
        distances = crowding_distances(
            objectives, (0, 1, 2), candidate_ids=[30, 10, 20]
        )
        self.assertTrue(np.isinf(distances[0]))
        self.assertTrue(np.isinf(distances[2]))
        self.assertAlmostEqual(distances[1], 2.0)

    def test_crowded_tournament_prioritizes_rank_distance_then_id(self):
        rng = np.random.default_rng(9)
        winner = crowded_tournament(
            ranks=[0, 0],
            distances=[1.0, 1.0],
            candidate_ids=[20, 10],
            rng=rng,
            tournament_size=2,
            pool=[0, 1],
        )
        self.assertEqual(winner, 1)

        rank_winner = crowded_tournament(
            ranks=[1, 0],
            distances=[float("inf"), 0.0],
            candidate_ids=[1, 2],
            rng=np.random.default_rng(9),
            tournament_size=2,
        )
        self.assertEqual(rank_winner, 1)

        distance_winner = crowded_tournament(
            ranks=[0, 0],
            distances=[0.5, 1.5],
            candidate_ids=[1, 2],
            rng=np.random.default_rng(9),
            tournament_size=2,
        )
        self.assertEqual(distance_winner, 1)

    def test_survivor_selection_is_deterministic_on_partial_front(self):
        objectives = np.asarray(
            [[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]]
        )
        candidate_ids = np.asarray([50, 10, 20, 30, 40])
        first = select_survivors(objectives, candidate_ids, population_size=3)
        second = select_survivors(objectives, candidate_ids, population_size=3)
        self.assertEqual(first.tolist(), [4, 0, 1])
        self.assertTrue(np.array_equal(first, second))

        ranks, distances = rank_and_crowding(
            objectives, candidate_ids=candidate_ids
        )
        self.assertTrue(np.all(ranks == 0))
        self.assertTrue(np.isinf(distances[0]))
        self.assertTrue(np.isinf(distances[4]))

    def test_invalid_cardinality_and_duplicate_ids_are_rejected(self):
        invalid = self.mask.copy()
        invalid[0, 0] = False
        with self.assertRaisesRegex(ValueError, "every layer"):
            balanced_set_crossover(
                self.mask, invalid, rng=np.random.default_rng(1)
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            nondominated_sort(np.ones((2, 2)), candidate_ids=[3, 3])


if __name__ == "__main__":
    unittest.main()
