"""电导加权随机游走期望首次到达时间测试。"""

from __future__ import annotations

import math
import unittest
from fractions import Fraction

from resistor_elim import (
    Network,
    PortError,
    StayWeightError,
    expected_hitting_times,
    schur_eliminate,
)
from reference import hitting_time_lower_bounds


class TestHittingTimesHandCalculated(unittest.TestCase):
    def test_reflecting_path_hand_computed(self):
        net = Network([0, 1, 2, 3])
        for i in range(3):
            net.add_conductor(i, i + 1, 1)
        result = expected_hitting_times(net, [0])
        self.assertEqual(result.expectations, {0: 0, 1: 5, 2: 8, 3: 9})
        self.assertEqual(result.max_abs_residual, 0)
        self.assertEqual(result.unreachable_components, [])
        for row in result.transition_probabilities.values():
            self.assertEqual(sum(row.values(), Fraction(0)), 1)

    def test_weighted_star_with_self_loop(self):
        net = Network(["t", "c", "l1", "l2"])
        net.add_conductor("t", "c", 2)
        net.add_conductor("c", "l1", 1)
        net.add_conductor("c", "l2", 3)
        net.set_stay_weight("c", 4)
        result = expected_hitting_times(net, ["t"])
        self.assertEqual(result.expectations["c"], 7)
        self.assertEqual(result.expectations["l1"], 8)
        self.assertEqual(result.expectations["l2"], 8)
        self.assertEqual(
            result.transition_probabilities["c"],
            {
                "t": Fraction(1, 5),
                "l1": Fraction(1, 10),
                "l2": Fraction(3, 10),
                "c": Fraction(2, 5),
            },
        )
        self.assertEqual(result.residuals["t"], 0)
        self.assertEqual(result.max_abs_residual, 0)

    def test_parallel_edges_add_before_normalization(self):
        net = Network(["a", "t", "b"])
        net.add_conductor("a", "t", Fraction(1, 3))
        net.add_conductor("a", "t", Fraction(2, 3))
        net.add_conductor("a", "b", 1)
        result = expected_hitting_times(net, ["t"])
        self.assertEqual(result.transition_probabilities["a"]["t"], Fraction(1, 2))
        self.assertEqual(result.expectations["a"], 3)
        self.assertEqual(result.expectations["b"], 4)

    def test_multiple_absorbing_targets(self):
        net = Network(["a", "b", "t1", "t2"])
        net.add_conductor("a", "b", 1)
        net.add_conductor("a", "t1", 1)
        net.add_conductor("b", "t2", 1)
        result = expected_hitting_times(net, ["t1", "t2"])
        self.assertEqual(result.expectations["a"], 2)
        self.assertEqual(result.expectations["b"], 2)


class TestFiniteStepTailBounds(unittest.TestCase):
    def test_path_tail_bounds_increase_toward_exact_expectation(self):
        nodes = [0, 1, 2, 3]
        branches = [(i, i + 1, 1) for i in range(3)]
        net = Network(nodes)
        for u, v, g in branches:
            net.add_conductor(u, v, g)
        result = expected_hitting_times(net, [0])
        bounds = hitting_time_lower_bounds(
            nodes,
            branches,
            target=0,
            horizons=(0, 1, 2, 8, 180),
            start=3,
        )
        values = [bounds[h] for h in (0, 1, 2, 8, 180)]
        self.assertEqual(values[0], 1)
        self.assertEqual(values[1], 2)
        self.assertEqual(values[2], 3)
        self.assertEqual(values, sorted(values))
        self.assertTrue(all(value < 9 for value in values))
        self.assertLess(bounds[180], result.expectations[3])
        self.assertLess(result.expectations[3] - bounds[180], Fraction(1, 1000000000))

    def test_star_tail_bounds_with_self_loop(self):
        nodes = ["t", "c", "l"]
        branches = [("t", "c", 2), ("c", "l", 1)]
        stays = {"c": 1}
        net = Network(nodes)
        for u, v, g in branches:
            net.add_conductor(u, v, g)
        result = expected_hitting_times(net, ["t"], stay_weights=stays)
        bounds = hitting_time_lower_bounds(
            nodes,
            branches,
            target="t",
            horizons=(0, 4, 24),
            start="c",
            stays=stays,
        )
        exact = result.expectations["c"]
        self.assertEqual(exact, Fraction(5, 2))
        self.assertEqual(bounds[0], 1)
        self.assertLess(bounds[4], bounds[24])
        self.assertLess(bounds[24], exact)
        self.assertLess(exact - bounds[24], Fraction(1, 1000))


class TestConnectivityAndValidation(unittest.TestCase):
    def test_disconnected_components_are_infinite_without_singular_solve(self):
        net = Network(["t", "a", "x", "y", "z"])
        net.add_conductor("t", "a", 1)
        net.add_conductor("x", "y", 1)
        result = expected_hitting_times(net, ["t"])
        self.assertEqual(result.expectations["t"], 0)
        self.assertEqual(result.expectations["a"], 1)
        for node in ("x", "y", "z"):
            self.assertTrue(math.isinf(result.expectations[node]))
            self.assertIsNone(result.residuals[node])
        self.assertEqual(result.reachable_nodes, ("t", "a"))
        self.assertEqual(result.unreachable_nodes, ("x", "y", "z"))
        self.assertEqual(
            result.unreachable_components,
            [("x", "y"), ("z",)],
        )
        self.assertEqual(result.zero_total_weight_nodes, ("z",))
        self.assertEqual(result.max_abs_residual, 0)

    def test_target_isolated_but_other_components_remain_infinite(self):
        net = Network(["t", "x"])
        result = expected_hitting_times(net, ["t"])
        self.assertEqual(result.expectations, {"t": 0, "x": math.inf})
        self.assertEqual(result.transition_probabilities["t"], {"t": 1})
        self.assertEqual(result.transition_probabilities["x"], {})

    def test_invalid_targets_and_stay_weights(self):
        net = Network(["a", "b"])
        net.add_conductor("a", "b", 1)
        with self.assertRaises(PortError):
            expected_hitting_times(net, [])
        with self.assertRaises(PortError):
            expected_hitting_times(net, ["a", "a"])
        with self.assertRaises(Exception):
            expected_hitting_times(net, ["missing"])
        with self.assertRaises(StayWeightError):
            expected_hitting_times(net, ["a"], stay_weights={"b": -1})
        with self.assertRaises(StayWeightError):
            expected_hitting_times(net, ["a"], stay_weights={"b": 0.5})

    def test_stay_weight_is_separate_from_resistor_self_loop_rule(self):
        net = Network(["a"])
        net.set_stay_weight("a", 3)
        self.assertEqual(net.stay_weights, (3,))
        self.assertEqual(net.laplacian(), [[0]])
        with self.assertRaises(Exception):
            net.add_conductor("a", "a", 1)

    def test_per_call_stay_weight_override_does_not_mutate_network(self):
        net = Network(["a", "b"])
        net.add_conductor("a", "b", 1)
        plain = expected_hitting_times(net, ["b"])
        held = expected_hitting_times(net, ["b"], stay_weights={"a": 3})
        self.assertEqual(plain.expectations["a"], 1)
        self.assertEqual(held.expectations["a"], 4)
        self.assertEqual(net.stay_weights, (0, 0))
        self.assertEqual(net.laplacian(), [[1, -1], [-1, 1]])


class TestCommuteTimeIdentity(unittest.TestCase):
    def test_weighted_triangle_commute_time_equals_2m_effective_resistance(self):
        net = Network(["a", "b", "c"])
        net.add_conductor("a", "b", 1)
        net.add_conductor("b", "c", 2)
        net.add_conductor("c", "a", 3)
        h_ac = expected_hitting_times(net, ["c"]).expectations["a"]
        h_ca = expected_hitting_times(net, ["a"]).expectations["c"]
        red = schur_eliminate(net, ["a", "c"])
        effective_resistance = Fraction(1) / red.dtn[0][0]
        total_conductance = Fraction(1 + 2 + 3)
        self.assertEqual(h_ac, Fraction(15, 11))
        self.assertEqual(h_ca, Fraction(21, 11))
        self.assertEqual(
            h_ac + h_ca,
            2 * total_conductance * effective_resistance,
        )


if __name__ == "__main__":
    unittest.main()
