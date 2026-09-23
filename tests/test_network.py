"""网络构建、拉普拉斯与输入校验测试。"""

from __future__ import annotations

import unittest
from fractions import Fraction

from resistor_elim import (
    ConductanceError,
    LimitError,
    Network,
    NetworkError,
    NodeError,
    ResistanceError,
    ZeroResistanceError,
)
from resistor_elim.network import MAX_NODES


class TestNetworkBuild(unittest.TestCase):
    def test_simple_laplacian(self):
        net = Network(["a", "b", "c"])
        net.add_conductor("a", "b", 2)
        net.add_conductor("b", "c", Fraction(1, 3))
        k = net.laplacian()
        self.assertEqual(k[0][0], 2)
        self.assertEqual(k[0][1], -2)
        self.assertEqual(k[1][1], Fraction(7, 3))
        self.assertEqual(k[1][2], -Fraction(1, 3))
        self.assertEqual(k[2][2], Fraction(1, 3))
        # 拉普拉斯对称、行和为 0
        for i in range(3):
            self.assertEqual(sum(k[i]), 0)
            for j in range(3):
                self.assertEqual(k[i][j], k[j][i])

    def test_parallel_conductors_add(self):
        net = Network(["a", "b"])
        net.add_conductor("a", "b", 1)
        net.add_conductor("a", "b", Fraction(1, 2))
        net.add_conductor("b", "a", Fraction(1, 4))
        self.assertEqual(net.conductance_between("a", "b"), Fraction(7, 4))

    def test_resistor_is_reciprocal_conductance(self):
        net = Network(["a", "b"])
        net.add_resistor("a", "b", 7)
        self.assertEqual(net.conductance_between("a", "b"), Fraction(1, 7))
        net.add_resistor("a", "b", Fraction(2, 3))
        self.assertEqual(net.conductance_between("a", "b"), Fraction(1, 7) + Fraction(3, 2))

    def test_components_including_isolated(self):
        net = Network([1, 2, 3, 4, 5])
        net.add_conductor(1, 2, 1)
        net.add_conductor(3, 4, 2)
        comps = net.components()
        self.assertEqual(comps, [(1, 2), (3, 4), (5,)])

    def test_node_limit_boundary(self):
        Network(range(MAX_NODES))  # 30 可用
        with self.assertRaises(LimitError):
            Network(range(MAX_NODES + 1))

    def test_non_hashable_and_bool_labels(self):
        with self.assertRaises(NodeError):
            Network([True])
        with self.assertRaises(NodeError):
            Network([["x"]])

    def test_duplicate_label(self):
        with self.assertRaises(NodeError):
            Network(["a", "a"])


class TestInputValidation(unittest.TestCase):
    def _net(self):
        return Network(["a", "b", "c"])

    def test_zero_resistance_explicitly_rejected(self):
        net = self._net()
        with self.assertRaises(ZeroResistanceError) as ctx:
            net.add_resistor("a", "b", 0)
        self.assertEqual(ctx.exception.edge, ("a", "b"))
        self.assertIsInstance(ctx.exception, ResistanceError)
        self.assertEqual(len(net.branches), 0)  # 失败不留部分变更

    def test_negative_resistance_and_conductance(self):
        net = self._net()
        with self.assertRaises(ResistanceError):
            net.add_resistor("a", "b", -1)
        with self.assertRaises(ConductanceError):
            net.add_conductor("a", "b", 0)
        with self.assertRaises(ConductanceError):
            net.add_conductor("a", "b", Fraction(-1, 2))
        self.assertEqual(len(net.branches), 0)

    def test_float_conductance_rejected_even_finite(self):
        net = self._net()
        with self.assertRaises(ConductanceError):
            net.add_conductor("a", "b", 1.0)
        self.assertEqual(len(net.branches), 0)

    def test_nan_infinity_conductance_messages(self):
        net = self._net()
        with self.assertRaises(ConductanceError) as ctx:
            net.add_conductor("a", "b", float("nan"))
        self.assertIn("NaN", str(ctx.exception))
        with self.assertRaises(ConductanceError) as ctx:
            net.add_conductor("a", "b", float("inf"))
        self.assertIn("Infinity", str(ctx.exception))
        with self.assertRaises(ConductanceError):
            net.add_conductor("a", "b", float("-inf"))

    def test_bool_rejected_everywhere(self):
        net = self._net()
        with self.assertRaises(TypeError):
            net.add_conductor("a", "b", True)
        with self.assertRaises(TypeError):
            net.add_resistor("a", "b", False)

    def test_unknown_node_and_self_loop(self):
        net = self._net()
        with self.assertRaises(NodeError) as ctx:
            net.add_conductor("a", "z", 1)
        self.assertEqual(ctx.exception.node, "z")
        with self.assertRaises(ConductanceError):
            net.add_conductor("a", "a", 1)
        self.assertEqual(len(net.branches), 0)

    def test_bad_value_types(self):
        net = self._net()
        with self.assertRaises(ConductanceError):
            net.add_conductor("a", "b", "1")
        with self.assertRaises(ResistanceError):
            net.add_resistor("a", "b", 2 + 0j)

    def test_failed_add_is_atomic(self):
        net = self._net()
        net.add_conductor("a", "b", 3)
        with self.assertRaises(NetworkError):
            net.add_conductor("a", "b", -3)
        with self.assertRaises(NetworkError):
            net.add_conductor("a", "c", 0)
        # 仅第一次成功的支路保留，且值未被污染
        self.assertEqual(len(net.branches), 1)
        self.assertEqual(net.conductance_between("a", "b"), 3)


if __name__ == "__main__":
    unittest.main()
