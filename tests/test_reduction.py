"""Schur 消元、DtN 性质、内部量恢复与功率守恒测试。

期望值来源彼此独立：
* 串/并联解析式（``tests.reference``）；
* 全图线性系统独立参考（独立拼装拉普拉斯 + 独立高斯消元）；
* 简单网络的手算分数值。
被测核心不参与任何期望值的计算。
"""

from __future__ import annotations

import unittest
from fractions import Fraction

from resistor_elim import (
    Network,
    PortError,
    PotentialError,
    ZeroResistanceError,
    schur_eliminate,
)
from reference import RefNetwork, parallel_dtn, series_dtn


def frac_matrix(m):
    return [[Fraction(x) for x in row] for row in m]


class TestSeriesParallelAnalytic(unittest.TestCase):
    def test_series_three_resistors(self):
        net = Network(["p", "m", "q"])
        net.add_resistor("p", "m", 2)
        net.add_resistor("m", "q", 6)
        red = schur_eliminate(net, ["p", "q"])
        self.assertEqual(red.dtn, frac_matrix(series_dtn([2, 6])))
        # 消除后只剩端口；m 是被消除的内部节点
        comp = red.ported_components[0]
        self.assertEqual(comp.interior_labels, ("m",))

    def test_series_fractional_resistances(self):
        net = Network(["p", "a", "b", "q"])
        net.add_resistor("p", "a", Fraction(1, 2))
        net.add_resistor("a", "b", Fraction(1, 3))
        net.add_resistor("b", "q", Fraction(1, 6))
        red = schur_eliminate(net, ["p", "q"])
        self.assertEqual(red.dtn, frac_matrix(series_dtn([Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)])))

    def test_parallel_three_conductors(self):
        net = Network(["p", "q"])
        net.add_conductor("p", "q", 2)
        net.add_conductor("p", "q", Fraction(1, 4))
        net.add_resistor("p", "q", 5)
        red = schur_eliminate(net, ["p", "q"])
        self.assertEqual(
            red.dtn,
            frac_matrix(parallel_dtn([2, Fraction(1, 4), Fraction(1, 5)])),
        )

    def test_wheatstone_equivalent_analytic(self):
        # 两条 1Ω+1Ω 串联支路（各 2Ω）并联 ⇒ R_eq = 1Ω，g_eq = 1
        net = Network(["p", "q", "a", "b"])
        for u, v in [("p", "a"), ("a", "q"), ("p", "b"), ("b", "q")]:
            net.add_resistor(u, v, 1)
        red = schur_eliminate(net, ["p", "q"])
        self.assertEqual(red.dtn, [[1, -1], [-1, 1]])


class TestDtNProperties(unittest.TestCase):
    def _bridged_network(self):
        net = Network(["p", "q", "a", "b", "c"])
        net.add_resistor("p", "a", 1)
        net.add_resistor("a", "q", 2)
        net.add_resistor("p", "b", 3)
        net.add_resistor("b", "c", 4)
        net.add_resistor("c", "q", 5)
        net.add_resistor("a", "b", 6)
        return net

    def test_symmetric_row_sums_zero(self):
        net = self._bridged_network()
        red = schur_eliminate(net, ["p", "q"])
        n = 2
        for i in range(n):
            self.assertEqual(sum(red.dtn[i]), 0)
            for j in range(n):
                self.assertEqual(red.dtn[i][j], red.dtn[j][i])

    def test_diagonal_dominant_and_nonnegative_effective_conductance(self):
        net = self._bridged_network()
        red = schur_eliminate(net, ["p", "q"])
        self.assertGreater(red.dtn[0][0], 0)
        self.assertLess(red.dtn[0][1], 0)

    def test_schur_record_reproduces_block_formula(self):
        net = self._bridged_network()
        red = schur_eliminate(net, ["p", "q"])
        rec = red.ported_components[0].record
        # S = K_PP - K_PI K_II^-1 K_IP，用记录内的因子独立重算
        from resistor_elim.linalg import Mat

        k_ip = [list(col) for col in zip(*rec.k_pi)]
        prod = Mat.matmul(Mat.matmul(rec.k_pi, rec.kii_inverse), k_ip)
        rebuilt = [
            [rec.k_pp[i][j] - prod[i][j] for j in range(2)]
            for i in range(2)
        ]
        self.assertEqual(rebuilt, red.dtn)


class TestSolutionRecovery(unittest.TestCase):
    def test_voltage_divider_hand_computed(self):
        # 两个 2Ω 串联接 10V：中点 5V，电流 2.5A，总耗散 25W
        net = Network(["p", "m", "q"])
        net.add_resistor("p", "m", 2)
        net.add_resistor("m", "q", 2)
        red = schur_eliminate(net, ["p", "q"])
        sol = red.solve({"p": 10, "q": 0})
        self.assertEqual(sol.potentials["m"], 5)
        self.assertEqual(sol.port_currents["p"], Fraction(5, 2))
        self.assertEqual(sol.port_currents["q"], Fraction(-5, 2))
        self.assertEqual(sol.branch_currents, [Fraction(5, 2), Fraction(5, 2)])
        self.assertEqual(sol.branch_power, [Fraction(25, 2), Fraction(25, 2)])
        self.assertEqual(sol.port_power, 25)
        self.assertEqual(sol.dissipated_power, 25)
        self.assertEqual(sol.port_power, sol.dissipated_power)

    def test_sequence_potentials_order(self):
        net = Network(["p", "m", "q"])
        net.add_resistor("p", "m", 1)
        net.add_resistor("m", "q", 1)
        red = schur_eliminate(net, ["p", "q"])
        sol = red.solve([3, 1])
        self.assertEqual(sol.potentials["m"], 2)

    def test_float_potential_exact_conversion(self):
        net = Network(["p", "q"])
        net.add_resistor("p", "q", 4)
        red = schur_eliminate(net, ["p", "q"])
        sol = red.solve([0.5, 0.0])  # 0.5 精确可表示
        self.assertEqual(sol.port_currents["p"], Fraction(1, 8))

    def test_nan_inf_potentials_rejected(self):
        net = Network(["p", "q"])
        net.add_resistor("p", "q", 1)
        red = schur_eliminate(net, ["p", "q"])
        with self.assertRaises(PotentialError):
            red.solve([float("nan"), 0])
        with self.assertRaises(PotentialError):
            red.solve([float("inf"), 0])

    def test_power_conservation_random_deterministic_graphs(self):
        # 确定性“随机”图：用线性同余生成器，不依赖随机种子外状态
        rng_state = 1234567

        def rng():
            nonlocal rng_state
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            return rng_state

        for trial in range(12):
            n = 5 + trial % 4
            labels = list(range(n))
            net = Network(labels)
            ref_branches = []
            for i in range(n):
                for j in range(i + 1, n):
                    if rng() % 3 != 0:
                        g = Fraction(rng() % 7 + 1, rng() % 5 + 1)
                        net.add_conductor(i, j, g)
                        ref_branches.append((i, j, g))
            # 保证 0,1 在同一组件：加一条边
            net.add_conductor(0, 1, 1)
            ref_branches.append((0, 1, Fraction(1)))
            red = schur_eliminate(net, [0, 1, n - 1])
            vport = {0: Fraction(rng() % 6), 1: Fraction(rng() % 3), n - 1: 0}
            sol = red.solve(vport)

            # 与独立全图线性系统参考逐节点比较
            ref = RefNetwork(labels, ref_branches)
            ref_v = ref.solve_with_port_potentials([0, 1, n - 1], vport)
            for node, v in ref_v.items():
                self.assertEqual(sol.potentials[node], v, f"trial {trial} node {node}")
            ref_i = ref.port_currents([0, 1, n - 1], ref_v)
            for p, i in ref_i.items():
                self.assertEqual(sol.port_currents[p], i, f"trial {trial} port {p}")
            ref_pq = ref.branch_quantities(ref_v)
            for (got, want) in zip(sol.branch_power, ref_pq):
                self.assertIsNotNone(want)
                self.assertEqual(got, want[1])

            # 对称性与行和
            for i in range(3):
                self.assertEqual(sum(red.dtn[i]), 0)
                for j in range(3):
                    self.assertEqual(red.dtn[i][j], red.dtn[j][i])
            # 功率守恒（精确）：注入 == 耗散 >= 0
            self.assertEqual(sol.port_power, sol.dissipated_power)
            self.assertGreaterEqual(sol.dissipated_power, 0)


class TestFloatingComponents(unittest.TestCase):
    def test_floating_component_marked_not_hidden(self):
        net = Network(["p", "q", "x", "y", "z"])
        net.add_resistor("p", "q", 1)
        net.add_resistor("x", "y", 2)  # 无端口组件
        # z 为完全孤立节点
        red = schur_eliminate(net, ["p", "q"])
        floating = red.floating_components
        self.assertIn(("x", "y"), floating)
        self.assertIn(("z",), floating)
        # DtN 只描述端口组件
        self.assertEqual(red.dtn, [[1, -1], [-1, 1]])
        sol = red.solve({"p": 4, "q": 0})
        self.assertNotIn("x", sol.potentials)
        self.assertIn("z", sol.undetermined_nodes)
        self.assertIn("x", sol.undetermined_nodes)
        # 浮接支路返回 None，而非被当作 0 电流掩盖
        self.assertEqual(sol.branch_currents, [Fraction(4), None])
        self.assertEqual(sol.branch_power, [16, None])
        self.assertEqual(sol.port_power, sol.dissipated_power)

    def test_port_in_disconnected_component_still_solved(self):
        # 两个互相独立的端口组件分别消元、分别求解
        net = Network(["p", "q", "r", "s", "t"])
        net.add_resistor("p", "q", 2)
        net.add_resistor("q", "r", 2)
        net.add_resistor("s", "t", 4)
        red = schur_eliminate(net, ["p", "r", "s", "t"])
        self.assertEqual(len(red.ported_components), 2)
        sol = red.solve({"p": 8, "r": 0, "s": 8, "t": 0})
        self.assertEqual(sol.potentials["q"], 4)
        self.assertEqual(sol.port_currents["p"], 2)
        self.assertEqual(sol.port_currents["s"], 2)

    def test_multiple_ported_components_independent_dtN_blocks(self):
        net = Network(["p", "q", "r", "s"])
        net.add_resistor("p", "q", 1)
        net.add_resistor("r", "s", 1)
        red = schur_eliminate(net, ["p", "q", "r", "s"])
        # 全局 DtN 是两个 2x2 块的直和
        self.assertEqual(red.dtn[0][2], 0)
        self.assertEqual(red.dtn[2][0], 0)
        self.assertEqual(sum(red.dtn[0]), 0)


class TestPortAndReductionValidation(unittest.TestCase):
    def test_empty_unknown_duplicate_ports(self):
        net = Network(["a", "b"])
        net.add_resistor("a", "b", 1)
        with self.assertRaises(PortError):
            schur_eliminate(net, [])
        with self.assertRaises(Exception):
            schur_eliminate(net, ["a", "z"])
        with self.assertRaises(PortError):
            schur_eliminate(net, ["a", "a"])
        with self.assertRaises(PortError):
            schur_eliminate(net, [True])
        with self.assertRaises(PortError):
            schur_eliminate(net, "ab")  # 字符串不是端口序列

    def test_bool_potential_rejected_with_location(self):
        net = Network([1, 2])
        net.add_resistor(1, 2, 1)
        red = schur_eliminate(net, [1, 2])
        with self.assertRaises(PotentialError) as ctx:
            red.solve([True, 0])
        self.assertEqual(ctx.exception.node, 1)

    def test_solve_wrong_potential_count(self):
        net = Network(["a", "b"])
        net.add_resistor("a", "b", 1)
        red = schur_eliminate(net, ["a", "b"])
        with self.assertRaises(PotentialError):
            red.solve([1])
        with self.assertRaises(PotentialError):
            red.solve({"a": 1})
        with self.assertRaises(PotentialError):
            red.solve({"a": 1, "b": 0, "c": 2})

    def test_isolated_port_has_zero_current_row(self):
        # 端口本身无任何电导：DtN 对应行为零（不能伪造连接），求解不报错
        net = Network(["a", "b", "c"])
        net.add_resistor("a", "b", 1)
        red = schur_eliminate(net, ["a", "b", "c"])
        self.assertEqual(red.dtn[2], [0, 0, 0])
        sol = red.solve([5, 0, 9])
        self.assertEqual(sol.port_currents["c"], 0)

    def test_zero_resistance_boundary_in_demo_domain(self):
        net = Network(["a", "b"])
        with self.assertRaises(ZeroResistanceError):
            net.add_resistor("a", "b", 0)

    def test_30_node_chain_hits_limit_boundary_exactly(self):
        # 接口上限 30：29 条单位电阻串联 ⇒ 等效电导 1/29
        n = 30
        net = Network(range(n))
        for i in range(n - 1):
            net.add_resistor(i, i + 1, 1)
        red = schur_eliminate(net, [0, n - 1])
        self.assertEqual(
            red.dtn,
            frac_matrix(series_dtn([Fraction(1)] * (n - 1))),
        )
        self.assertEqual(red.dtn[0][0], Fraction(1, 29))
        comp = red.ported_components[0]
        self.assertEqual(len(comp.interior_labels), 28)
        # 内部电势线性分布
        sol = red.solve([29, 0])
        for i in range(n):
            self.assertEqual(sol.potentials[i], Fraction(29 - i))


if __name__ == "__main__":
    unittest.main()
