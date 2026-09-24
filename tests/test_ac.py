"""R/L/C 相量修正结点分析与端口复导纳测试。

期望值来源彼此独立：
* 解析公式：单 RC（并/串）、串 RLC、纯电阻导纳（``tests.reference``）；
* 全图独立相量参考：独立复数四则 + 独立复高斯消元 + 独立 Schur 拼装；
* 手算分数值与无源网络性质（Hermitian、行和 0、实部非负、谐振极点）。
被测核心不参与任何期望值计算。
"""

from __future__ import annotations

import unittest
from fractions import Fraction

from resistor_elim import (
    CFrac,
    CapacitanceError,
    FrequencyError,
    InductanceError,
    Network,
    NetworkError,
    NodeError,
    PortError,
    PotentialError,
    SingularSystemError,
    ac_schur_eliminate,
    schur_eliminate,
)
from reference import (
    RefACNetwork,
    analytic_parallel_rc,
    analytic_resistor_admittance,
    analytic_series_rc,
    analytic_series_rlc,
    series_dtn,
)


def cf(t):
    """参考的 (re, im) 二元组 -> CFrac。"""
    return CFrac(t[0], t[1])


def assert_cf(testcase, got: CFrac, want, msg=""):
    testcase.assertEqual(got.re, want[0], f"{msg} 实部 {got} != {want}")
    testcase.assertEqual(got.im, want[1], f"{msg} 虚部 {got} != {want}")


def dtn2(y):
    """二端口导纳矩阵取 (0,0) 与 (0,1)。"""
    return y[0][0], y[0][1]


class TestAnalyticSingleRC(unittest.TestCase):
    def test_parallel_rc(self):
        # R=3（g=1/3）与 C=2/5 并联，ω=7：Y = 1/3 + j·14/5
        net = Network(["p", "q"])
        net.add_resistor("p", "q", 3)
        net.add_capacitor("p", "q", Fraction(2, 5))
        w = 7
        red = ac_schur_eliminate(net, ["p", "q"], w)
        want = analytic_parallel_rc(Fraction(1, 3), Fraction(2, 5), w)
        y00, y01 = dtn2(red.admittance)
        assert_cf(self, y00, want, "并联 RC Y00")
        assert_cf(self, y01, (-want[0], -want[1]), "并联 RC Y01")
        self.assertEqual(y00, red.admittance[1][1])
        self.assertEqual(y01, red.admittance[1][0])

    def test_series_rc_hand_values(self):
        # R=2, C=1/4, ω=2：Z=2+1/(j·2·1/4)=2-2j；Y=(1/4)+(1/4)j
        net = Network(["p", "m", "q"])
        net.add_resistor("p", "m", 2)
        net.add_capacitor("m", "q", Fraction(1, 4))
        red = ac_schur_eliminate(net, ["p", "q"], 2)
        want = analytic_series_rc(2, Fraction(1, 4), 2)
        assert_cf(self, red.admittance[0][0], want, "串联 RC Y00")
        assert_cf(self, red.admittance[0][1], (-want[0], -want[1]))
        # 手算内部量：Vp=1,Vq=0，I=Y·1=(1/4+1/4 j)；
        # Vm = Vp - I·R = 1 - 2·(1/4+1/4j) = 1/2 - 1/2 j
        sol = red.solve({"p": 1, "q": 0})
        assert_cf(self, sol.potentials["m"], (Fraction(1, 2), Fraction(-1, 2)))
        assert_cf(self, sol.port_currents["p"], want)

    def test_series_rc_several_frequencies_against_formula(self):
        for w in (Fraction(1, 2), 1, 3, Fraction(17, 4)):
            net = Network(["p", "m", "q"])
            net.add_resistor("p", "m", 5)
            net.add_capacitor("m", "q", Fraction(2, 3))
            red = ac_schur_eliminate(net, ["p", "q"], w)
            want = analytic_series_rc(5, Fraction(2, 3), w)
            assert_cf(self, red.admittance[0][0], want, f"ω={w}")


class TestAnalyticSeriesRLC(unittest.TestCase):
    def test_resonance_impedance_real(self):
        # R=L=C=1, ω=1：jωL + 1/(jωC) = 0 ⇒ Z=1 ⇒ Y=1（实数）
        net = Network(["p", "a", "b", "q"])
        net.add_resistor("p", "a", 1)
        net.add_inductor("a", "b", 1)
        net.add_capacitor("b", "q", 1)
        red = ac_schur_eliminate(net, ["p", "q"], 1)
        assert_cf(self, red.admittance[0][0], analytic_series_rlc(1, 1, 1, 1))
        self.assertEqual(red.admittance[0][0], CFrac(1))

    def test_sweep_against_formula(self):
        r, l, c = 2, Fraction(3, 2), Fraction(4, 5)
        for w in (Fraction(1, 3), 1, Fraction(5, 2), 4, 11):
            net = Network(["p", "x", "y", "q"])
            net.add_resistor("p", "x", r)
            net.add_inductor("x", "y", l)
            net.add_capacitor("y", "q", c)
            red = ac_schur_eliminate(net, ["p", "q"], w)
            want = analytic_series_rlc(r, l, c, w)
            assert_cf(self, red.admittance[0][0], want, f"串 RLC ω={w}")

    def test_below_above_resonance_reactance_sign(self):
        # ω0 = 1/sqrt(LC) = 1/sqrt(2)；取 ω=1/2（容性，Z 虚部<0，Y 虚部>0）
        # 与 ω=2（感性，Z 虚部>0，Y 虚部<0）
        net = Network(["p", "x", "y", "q"])
        net.add_resistor("p", "x", 1)
        net.add_inductor("x", "y", 2)
        net.add_capacitor("y", "q", 1)
        y_low = ac_schur_eliminate(net, ["p", "q"], Fraction(1, 2)).admittance[0][0]
        y_high = ac_schur_eliminate(net, ["p", "q"], 2).admittance[0][0]
        self.assertGreater(y_low.im, 0)
        self.assertLess(y_high.im, 0)


class TestPureResistorAgreesWithDC(unittest.TestCase):
    def test_ac_admittance_equals_dc_laplacian_schur(self):
        net = Network(["p", "q", "a", "b"])
        net.add_resistor("p", "a", 2)
        net.add_resistor("a", "q", 3)
        net.add_resistor("p", "b", 4)
        net.add_resistor("b", "q", 5)
        net.add_resistor("a", "b", 6)
        dc = schur_eliminate(net, ["p", "q"])
        for w in (Fraction(1, 7), 1, 13):
            ac = ac_schur_eliminate(net, ["p", "q"], w)
            for i in range(2):
                for j in range(2):
                    self.assertEqual(ac.admittance[i][j].re, dc.dtn[i][j])
                    self.assertEqual(ac.admittance[i][j].im, 0)

    def test_pure_resistor_series_analytic(self):
        net = Network(["p", "m", "q"])
        net.add_resistor("p", "m", 2)
        net.add_resistor("m", "q", 6)
        ac = ac_schur_eliminate(net, ["p", "q"], 5)
        want = series_dtn([2, 6])
        self.assertEqual(
            [[(z.re, z.im) for z in row] for row in ac.admittance],
            [[(Fraction(1, 8), 0), (-Fraction(1, 8), 0)],
             [(-Fraction(1, 8), 0), (Fraction(1, 8), 0)]],
        )
        assert_cf(self, ac.admittance[0][0], analytic_resistor_admittance(Fraction(1, 8)))
        self.assertEqual(want[0][0], Fraction(1, 8))


class TestPassiveProperties(unittest.TestCase):
    def _net(self):
        net = Network(["p", "q", "a", "b", "c"])
        net.add_resistor("p", "a", 1)
        net.add_capacitor("a", "q", 2)
        net.add_inductor("p", "b", 3)
        net.add_resistor("b", "c", 4)
        net.add_capacitor("c", "q", Fraction(1, 2))
        net.add_inductor("a", "b", 5)
        return net

    def test_symmetric_row_sums_and_passive(self):
        # 互易无源网络的相量结点导纳是**复对称**矩阵（Y_ij = Y_ji），
        # 行和为 0；正实性体现为 Hermitian 部分 (Y+Y^H)/2 = Re(Y)
        # 半正定，即任意相量激励 v 下 Re(v^H Y v) >= 0（平均耗散非负）。
        net = self._net()
        red = ac_schur_eliminate(net, ["p", "q"], Fraction(3, 2))
        y = red.admittance
        zero = CFrac()
        probes = [
            (CFrac(1), CFrac()),
            (CFrac(), CFrac(1)),
            (CFrac(1), CFrac(0, 1)),
            (CFrac(2, -3), CFrac(Fraction(1, 2), 4)),
        ]
        for i in range(2):
            self.assertEqual(sum(y[i], CFrac()), zero)  # 行和为 0
            for j in range(2):
                self.assertEqual(y[i][j], y[j][i])      # 复对称
        for v in probes:
            yv = [
                y[0][0] * v[0] + y[0][1] * v[1],
                y[1][0] * v[0] + y[1][1] * v[1],
            ]
            diss = (v[0].conjugate() * yv[0] + v[1].conjugate() * yv[1]).re
            self.assertGreaterEqual(diss, 0)

    def test_driving_point_conductance_positive(self):
        net = self._net()
        red = ac_schur_eliminate(net, ["p", "q"], Fraction(3, 2))
        self.assertGreater(red.admittance[0][0].re, 0)

    def test_negative_frequency_is_conjugate(self):
        net = self._net()
        yp = ac_schur_eliminate(net, ["p", "q"], 7).admittance
        ym = ac_schur_eliminate(net, ["p", "q"], -7).admittance
        for i in range(2):
            for j in range(2):
                self.assertEqual(ym[i][j], yp[i][j].conjugate())

    def test_reactive_branch_signs(self):
        net = Network(["p", "q"])
        net.add_capacitor("p", "q", 3)
        net.add_inductor("p", "q", 2)
        y = ac_schur_eliminate(net, ["p", "q"], 4).admittance[0][0]
        # jωC = j12；1/(jωL) = -j/8；合起来 j(12 - 1/8) = j·95/8，实部 0
        self.assertEqual(y.re, 0)
        self.assertEqual(y.im, 12 - Fraction(1, 8))


class TestIndependentReference(unittest.TestCase):
    def test_random_deterministic_rlc_networks(self):
        rng_state = 987654321

        def rng():
            nonlocal rng_state
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            return rng_state

        for trial in range(10):
            n = 5 + trial % 4
            labels = list(range(n))
            net = Network(labels)
            ref_branches = []
            kinds = ("R", "C", "L")
            for i in range(n):
                for j in range(i + 1, n):
                    if rng() % 3 != 0:
                        kind = kinds[rng() % 3]
                        val = Fraction(rng() % 5 + 1, rng() % 4 + 1)
                        if kind == "R":
                            net.add_conductor(i, j, val)
                        elif kind == "C":
                            net.add_capacitor(i, j, val)
                        else:
                            net.add_inductor(i, j, val)
                        ref_branches.append((i, j, kind, val))
            # 保证全图连通且严格有耗散：每个节点经 1Ω 接到节点 0，
            # 端口组件的 Y_II 实部严格正定，任何频率都不会谐振
            for i in range(1, n):
                net.add_resistor(0, i, 1)
                ref_branches.append((0, i, "R", Fraction(1)))
            ports = [0, 1, n - 1]
            w = Fraction(rng() % 9 + 1, rng() % 3 + 1)

            red = ac_schur_eliminate(net, ports, w)
            ref = RefACNetwork(labels, ref_branches)
            want_y = ref.port_admittance(ports, w)
            for a in range(3):
                for b in range(3):
                    assert_cf(
                        self, red.admittance[a][b], want_y[a][b],
                        f"trial {trial} ω={w} Y[{a}][{b}]",
                    )

            # 内部电势与端口电流对照独立全图求解
            vport = {
                0: CFrac(Fraction(rng() % 5), Fraction(rng() % 3 + 1)),
                1: CFrac(Fraction(rng() % 3), 0),
                n - 1: CFrac(),
            }
            sol = red.solve(vport)
            ref_v = ref.solve_with_port_potentials(
                ports, {k: (v.re, v.im) for k, v in vport.items()}, w
            )
            for node, v in ref_v.items():
                assert_cf(self, sol.potentials[node], v,
                          f"trial {trial} node {node}")
            ref_i = ref.port_currents(
                ports, ref_v, w
            )
            for p, cur in ref_i.items():
                assert_cf(self, sol.port_currents[p], cur,
                          f"trial {trial} port {p}")

            # 复对称 + 行和（互易无源网络）
            for a in range(3):
                self.assertEqual(sum(red.admittance[a], CFrac()), CFrac())
                for b in range(3):
                    self.assertEqual(red.admittance[a][b], red.admittance[b][a])
            # 正实性抽样：Re(v^H Y v) >= 0
            probes_v = [
                [CFrac(1), CFrac(), CFrac()],
                [CFrac(), CFrac(1), CFrac()],
                [CFrac(1), CFrac(0, 1), CFrac(Fraction(1, 2), -3)],
            ]
            for v in probes_v:
                yv = [
                    sum((red.admittance[r][c] * v[c] for c in range(3)), CFrac())
                    for r in range(3)
                ]
                diss = sum((v[r].conjugate() * yv[r] for r in range(3)), CFrac()).re
                self.assertGreaterEqual(diss, 0, f"trial {trial} 正实性")


class TestSolutionPower(unittest.TestCase):
    def test_complex_power_balance(self):
        net = Network(["p", "a", "q"])
        net.add_resistor("p", "a", 2)
        net.add_capacitor("a", "q", Fraction(1, 2))
        red = ac_schur_eliminate(net, ["p", "q"], 3)
        sol = red.solve({"p": CFrac(0, 4), "q": 0})  # Vp = 4j
        # 端口复功率实部 = 电阻平均耗散（精确），且 ≥ 0
        self.assertEqual(sol.port_power.re, sol.dissipated_power)
        self.assertGreaterEqual(sol.dissipated_power, 0)
        # 电容支路平均耗散为 0（复功率纯虚）
        cap_power = sol.branch_power[1]
        self.assertEqual(cap_power.re, 0)
        self.assertNotEqual(cap_power.im, 0)
        # 复功率 Tellegen 恒等式（精确）：端口复功率 == 全部支路复功率之和
        branch_total = sol.determined_branch_power()
        self.assertEqual(sol.port_power, branch_total)

    def test_complex_power_balance_general_random(self):
        net = Network(["p", "a", "b", "q"])
        net.add_resistor("p", "a", 2)
        net.add_inductor("a", "b", Fraction(3, 2))
        net.add_capacitor("b", "q", Fraction(4, 5))
        net.add_resistor("a", "q", 7)
        red = ac_schur_eliminate(net, ["p", "q"], Fraction(2, 3))
        sol = red.solve({"p": CFrac(3, -2), "q": CFrac(1, 1)})
        self.assertEqual(sol.port_power, sol.determined_branch_power())

    def test_branch_currents_align_with_branches_ac(self):
        net = Network(["p", "q"])
        net.add_resistor("p", "q", 2)
        net.add_capacitor("p", "q", 1)
        net.add_inductor("p", "q", 3)
        kinds = [b[2] for b in net.branches_ac]
        self.assertEqual(kinds, ["R", "C", "L"])
        red = ac_schur_eliminate(net, ["p", "q"], 2)
        sol = red.solve([6, 0])
        dv = CFrac(6)
        self.assertEqual(sol.branch_currents[0], CFrac(Fraction(1, 2)) * dv)
        self.assertEqual(sol.branch_currents[1], CFrac(0, 2) * dv)
        self.assertEqual(sol.branch_currents[2], CFrac(0, -Fraction(1, 6)) * dv)

    def test_potential_count_and_bad_complex(self):
        net = Network(["p", "q"])
        net.add_resistor("p", "q", 1)
        red = ac_schur_eliminate(net, ["p", "q"], 1)
        with self.assertRaises(PotentialError):
            red.solve([1])
        with self.assertRaises(PotentialError):
            red.solve([complex(float("nan"), 0), 0])
        with self.assertRaises(PotentialError):
            red.solve([1, complex(0, float("inf"))])
        with self.assertRaises(PotentialError):
            red.solve([True, 0])

    def test_finite_float_omega_exact_conversion(self):
        net = Network(["p", "q"])
        net.add_capacitor("p", "q", 4)
        red = ac_schur_eliminate(net, ["p", "q"], 0.5)  # 精确 1/2
        self.assertEqual(red.omega, Fraction(1, 2))
        # Y = j·(1/2)·4 = 2j
        self.assertEqual(red.admittance[0][0], CFrac(0, 2))


class TestFloatingAndResonance(unittest.TestCase):
    def test_floating_lc_component_marked_even_at_resonance(self):
        # 无端口的 L//C 在 ω=1 谐振：无法被激励，不报错，标记 floating
        net = Network(["p", "q", "x", "y", "z"])
        net.add_resistor("p", "q", 1)
        net.add_inductor("x", "y", 1)
        net.add_capacitor("x", "y", 1)
        red = ac_schur_eliminate(net, ["p", "q"], 1)
        self.assertEqual(red.floating_components, [("x", "y"), ("z",)])
        self.assertEqual(red.admittance[0][0], CFrac(1))
        sol = red.solve([4, 0])
        self.assertNotIn("x", sol.potentials)
        self.assertIn("z", sol.undetermined_nodes)
        # 浮置支路电流/功率为 None，而非伪装成 0
        kinds_idx = {b[2]: i for i, b in enumerate(net.branches_ac)}
        self.assertIsNone(sol.branch_currents[kinds_idx["L"]])
        self.assertIsNone(sol.branch_power[kinds_idx["C"]])

    def test_series_lc_resonance_reported_explicitly(self):
        # p-L-m-C-q：L=C=1，ω=1 时串联阻抗 0、导纳极点（Y_II 奇异）
        net = Network(["p", "m", "q"])
        net.add_inductor("p", "m", 1)
        net.add_capacitor("m", "q", 1)
        with self.assertRaises(SingularSystemError) as ctx:
            ac_schur_eliminate(net, ["p", "q"], 1)
        self.assertEqual(ctx.exception.resonant_components, [("p", "m", "q")])
        # 同一网络偏离谐振即正常（Z = j(ω-1/ω)）
        red = ac_schur_eliminate(net, ["p", "q"], 2)
        self.assertEqual(red.admittance[0][0].re, 0)
        self.assertNotEqual(red.admittance[0][0].im, 0)

    def test_damped_resonance_is_regular(self):
        # 在谐振 LC 串里串一个小电阻：极点被阻尼，任何频率都可解
        net = Network(["p", "m", "q"])
        net.add_resistor("p", "m", Fraction(1, 10))
        net.add_inductor("m", "q", 1)
        net.add_capacitor("m", "q", 1)  # L、C 并联再与 R 串联
        red = ac_schur_eliminate(net, ["p", "q"], 1)
        # ω=1 时 L//C 并联导纳为 0（开路）⇒ 总支路导纳为 0
        self.assertEqual(red.admittance[0][0], CFrac())

    def test_isolated_port_zero_row(self):
        net = Network(["p", "q", "r"])
        net.add_resistor("p", "q", 1)
        red = ac_schur_eliminate(net, ["p", "q", "r"], 2)
        self.assertEqual(red.admittance[2], [CFrac(), CFrac(), CFrac()])

    def test_independent_components_are_direct_sum(self):
        # 两个互不相连的端口组件：一个纯电阻，一个串 RC；全局导纳为直和
        net = Network(["p", "q", "r", "m", "s"])
        net.add_resistor("p", "q", 1)
        net.add_resistor("r", "m", 2)
        net.add_capacitor("m", "s", Fraction(1, 4))
        red = ac_schur_eliminate(net, ["p", "q", "r", "s"], 2)
        self.assertEqual(len(red.ported_components), 2)
        # 跨组件块为零
        for i in range(4):
            for j in range(4):
                if (i < 2) != (j < 2):
                    self.assertEqual(red.admittance[i][j], CFrac())
        # 第一块 1Ω 纯电阻；第二块串 RC = 1/4+1/4j
        self.assertEqual(red.admittance[0][0], CFrac(1))
        self.assertEqual(red.admittance[2][2], CFrac(Fraction(1, 4), Fraction(1, 4)))

    def test_30_node_rlc_chain_boundary(self):
        # 接口上限 30：交替 R/C 的 29 支路链，与独立参考逐值对照
        n = 30
        net = Network(range(n))
        ref_branches = []
        for i in range(n - 1):
            if i % 2 == 0:
                net.add_resistor(i, i + 1, 1)
                ref_branches.append((i, i + 1, "R", Fraction(1)))
            else:
                net.add_capacitor(i, i + 1, 1)
                ref_branches.append((i, i + 1, "C", Fraction(1)))
        red = ac_schur_eliminate(net, [0, n - 1], 3)
        ref = RefACNetwork(list(range(n)), ref_branches)
        want = ref.port_admittance([0, n - 1], 3)
        assert_cf(self, red.admittance[0][0], want[0][0])
        assert_cf(self, red.admittance[0][1], want[0][1])
        self.assertEqual(len(red.ported_components[0].interior_labels), 28)

    def test_failure_leaves_no_partial_state(self):
        # 谐振报错时不修改网络本身
        net = Network(["p", "m", "q"])
        net.add_inductor("p", "m", 1)
        net.add_capacitor("m", "q", 1)
        n_branches = len(net.branches_ac)
        with self.assertRaises(SingularSystemError):
            ac_schur_eliminate(net, ["p", "q"], 1)
        self.assertEqual(len(net.branches_ac), n_branches)


class TestFrequencyValidation(unittest.TestCase):
    def _net(self):
        net = Network(["p", "q"])
        net.add_resistor("p", "q", 1)
        return net

    def test_zero_frequency_rejected(self):
        with self.assertRaises(FrequencyError):
            ac_schur_eliminate(self._net(), ["p", "q"], 0)
        with self.assertRaises(FrequencyError):
            ac_schur_eliminate(self._net(), ["p", "q"], 0.0)

    def test_nan_inf_frequency_rejected(self):
        with self.assertRaises(FrequencyError):
            ac_schur_eliminate(self._net(), ["p", "q"], float("nan"))
        with self.assertRaises(FrequencyError):
            ac_schur_eliminate(self._net(), ["p", "q"], float("inf"))
        with self.assertRaises(FrequencyError):
            ac_schur_eliminate(self._net(), ["p", "q"], float("-inf"))

    def test_bad_type_and_bool(self):
        with self.assertRaises(FrequencyError):
            ac_schur_eliminate(self._net(), ["p", "q"], "1")
        with self.assertRaises(TypeError):
            ac_schur_eliminate(self._net(), ["p", "q"], True)

    def test_port_errors_shared_contract(self):
        net = self._net()
        with self.assertRaises(PortError):
            ac_schur_eliminate(net, [], 1)
        with self.assertRaises(PortError):
            ac_schur_eliminate(net, ["p", "p"], 1)
        with self.assertRaises(NodeError):
            ac_schur_eliminate(net, ["p", "z"], 1)
        with self.assertRaises(TypeError):
            ac_schur_eliminate("not a network", ["p"], 1)


class TestComponentValidation(unittest.TestCase):
    def _net(self):
        return Network(["p", "q"])

    def test_nonpositive(self):
        net = self._net()
        for fn, err in ((net.add_capacitor, CapacitanceError),
                        (net.add_inductor, InductanceError)):
            with self.assertRaises(err):
                fn("p", "q", 0)
            with self.assertRaises(err):
                fn("p", "q", -1)
            with self.assertRaises(err):
                fn("p", "q", Fraction(-1, 2))
        self.assertEqual(len(net.branches_ac), 0)

    def test_float_nan_inf_rejected(self):
        net = self._net()
        with self.assertRaises(CapacitanceError):
            net.add_capacitor("p", "q", 1.0)
        with self.assertRaises(CapacitanceError):
            net.add_capacitor("p", "q", float("nan"))
        with self.assertRaises(InductanceError):
            net.add_inductor("p", "q", float("inf"))
        with self.assertRaises(InductanceError):
            net.add_inductor("p", "q", 2.5)
        self.assertEqual(len(net.branches_ac), 0)

    def test_bool_self_loop_unknown_node(self):
        net = self._net()
        with self.assertRaises(TypeError):
            net.add_capacitor("p", "q", True)
        with self.assertRaises(TypeError):
            net.add_inductor("p", "q", False)
        with self.assertRaises(CapacitanceError):
            net.add_capacitor("p", "p", 1)
        with self.assertRaises(InductanceError):
            net.add_inductor("q", "q", 1)
        with self.assertRaises(NodeError) as ctx:
            net.add_capacitor("p", "z", 1)
        self.assertEqual(ctx.exception.node, "z")
        self.assertEqual(len(net.branches_ac), 0)

    def test_bad_types_and_atomicity(self):
        net = self._net()
        with self.assertRaises(CapacitanceError):
            net.add_capacitor("p", "q", "1")
        with self.assertRaises(InductanceError):
            net.add_inductor("p", "q", 1 + 0j)
        net.add_capacitor("p", "q", 2)
        with self.assertRaises(NetworkError):
            net.add_inductor("p", "q", 0)
        self.assertEqual(len(net.branches_ac), 1)
        self.assertEqual(net.capacitance_between("p", "q"), 2)

    def test_parallel_c_and_l_combination(self):
        net = self._net()
        net.add_capacitor("p", "q", 1)
        net.add_capacitor("q", "p", Fraction(1, 2))
        net.add_inductor("p", "q", 2)
        net.add_inductor("p", "q", 2)  # 两 2H 并联 ⇒ 1H
        self.assertEqual(net.capacitance_between("p", "q"), Fraction(3, 2))
        self.assertEqual(net.inductance_between("p", "q"), 1)

    def test_dc_laplacian_ignores_reactive_elements(self):
        net = self._net()
        net.add_resistor("p", "q", 2)
        net.add_capacitor("p", "q", 9)
        net.add_inductor("p", "q", 9)
        k = net.laplacian()
        self.assertEqual(k[0][0], Fraction(1, 2))  # 仅电阻
        # 旧 DC 消元结果不受影响
        dc = schur_eliminate(net, ["p", "q"])
        self.assertEqual(dc.dtn[0][0], Fraction(1, 2))
        # 旧 branches 只含电阻
        self.assertEqual([b[2] for b in net.branches], [Fraction(1, 2)])
        # 连通性把 C/L 支路也计入
        net2 = Network(["a", "b"])
        net2.add_capacitor("a", "b", 1)
        self.assertEqual(net2.components(), [("a", "b")])


if __name__ == "__main__":
    unittest.main()
