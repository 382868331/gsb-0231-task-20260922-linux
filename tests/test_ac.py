"""R/L/C 交流相量分析（修正结点方程 + Schur 消元）测试。

期望值来源彼此独立：
* 单 RC、串/并 RLC、纯电感/电容支路的**解析公式**（精确分数，assertEqual）；
* ``tests/reference`` 中用内置 ``complex`` **独立拼装 MNA、独立高斯消元**
  的全图参考（逐列激励法，不做消元），用于确定性小图枚举；
* 手算的支路电流/功率符号与数值。
被测核心不参与任何期望值的计算。
"""

from __future__ import annotations

import math
import unittest
from fractions import Fraction

from resistor_elim import (
    CapacitanceError,
    FrequencyError,
    InductanceError,
    Network,
    NetworkError,
    NodeError,
    PortError,
    PotentialError,
    QComplex,
    SingularSystemError,
    ac_admittance,
    schur_eliminate,
)
from reference import (
    RefACNetwork,
    cmath_is_close,
    parallel_admittance,
    rc_analytic,
    series_impedance_admittance,
)

F = Fraction


def qc(re, im=0):
    return QComplex(F(re), F(im))


def qz(z: QComplex) -> complex:
    """精确 QComplex → 内置 complex（仅用于与浮点参考比较）。"""
    return complex(float(z.re), float(z.im))


class TestQComplexArithmetic(unittest.TestCase):
    def test_basic_operations_exact(self):
        a = QComplex(F(1, 2), F(1, 3))
        b = QComplex(F(1, 4), F(1, 5))
        self.assertEqual(a + b, QComplex(F(3, 4), F(8, 15)))
        self.assertEqual(
            a * b,
            QComplex(F(1, 8) - F(1, 15), F(1, 10) + F(1, 12)),
        )
        self.assertEqual((a * b) / b, a)
        self.assertEqual(a / a, QComplex(1))
        self.assertEqual(QComplex(1) / QComplex(0, 1), QComplex(0, -1))

    def test_division_formula(self):
        # (1+2j)/(3+4j) = (11+2j)/25
        z = QComplex(1, 2) / QComplex(3, 4)
        self.assertEqual(z, QComplex(F(11, 25), F(2, 25)))

    def test_rejects_non_finite(self):
        with self.assertRaises(ValueError):
            QComplex(float("nan"))
        with self.assertRaises(ValueError):
            QComplex(0, float("inf"))
        with self.assertRaises(TypeError):
            QComplex(True)

    def test_float_construction_is_exact_binary(self):
        self.assertEqual(QComplex(0.5), QComplex(F(1, 2)))
        self.assertEqual(QComplex(complex(0.25, -0.5)),
                         QComplex(F(1, 4), F(-1, 2)))


class TestSingleBranchAnalytic(unittest.TestCase):
    """最简电路：解析值精确对照（assertEqual，无容差）。"""

    def test_single_resistor_at_ac(self):
        net = Network(["p", "q"])
        net.add_resistor("p", "q", 4)
        r = ac_admittance(net, ["p", "q"], 3)
        self.assertEqual(r.y, [[qc(F(1, 4)), qc(-F(1, 4))],
                               [qc(-F(1, 4)), qc(F(1, 4))]])

    def test_single_capacitor(self):
        # Y = jωC
        net = Network(["p", "q"])
        net.add_capacitor("p", "q", F(2, 7))
        r = ac_admittance(net, ["p", "q"], F(3, 5))
        y = QComplex(0, F(3, 5) * F(2, 7))
        self.assertEqual(r.y[0][0], y)
        self.assertEqual(r.y[0][1], -y)

    def test_single_inductor(self):
        # Y = 1/(jωL) = -j/(ωL)
        net = Network(["p", "q"])
        net.add_inductor("p", "q", F(3, 2))
        r = ac_admittance(net, ["p", "q"], F(4, 9))
        y = QComplex(0, -1 / (F(4, 9) * F(3, 2)))
        self.assertEqual(r.y[0][0], y)
        self.assertEqual(r.y[1][0], -y)

    def test_rc_parallel_analytic_exact(self):
        # 单 RC（并联）：Y = 1/R + jωC
        net = Network(["p", "q"])
        net.add_resistor("p", "q", 2)
        net.add_capacitor("p", "q", 3)
        w = 5
        r = ac_admittance(net, ["p", "q"], w)
        expect = rc_analytic(2, 3, w)  # complex 参考，仅做换算核对
        self.assertEqual(r.y[0][0], QComplex(F(1, 2), F(15)))
        self.assertTrue(cmath_is_close(qz(r.y[0][0]), expect))

    def test_rc_series_analytic_exact(self):
        # R=2 与 C=3 串联，ω=5：Z = 2 - j/15；Y = (450 + j15)/901
        net = Network(["p", "a", "q"])
        net.add_resistor("p", "a", 2)
        net.add_capacitor("a", "q", 3)
        r = ac_admittance(net, ["p", "q"], 5)
        y_expect = QComplex(F(450, 901), F(15, 901))
        self.assertEqual(r.y[0][0], y_expect)
        self.assertEqual(r.y[0][1], -y_expect)
        # 与串联解析式（complex）一致
        ref = series_impedance_admittance([("r", 2), ("c", 3)], 5)
        self.assertTrue(cmath_is_close(qz(r.y[0][0]), ref[0][0]))

    def test_series_rlc_analytic_exact(self):
        # R=2, L=3, C=1/6, ω=5：Z = 2 + j·69/5；Y = (50 - j345)/4861
        net = Network(["p", "a", "b", "q"])
        net.add_resistor("p", "a", 2)
        net.add_inductor("a", "b", 3)
        net.add_capacitor("b", "q", F(1, 6))
        r = ac_admittance(net, ["p", "q"], 5)
        y_expect = QComplex(F(50, 4861), F(-345, 4861))
        self.assertEqual(r.y[0][0], y_expect)
        ref = series_impedance_admittance(
            [("r", 2), ("l", 3), ("c", F(1, 6))], 5
        )
        self.assertTrue(cmath_is_close(qz(r.y[0][0]), ref[0][0]))

    def test_parallel_rlc_analytic_exact(self):
        # R=2, L=3, C=1/6 并联，ω=5：Y = 1/2 + j·(5/6 - 1/15) = 1/2 + j·23/30
        net = Network(["p", "q"])
        net.add_resistor("p", "q", 2)
        net.add_inductor("p", "q", 3)
        net.add_capacitor("p", "q", F(1, 6))
        r = ac_admittance(net, ["p", "q"], 5)
        self.assertEqual(r.y[0][0], QComplex(F(1, 2), F(23, 30)))
        ref = parallel_admittance(
            [("r", 2), ("l", 3), ("c", F(1, 6))], 5
        )
        self.assertTrue(cmath_is_close(qz(r.y[0][0]), ref[0][0]))

    def test_parallel_inductors_analytic(self):
        # L=3 与 L=6 并联 ⇔ 等效 L=2：Y = -j/10 (ω=5)
        net = Network(["p", "q"])
        net.add_inductor("p", "q", 3)
        net.add_inductor("q", "p", 6)
        r = ac_admittance(net, ["p", "q"], 5)
        self.assertEqual(r.y[0][0], QComplex(0, F(-1, 10)))

    def test_parallel_capacitors_add(self):
        # C=1/3 与 C=2/3 并联 ⇔ C=1：Y = jω (ω=4)
        net = Network(["p", "q"])
        net.add_capacitor("p", "q", F(1, 3))
        net.add_capacitor("q", "p", F(2, 3))
        r = ac_admittance(net, ["p", "q"], 4)
        self.assertEqual(r.y[0][0], QComplex(0, 4))

    def test_schur_record_rebuilds_port_y(self):
        # Y_port = A_PP - A_PI D^-1 A_IP，用记录内因子独立重建
        from resistor_elim.cx import CxMat

        net = Network(["p", "q", "a", "b"])
        net.add_resistor("p", "a", 2)
        net.add_capacitor("a", "q", 3)
        net.add_inductor("a", "b", F(1, 2))
        net.add_resistor("b", "q", 4)
        r = ac_admittance(net, ["p", "q"], F(3, 2))
        rec = r.ported_components[0].record
        rebuilt = CxMat.matmul(CxMat.matmul(rec.a_pi, rec.d_inverse), rec.a_ip)
        expect = [
            [rec.a_pp[i][j] - rebuilt[i][j] for j in range(2)]
            for i in range(2)
        ]
        self.assertEqual(expect, r.y)

    def test_single_port_dangling_branch_zero_input_admittance(self):
        # 单端口 p 接一条到悬空内部节点 a 的 RLC 支路：无回路 ⇒ 输入导纳 0
        net = Network(["p", "a"])
        net.add_resistor("p", "a", 2)
        net.add_capacitor("p", "a", 3)
        r = ac_admittance(net, ["p"], 7)
        self.assertEqual(r.y, [[QComplex()]])
        sol = r.solve([1])
        self.assertEqual(sol.port_currents["p"], QComplex())
        self.assertEqual(sol.potentials["a"], QComplex(1))  # 无电流 ⇒ 等势随动

    def test_dangling_pure_inductor(self):
        # 单端口经纯电感到悬空内部节点：i_L=0、内部等势随动（不奇异）
        net = Network(["p", "a"])
        net.add_inductor("p", "a", 3)
        r = ac_admittance(net, ["p"], 5)
        self.assertEqual(r.y, [[QComplex()]])
        sol = r.solve([1])
        self.assertEqual(sol.inductor_currents[0], QComplex())
        self.assertEqual(sol.potentials["a"], QComplex(1))

    def test_internal_parallel_lc_resonance_is_open_not_singular(self):
        # R - (L//C) - R 结构：内部并联谐振点导纳恰为 0 ⇒ 串联开路，有限
        net = Network(["p", "q", "a", "b"])
        net.add_resistor("p", "a", 1)
        net.add_resistor("b", "q", 1)
        net.add_inductor("a", "b", 1)
        net.add_capacitor("a", "b", 1)
        r = ac_admittance(net, ["p", "q"], 1)
        self.assertEqual(r.y[0][0], QComplex(0))  # 开路 ⇒ 零电流，非奇异
        # ω=2：Z_ab = 1/(j·3/2) = -j·2/3；总 Z = 2 - j·2/3；Y = 9/20 + j·3/20
        r2 = ac_admittance(net, ["p", "q"], 2)
        self.assertEqual(r2.y[0][0], QComplex(F(9, 20), F(3, 20)))

    def test_30_node_rlc_chain(self):
        # 接口上限：29 条交替 R/L/C 串联支路，与独立全图 MNA 参考比较
        n = 30
        net = Network(range(n))
        ref_branches = []
        for i in range(n - 1):
            kind = i % 3
            if kind == 0:
                net.add_resistor(i, i + 1, 1)
                ref_branches.append(("r", i, i + 1, F(1)))
            elif kind == 1:
                net.add_inductor(i, i + 1, 1)
                ref_branches.append(("l", i, i + 1, F(1)))
            else:
                net.add_capacitor(i, i + 1, 1)
                ref_branches.append(("c", i, i + 1, F(1)))
        omega = F(2)
        r = ac_admittance(net, [0, n - 1], omega)
        ref = RefACNetwork(list(range(n)), ref_branches)
        ref_y = ref.port_admittance([0, n - 1], float(omega))
        self.assertTrue(cmath_is_close(qz(r.y[0][0]), ref_y[0][0], tol=1e-9))
        # 串联解析式：Z = 10·(1 + j2) + 9·(1 + 1/(j2)) + ... 直接逐元件求和
        z = 0j
        for i in range(n - 1):
            if i % 3 == 0:
                z += 1.0
            elif i % 3 == 1:
                z += 2j
            else:
                z += 1.0 / (2j)
        self.assertTrue(cmath_is_close(qz(r.y[0][0]), 1.0 / z, tol=1e-9))


class TestPureResistorMatchesDC(unittest.TestCase):
    def test_ac_reduces_to_dc_dtn(self):
        # 纯电阻网络：交流导纳矩阵必须与直流 Schur DtN 逐值相等（虚部为 0）
        net = Network(["p", "q", "a", "b"])
        net.add_resistor("p", "a", 2)
        net.add_resistor("a", "q", 3)
        net.add_resistor("p", "b", 4)
        net.add_resistor("b", "q", 5)
        net.add_resistor("a", "b", 6)
        dtn = schur_eliminate(net, ["p", "q"]).dtn
        y = ac_admittance(net, ["p", "q"], F(123, 7)).y
        for i in range(2):
            for j in range(2):
                self.assertEqual(y[i][j], QComplex(dtn[i][j], F(0)))

    def test_dc_rejects_reactive_network(self):
        net = Network(["p", "q"])
        net.add_capacitor("p", "q", 1)
        with self.assertRaises(NetworkError):
            schur_eliminate(net, ["p", "q"])
        net2 = Network(["p", "q"])
        net2.add_inductor("p", "q", 1)
        with self.assertRaises(NetworkError):
            schur_eliminate(net2, ["p", "q"])


class TestSymmetryAndFrequencyRelations(unittest.TestCase):
    def test_y_is_complex_symmetric_and_zero_rowsum(self):
        net = Network(["p", "q", "a", "b"])
        net.add_resistor("p", "a", 1)
        net.add_capacitor("a", "q", 2)
        net.add_inductor("p", "b", 3)
        net.add_resistor("b", "q", F(1, 4))
        net.add_capacitor("a", "b", F(2, 3))
        y = ac_admittance(net, ["p", "q"], F(2, 5)).y
        self.assertEqual(y[0][1], y[1][0])          # 复对称（转置）
        self.assertEqual(y[0][0] + y[0][1], QComplex())
        self.assertEqual(y[1][0] + y[1][1], QComplex())

    def test_negative_frequency_is_conjugate(self):
        net = Network(["p", "q", "a"])
        net.add_resistor("p", "a", 2)
        net.add_capacitor("a", "q", F(3, 2))
        net.add_inductor("p", "q", 5)
        yp = ac_admittance(net, ["p", "q"], F(4, 7)).y
        ym = ac_admittance(net, ["p", "q"], F(-4, 7)).y
        for i in range(2):
            for j in range(2):
                self.assertEqual(ym[i][j], yp[i][j].conj())


class TestSolutionRecovery(unittest.TestCase):
    def test_series_rlc_branch_currents_and_kvl(self):
        net = Network(["p", "a", "b", "q"])
        net.add_resistor("p", "a", 2)
        net.add_inductor("a", "b", 3)
        net.add_capacitor("b", "q", F(1, 6))
        r = ac_admittance(net, ["p", "q"], 5)
        sol = r.solve({"p": 1, "q": 0})
        i_total = r.y[0][0]  # V=1 ⇒ 端口电流 = Y11
        # 串联：所有支路电流相同
        self.assertEqual(sol.branch_currents[0], i_total)
        self.assertEqual(sol.inductor_currents[0], i_total)
        self.assertEqual(sol.capacitor_currents[0], i_total)
        # 电感电压 jωL·i = V(a)-V(b)
        vl = QComplex(0, 15) * i_total
        self.assertEqual(sol.potentials["a"] - sol.potentials["b"], vl)
        # 电容电压 i/(jωC) = V(b)-V(q)=V(b)
        vc = i_total / QComplex(0, F(5, 6))
        self.assertEqual(sol.potentials["b"], vc)
        # 回路电压和 = 激励 1
        vr = QComplex(2) * i_total
        self.assertEqual(vr + vl + vc, QComplex(1))

    def test_average_power_balance_exact(self):
        # 串联 RC，峰值相量 V=1：有功全部耗在电阻上（精确分数）
        net = Network(["p", "a", "q"])
        net.add_resistor("p", "a", 2)
        net.add_capacitor("a", "q", 3)
        r = ac_admittance(net, ["p", "q"], 5)
        sol = r.solve([1, 0])
        self.assertEqual(sol.average_port_power, F(225, 901))
        self.assertEqual(sol.average_dissipated_power, F(225, 901))
        self.assertEqual(sol.average_port_power, sol.average_dissipated_power)
        # 无功功率非零（电容），端口视在功率为复数
        self.assertNotEqual(sol.port_apparent_power.im, 0)

    def test_reactive_only_network_zero_dissipation(self):
        # 纯 L//C：无损耗，有功为 0（并联谐振点 Y=0 时电流也为 0）
        net = Network(["p", "q"])
        net.add_inductor("p", "q", 1)
        net.add_capacitor("p", "q", 1)
        r = ac_admittance(net, ["p", "q"], 1)
        self.assertEqual(r.y[0][0], QComplex(0))  # 理想并联谐振 ⇒ 开路
        sol = r.solve([1, 0])
        self.assertEqual(sol.port_currents["p"], QComplex())
        self.assertEqual(sol.average_dissipated_power, 0)
        # 非谐振点仍无耗
        sol2 = ac_admittance(net, ["p", "q"], 2).solve([1, 0])
        self.assertEqual(sol2.average_dissipated_power, 0)
        self.assertEqual(sol2.average_port_power, 0)

    def test_phasor_sequence_and_complex_input(self):
        net = Network(["p", "q"])
        net.add_resistor("p", "q", 2)
        r = ac_admittance(net, ["p", "q"], 5)
        sol = r.solve([complex(1, 1), 0])
        self.assertEqual(sol.port_currents["p"],
                         QComplex(F(1, 2), F(1, 2)))
        sol2 = r.solve([QComplex(1, 1), QComplex()])
        self.assertEqual(sol2.port_currents["p"], sol.port_currents["p"])

    def test_float_omega_exact_binary(self):
        net = Network(["p", "q"])
        net.add_capacitor("p", "q", 4)
        r = ac_admittance(net, ["p", "q"], 0.5)  # = 1/2 精确
        self.assertEqual(r.y[0][0], QComplex(0, 2))


class TestFloatingACComponents(unittest.TestCase):
    def test_floating_component_marked(self):
        net = Network(["p", "q", "x", "y", "z"])
        net.add_resistor("p", "q", 1)
        net.add_capacitor("x", "y", 2)
        net.add_inductor("y", "x", 3)
        r = ac_admittance(net, ["p", "q"], 7)
        self.assertIn(("x", "y"), r.floating_components)
        self.assertIn(("z",), r.floating_components)
        self.assertEqual(r.y[0][0], QComplex(1))
        sol = r.solve([1, 0])
        self.assertNotIn("x", sol.potentials)
        self.assertEqual(sol.capacitor_currents[0], None)
        self.assertEqual(sol.inductor_currents[0], None)
        self.assertEqual(sol.average_port_power, sol.average_dissipated_power)

    def test_components_use_reactive_edges(self):
        # 仅由电容/电感连接的节点仍属于同一组件（直流拉普拉斯看不到它们）
        net = Network(["p", "a"])
        net.add_capacitor("p", "a", 1)
        self.assertEqual(net.components(), [("p", "a")])
        r = ac_admittance(net, ["p"], 3)
        # 单端口：电容到“内部”节点 a，a 无其他连接 ⇒ 内部自导 jωC 可逆，
        # 端口看进去为 0（a 浮置随动，无电流）
        self.assertEqual(r.y, [[QComplex()]])

    def test_two_independent_ported_components(self):
        net = Network(["p", "q", "r", "s"])
        net.add_resistor("p", "q", 2)
        net.add_inductor("r", "s", 4)
        r = ac_admittance(net, ["p", "q", "r", "s"], 5)
        self.assertEqual(len(r.ported_components), 2)
        self.assertEqual(r.y[0][2], QComplex())
        self.assertEqual(r.y[2][2], QComplex(0, F(-1, 20)))


class TestSingularSystems(unittest.TestCase):
    def test_ideal_lc_series_resonance_is_reported(self):
        # L=C=1 串联，ω=1 = 1/√(LC)：零阻抗支路 ⇒ 内部块奇异
        net = Network(["p", "a", "q"])
        net.add_inductor("p", "a", 1)
        net.add_capacitor("a", "q", 1)
        with self.assertRaises(SingularSystemError):
            ac_admittance(net, ["p", "q"], 1)

    def test_resonance_goes_away_with_resistor(self):
        net = Network(["p", "a", "b", "q"])
        net.add_resistor("p", "a", 1)
        net.add_inductor("a", "b", 1)
        net.add_capacitor("b", "q", 1)
        r = ac_admittance(net, ["p", "q"], 1)
        self.assertEqual(r.y[0][0], QComplex(1))

    def test_off_resonance_finite(self):
        net = Network(["p", "a", "q"])
        net.add_inductor("p", "a", 1)
        net.add_capacitor("a", "q", 1)
        r = ac_admittance(net, ["p", "q"], 2)
        self.assertEqual(r.y[0][0], QComplex(0, F(-2, 3)))

    def test_network_unchanged_after_singularity(self):
        net = Network(["p", "a", "q"])
        net.add_inductor("p", "a", 1)
        net.add_capacitor("a", "q", 1)
        with self.assertRaises(SingularSystemError):
            ac_admittance(net, ["p", "q"], 1)
        # 网络对象仍可在其他频率正常求解
        r = ac_admittance(net, ["p", "q"], 3)
        self.assertEqual(r.y[0][0], QComplex(0, F(-3, 8)))


class TestFrequencyValidation(unittest.TestCase):
    def _net(self):
        net = Network(["p", "q"])
        net.add_resistor("p", "q", 1)
        return net

    def test_zero_frequency_rejected(self):
        with self.assertRaises(FrequencyError) as ctx:
            ac_admittance(self._net(), ["p", "q"], 0)
        self.assertIn("零", str(ctx.exception))

    def test_non_finite_frequency_rejected(self):
        with self.assertRaises(FrequencyError):
            ac_admittance(self._net(), ["p", "q"], float("nan"))
        with self.assertRaises(FrequencyError):
            ac_admittance(self._net(), ["p", "q"], float("inf"))
        with self.assertRaises(FrequencyError):
            ac_admittance(self._net(), ["p", "q"], float("-inf"))

    def test_bad_frequency_types(self):
        with self.assertRaises(TypeError):
            ac_admittance(self._net(), ["p", "q"], True)
        with self.assertRaises(FrequencyError):
            ac_admittance(self._net(), ["p", "q"], 1 + 2j)
        with self.assertRaises(FrequencyError):
            ac_admittance(self._net(), ["p", "q"], "5")

    def test_ports_still_validated(self):
        net = self._net()
        with self.assertRaises(PortError):
            ac_admittance(net, [], 1)
        with self.assertRaises(NodeError):
            ac_admittance(net, ["p", "z"], 1)
        with self.assertRaises(PortError):
            ac_admittance(net, ["p", "p"], 1)

    def test_isolated_port_zero_row(self):
        net = Network(["p", "q", "z"])
        net.add_resistor("p", "q", 1)
        r = ac_admittance(net, ["p", "q", "z"], 2)
        self.assertEqual(r.y[2], [QComplex(), QComplex(), QComplex()])


class TestComponentValueValidation(unittest.TestCase):
    def _net(self):
        return Network(["a", "b", "c"])

    def test_non_positive_values(self):
        net = self._net()
        with self.assertRaises(CapacitanceError):
            net.add_capacitor("a", "b", 0)
        with self.assertRaises(CapacitanceError):
            net.add_capacitor("a", "b", F(-1, 2))
        with self.assertRaises(InductanceError):
            net.add_inductor("a", "b", 0)
        with self.assertRaises(InductanceError):
            net.add_inductor("a", "b", -3)

    def test_float_values_rejected(self):
        net = self._net()
        with self.assertRaises(CapacitanceError):
            net.add_capacitor("a", "b", 1.0)
        with self.assertRaises(InductanceError):
            net.add_inductor("a", "b", 2.0)

    def test_nan_infinity_messages(self):
        net = self._net()
        with self.assertRaises(CapacitanceError) as ctx:
            net.add_capacitor("a", "b", float("nan"))
        self.assertIn("NaN", str(ctx.exception))
        with self.assertRaises(InductanceError) as ctx:
            net.add_inductor("a", "b", float("inf"))
        self.assertIn("Infinity", str(ctx.exception))

    def test_bool_and_bad_types(self):
        net = self._net()
        with self.assertRaises(TypeError):
            net.add_capacitor("a", "b", True)
        with self.assertRaises(TypeError):
            net.add_inductor("a", "b", False)
        with self.assertRaises(CapacitanceError):
            net.add_capacitor("a", "b", "1e-9")
        with self.assertRaises(InductanceError):
            net.add_inductor("a", "b", 1 + 0j)

    def test_unknown_node_self_loop_and_atomicity(self):
        net = self._net()
        with self.assertRaises(NodeError) as ctx:
            net.add_capacitor("a", "z", 1)
        self.assertEqual(ctx.exception.node, "z")
        with self.assertRaises(CapacitanceError):
            net.add_capacitor("a", "a", 1)
        with self.assertRaises(InductanceError):
            net.add_inductor("b", "b", 1)
        net.add_capacitor("a", "b", 2)
        with self.assertRaises(NetworkError):
            net.add_capacitor("a", "b", -2)
        with self.assertRaises(NetworkError):
            net.add_inductor("a", "c", 0)
        # 失败不留部分变更
        self.assertEqual(len(net.capacitor_branches), 1)
        self.assertEqual(len(net.inductor_branches), 0)
        # 直流状态未被电容污染
        self.assertEqual(net.laplacian()[0][0], 0)


class TestPhasorValidation(unittest.TestCase):
    def test_wrong_count_and_keys(self):
        net = Network(["p", "q"])
        net.add_resistor("p", "q", 1)
        r = ac_admittance(net, ["p", "q"], 2)
        with self.assertRaises(PotentialError):
            r.solve([1])
        with self.assertRaises(PotentialError):
            r.solve({"p": 1})
        with self.assertRaises(PotentialError):
            r.solve({"p": 1, "q": 0, "x": 1})

    def test_non_finite_phasors(self):
        net = Network(["p", "q"])
        net.add_resistor("p", "q", 1)
        r = ac_admittance(net, ["p", "q"], 2)
        with self.assertRaises(PotentialError):
            r.solve([float("nan"), 0])
        with self.assertRaises(PotentialError):
            r.solve([complex(0, float("inf")), 0])
        with self.assertRaises(PotentialError):
            r.solve([True, 0])
        with self.assertRaises(PotentialError):
            r.solve(["1", 0])

    def test_solve_does_not_mutate_result(self):
        net = Network(["p", "q"])
        net.add_resistor("p", "q", 1)
        r = ac_admittance(net, ["p", "q"], 2)
        s1 = r.solve([1, 0])
        s2 = r.solve([1, 0])
        self.assertEqual(s1.port_currents["p"], s2.port_currents["p"])


class TestIndependentReferenceEnumeration(unittest.TestCase):
    """小规模确定性枚举：与独立 complex MNA 参考逐值比较导纳与支路电流。"""

    @staticmethod
    def _lcg_stream(seed):
        state = seed

        def nxt():
            nonlocal state
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            return state

        return nxt

    def _build_pair(self, n, rng):
        labels = list(range(n))
        net = Network(labels)
        ref_branches = []
        for i in range(n):
            for j in range(i + 1, n):
                if rng() % 4 == 0:
                    continue
                kind = rng() % 3
                val = F(rng() % 4 + 1, rng() % 3 + 1)
                if kind == 0:
                    net.add_resistor(i, j, val)
                    ref_branches.append(("r", i, j, val))
                elif kind == 1:
                    net.add_capacitor(i, j, val)
                    ref_branches.append(("c", i, j, val))
                else:
                    net.add_inductor(i, j, val)
                    ref_branches.append(("l", i, j, val))
        # 保证端口 0/1 之间有连接：强制一条电阻
        net.add_resistor(0, 1, 1)
        ref_branches.append(("r", 0, 1, F(1)))
        return net, ref_branches

    def test_enumerated_networks_against_reference(self):
        rng = self._lcg_stream(987654321)
        computed = 0
        for trial in range(24):
            n = 3 + rng() % 4
            net, ref_branches = self._build_pair(n, rng)
            ports = [0, 1, n - 1]
            omega = F(rng() % 5 + 1, rng() % 3 + 1)  # 避开可能的精确谐振
            try:
                res = ac_admittance(net, ports, omega)
            except SingularSystemError:
                # 若精确实现判定奇异（如精确谐振），跳过该组合
                continue
            computed += 1
            ref = RefACNetwork(list(range(n)), ref_branches)
            ref_y = ref.port_admittance(ports, float(omega))
            for i in range(3):
                for j in range(3):
                    self.assertTrue(
                        cmath_is_close(qz(res.y[i][j]), ref_y[i][j], tol=1e-9),
                        f"trial {trial} Y[{i}][{j}]: {res.y[i][j]} vs {ref_y[i][j]}",
                    )
            # 随机端口相量，比较节点电压与支路电流
            vp = {p: complex(rng() % 5 - 2, rng() % 5 - 2) for p in ports}
            vp_q = {p: QComplex(F(vp[p].real), F(vp[p].imag)) for p in ports}
            sol = res.solve(vp_q)
            ref_v, ref_branch_i = ref.solve(ports, vp, float(omega))
            for node, zv in ref_v.items():
                self.assertTrue(
                    cmath_is_close(qz(sol.potentials[node]), zv, tol=1e-9),
                    f"trial {trial} node {node}",
                )
            # 参考支路按插入顺序交错；被测按 R/C/L 三类分别对齐，用计数器取
            counters = {"r": 0, "c": 0, "l": 0}
            lists = {
                "r": sol.branch_currents,
                "c": sol.capacitor_currents,
                "l": sol.inductor_currents,
            }
            for (t, _u, _v, _val), want in zip(ref_branches, ref_branch_i):
                got = lists[t][counters[t]]
                counters[t] += 1
                if want is None:
                    self.assertIsNone(got)
                    continue
                self.assertIsNotNone(got)
                self.assertTrue(
                    cmath_is_close(qz(got), want, tol=1e-9),
                    f"trial {trial} {t}-branch current {got} vs {want}",
                )
            self.assertEqual(
                (counters["r"], counters["c"], counters["l"]),
                (len(net.branches), len(net.capacitor_branches),
                 len(net.inductor_branches)),
            )

            # 复对称、行和 0、功率守恒
            for i in range(3):
                self.assertEqual(sum(res.y[i]), QComplex())
            self.assertTrue(
                math.isclose(
                    float(sol.average_port_power),
                    float(sol.average_dissipated_power),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
            self.assertGreaterEqual(sol.average_dissipated_power, 0)
        self.assertGreaterEqual(computed, 18)


if __name__ == "__main__":
    unittest.main()
