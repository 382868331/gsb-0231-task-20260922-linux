"""期望首次到达步数（Poisson 方程）测试。

期望值来源：
* 路径图、星形图上手算的分数值；
* tests/reference 独立编写的有限步概率递推（生存概率尾和）给出单调
  递增的下界，并随步数增长逼近精确期望；
* 单目标、无停留权重时，通勤时间 = 2 * 总电导 * 有效电阻（后者由
  schur_eliminate 的 DtN 独立得到）。
参考递推直接由支路表与停留权重实现，不调用 hitting_times 的任何算法。
"""

from __future__ import annotations

import math
import unittest
from fractions import Fraction

from resistor_elim import (
    ConductanceError,
    Network,
    StayWeightError,
    TargetError,
    hitting_times,
    schur_eliminate,
)


def finite_step_lower_bounds(net: Network, targets, start, horizons):
    """独立有限步递推：E[min(T, k)] = sum_{j=0..k-1} P(T > j)。

    只对与目标连通组件内的暂态节点做概率质量递推（停留权重作为自环
    保留），不导入也不调用被测 hitting 模块。
    """
    target_set = set(targets)
    stay = net.stay_weights()
    # 独立 BFS 找含 start 的连通组件
    seen = {start}
    stack = [start]
    while stack:
        x = stack.pop()
        for y in net.neighbors(x):
            if y not in seen:
                seen.add(y)
                stack.append(y)
    states = [x for x in seen if x not in target_set]
    index = {x: i for i, x in enumerate(states)}
    denom = {x: stay[x] + sum(net.neighbors(x).values(), Fraction(0)) for x in states}
    probs = {start: Fraction(1)} if start in index else {}
    horizons = sorted(horizons)
    bounds = {}
    tail_sum = Fraction(0)
    for j in range(max(horizons) + 1):
        if j in horizons:
            bounds[j] = tail_sum  # E[min(T, j)]
        survival = sum(probs.values(), Fraction(0))  # P(T > j)
        tail_sum += survival
        nxt = {}
        for x, mass in probs.items():
            d = denom[x]
            add = lambda y, p: nxt.__setitem__(y, nxt.get(y, Fraction(0)) + mass * p)
            add(x, stay[x] / d)
            for y, g in net.neighbors(x).items():
                if y in index:
                    add(y, g / d)
                # 落到目标的质量被吸收，不再贡献生存概率
        probs = nxt
    return bounds


class TestHittingTimesHandCalculated(unittest.TestCase):
    def test_path_three_nodes(self):
        # 1 --1-- 2 --1-- 3，目标 {3}：h1=4, h2=3（手算）。
        net = Network([1, 2, 3])
        net.add_conductor(1, 2, 1)
        net.add_conductor(2, 3, 1)
        res = hitting_times(net, [3])
        self.assertEqual(res.expectations[3], 0)
        self.assertEqual(res.expectations[2], 3)
        self.assertEqual(res.expectations[1], 4)
        self.assertEqual(res.unreachable, ())
        self.assertEqual(set(res.reachable_transient), {1, 2})
        for node in (1, 2):
            self.assertEqual(res.residuals[node], Fraction(0))
        self.assertIsNone(res.residuals[3])

    def test_path_with_conductances_fractions(self):
        # a --g=2-- b --g=1-- t：
        # h_a = 1 + h_b；h_b = 1 + (2 h_a + 0)/3 -> h_a=6, h_b=5。
        net = Network(["a", "b", "t"])
        net.add_conductor("a", "b", 2)
        net.add_conductor("b", "t", 1)
        res = hitting_times(net, ["t"])
        self.assertEqual(res.expectations["a"], 6)
        self.assertEqual(res.expectations["b"], 5)

    def test_star_two_leaves(self):
        # 中心 c，叶子 a,b（电导均 1），目标 {a}：h_c=3, h_b=4。
        net = Network(["c", "a", "b"])
        net.add_conductor("c", "a", 1)
        net.add_conductor("c", "b", 1)
        res = hitting_times(net, ["a"])
        self.assertEqual(res.expectations["c"], 3)
        self.assertEqual(res.expectations["b"], 4)

    def test_star_two_targets(self):
        # 中心 c 等概率走到两个吸收叶子：h_c = 1。
        net = Network(["c", "a", "b"])
        net.add_conductor("c", "a", 1)
        net.add_conductor("c", "b", 1)
        res = hitting_times(net, ["a", "b"])
        self.assertEqual(res.expectations["c"], 1)

    def test_parallel_conductances_combine(self):
        net = Network(["s", "t"])
        net.add_conductor("s", "t", 1)
        net.add_conductor("s", "t", Fraction(3, 2))
        res = hitting_times(net, ["t"])
        self.assertEqual(res.expectations["s"], 1)


class TestStayWeights(unittest.TestCase):
    def test_single_edge_stay_weight(self):
        # g=1，s 处停留权重 w：停留概率 w/(w+1)，h_s = 1 + w。
        net = Network(["s", "t"])
        net.add_conductor("s", "t", 1)
        net.set_stay_weight("s", Fraction(3, 2))
        res = hitting_times(net, ["t"])
        self.assertEqual(res.expectations["s"], Fraction(5, 2))
        self.assertEqual(res.residuals["s"], 0)

    def test_zero_stay_default_and_reset(self):
        net = Network(["s", "t"])
        net.add_conductor("s", "t", 1)
        self.assertEqual(net.stay_weight("s"), 0)
        net.set_stay_weight("s", 2)
        net.set_stay_weight("s", 0)
        self.assertEqual(hitting_times(net, ["t"]).expectations["s"], 1)

    def test_stay_does_not_change_laplacian_or_dtn(self):
        def build():
            n = Network(["p", "m", "q"])
            n.add_resistor("p", "m", 2)
            n.add_resistor("m", "q", 6)
            return n
        ref = build()
        net = build()
        net.set_stay_weight("m", 5)
        net.set_stay_weight("p", Fraction(1, 3))
        self.assertEqual(net.laplacian(), ref.laplacian())
        self.assertEqual(
            schur_eliminate(net, ["p", "q"]).dtn,
            schur_eliminate(ref, ["p", "q"]).dtn,
        )

    def test_self_loop_resistor_still_rejected(self):
        net = Network(["a"])
        with self.assertRaises(ConductanceError):
            net.add_conductor("a", "a", 1)
        # 停留权重是独立机制，合法
        net.set_stay_weight("a", 1)

    def test_bad_stay_weights(self):
        net = Network(["a", "b"])
        with self.assertRaises(StayWeightError):
            net.set_stay_weight("a", -1)
        with self.assertRaises(StayWeightError):
            net.set_stay_weight("a", 0.5)
        with self.assertRaises(TypeError):
            net.set_stay_weight("a", True)
        with self.assertRaises(Exception):
            net.set_stay_weight("z", 1)


class TestDisconnectedIsInfinity(unittest.TestCase):
    def test_unreachable_component(self):
        net = Network(["a", "b", "x", "y", "iso"])
        net.add_conductor("a", "b", 1)
        net.add_conductor("x", "y", 2)
        res = hitting_times(net, ["a"])
        self.assertEqual(res.expectations["a"], 0)
        self.assertEqual(res.expectations["b"], 1)
        for node in ("x", "y", "iso"):
            self.assertTrue(math.isinf(res.expectations[node]))
            self.assertIsNone(res.residuals[node])
        self.assertEqual(set(res.unreachable), {"x", "y", "iso"})

    def test_isolated_target_is_fine(self):
        net = Network(["t", "x"])
        res = hitting_times(net, ["t"])
        self.assertEqual(res.expectations["t"], 0)
        self.assertTrue(math.isinf(res.expectations["x"]))


class TestFiniteStepBounds(unittest.TestCase):
    def test_bounds_monotone_approach_expectation(self):
        net = Network([1, 2, 3])
        net.add_conductor(1, 2, 1)
        net.add_conductor(2, 3, 1)
        res = hitting_times(net, [3])
        horizons = [0, 1, 5, 20, 60, 120]
        for start in (1, 2):
            bounds = finite_step_lower_bounds(net, [3], start, horizons)
            h = res.expectations[start]
            prev = None
            for k in horizons:
                lb = bounds[k]
                self.assertLessEqual(lb, h)
                if prev is not None:
                    self.assertGreaterEqual(lb, prev)
                prev = lb
            # 尾和下界随步数增长逼近精确期望（误差单调缩小到可忽略）
            self.assertLess(float(h - bounds[120]), 1e-6)
            # 有限时间内严格为下界（仍有正的未吸收质量）
            self.assertLess(bounds[5], h)

    def test_bounds_with_stay_weight(self):
        net = Network(["s", "t"])
        net.add_conductor("s", "t", 1)
        net.set_stay_weight("s", 1)  # h_s = 2
        bounds = finite_step_lower_bounds(net, ["t"], "s", [0, 1, 10, 80])
        self.assertEqual(bounds[0], 0)
        self.assertEqual(bounds[1], 1)
        self.assertLess(bounds[10], 2)
        self.assertLess(2 - float(bounds[80]), 1e-9)


class TestCommuteTime(unittest.TestCase):
    def test_path_commute_equals_2GR(self):
        net = Network([1, 2, 3])
        net.add_conductor(1, 2, 1)
        net.add_conductor(2, 3, 1)
        h_to_3 = hitting_times(net, [3]).expectations
        h_to_1 = hitting_times(net, [1]).expectations
        commute = h_to_3[1] + h_to_1[3]
        red = schur_eliminate(net, [1, 3])
        g_eff = red.dtn[0][0]
        g_total = sum((g for _, _, g in net.branches), Fraction(0))
        self.assertEqual(g_total, 2)
        self.assertEqual(commute, 2 * g_total / g_eff)
        self.assertEqual(commute, 8)  # R_eff=2, G_total=2

    def test_star_commute_equals_2GR(self):
        net = Network(["c", "a", "b"])
        net.add_conductor("c", "a", 1)
        net.add_conductor("c", "b", 1)
        hab = hitting_times(net, ["b"]).expectations["a"]
        hba = hitting_times(net, ["a"]).expectations["b"]
        red = schur_eliminate(net, ["a", "b"])
        g_eff = red.dtn[0][0]  # 半电导
        g_total = Fraction(2)
        self.assertEqual(hab + hba, 2 * g_total / g_eff)
        self.assertEqual(hab, 4)
        self.assertEqual(hba, 4)


class TestTargetValidation(unittest.TestCase):
    def setUp(self):
        self.net = Network(["a", "b"])
        self.net.add_conductor("a", "b", 1)

    def test_empty_dup_unknown_bool(self):
        with self.assertRaises(TargetError):
            hitting_times(self.net, [])
        with self.assertRaises(TargetError):
            hitting_times(self.net, ["a", "a"])
        with self.assertRaises(Exception):
            hitting_times(self.net, ["z"])
        with self.assertRaises(TargetError):
            hitting_times(self.net, [True])

    def test_does_not_mutate_network(self):
        branches_before = self.net.branches
        stay_before = self.net.stay_weights()
        hitting_times(self.net, ["b"])
        self.assertEqual(self.net.branches, branches_before)
        self.assertEqual(self.net.stay_weights(), stay_before)


if __name__ == "__main__":
    unittest.main()
