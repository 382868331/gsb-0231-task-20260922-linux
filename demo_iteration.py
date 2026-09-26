"""本轮迭代演示：电导随机游走的期望首次到达时间。

运行：``python demo_iteration.py``。只使用标准库，无 sleep、无预录结果。
"""

from __future__ import annotations

from fractions import Fraction

from resistor_elim import (
    Network,
    StayWeightError,
    expected_hitting_times,
    schur_eliminate,
)


def independent_tail_bound(net: Network, target, start, horizon: int) -> Fraction:
    """独立有限步递推 sum_{t=0}^H P(T>t)，给出 E[T] 的严格下界。"""
    seen = {target}
    stack = [target]
    while stack:
        node = stack.pop()
        for neighbor in net.neighbors(node):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    active = [node for node in net.nodes if node != target and node in seen]
    alive = {node: Fraction(1) for node in active}
    total_bound = Fraction(0)
    for _ in range(horizon + 1):
        total_bound += alive[start]
        next_alive = {}
        for node in active:
            neighbors = net.neighbors(node)
            stay = net.stay_weights[net.index(node)]
            weight = sum(neighbors.values(), Fraction(0)) + stay
            next_alive[node] = (
                stay / weight * alive[node]
                + sum(
                    conductance / weight * alive[neighbor]
                    for neighbor, conductance in neighbors.items()
                    if neighbor != target
                )
            )
        alive = next_alive
    return total_bound


def main() -> None:
    print("=" * 70)
    print("随机游走：正电导归一化为转移概率，求吸收目标集合的期望首达步数")
    print("=" * 70)

    nodes = [0, 1, 2, 3, "x", "y", "z"]
    net = Network(nodes)
    for edge in ((0, 1), (1, 2), (2, 3)):
        net.add_conductor(*edge, Fraction(1))
    net.add_conductor("x", "y", Fraction(1))
    net.set_stay_weight(1, Fraction(2))

    result = expected_hitting_times(net, [0])
    print("\n有限路径 0-1-2-3：节点 1 的停留权重为 2；(x,y) 与孤立点 z 不连通")
    print("期望首次到达步数：")
    for node in nodes:
        value = result.expectations[node]
        print(f"  h({node!s:>1}) = {value}")
    print("节点 1 的转移概率：", result.transition_probabilities[1])
    print("方程残差：", {str(k): v for k, v in result.residuals.items() if v is not None})
    print("最大绝对残差：", result.max_abs_residual)
    print("不可达分量：", result.unreachable_components)
    print("不可达起点的期望为 +inf，而不是把整个奇异系统交给求解器失败")

    early_bound = independent_tail_bound(net, 0, start=3, horizon=20)
    h = 80
    lower_bound = independent_tail_bound(net, 0, start=3, horizon=h)
    gap = result.expectations[3] - lower_bound
    print(
        f"\n独立有限步尾和：H=20 时 {float(early_bound):.12f}；"
        f"H={h} 时 {float(lower_bound):.12f} <= E[T_0|3] = {result.expectations[3]}"
    )
    print(f"剩余尾概率上界差额 E[T]-尾和 = {gap}（约 {float(gap):.3e}），下界随 H 单调逼近")

    star = Network(["t", "c", "a", "b"])
    star.add_conductor("t", "c", 2)
    star.add_conductor("c", "a", 1)
    star.add_conductor("c", "b", 3)
    star.set_stay_weight("c", 4)
    star_result = expected_hitting_times(star, ["t"])
    print("\n加权星形 + 中心自环（停留权重不是电阻支路）：")
    print("  h(c) =", star_result.expectations["c"], "；h(a)=h(b) =", star_result.expectations["a"])
    print("  P(c,·) =", star_result.transition_probabilities["c"])

    triangle = Network(["a", "b", "c"])
    triangle.add_conductor("a", "b", 1)
    triangle.add_conductor("b", "c", 2)
    triangle.add_conductor("c", "a", 3)
    h_ac = expected_hitting_times(triangle, ["c"]).expectations["a"]
    h_ca = expected_hitting_times(triangle, ["a"]).expectations["c"]
    red = schur_eliminate(triangle, ["a", "c"])
    resistance = Fraction(1) / red.dtn[0][0]
    commute = h_ac + h_ca
    identity = 2 * Fraction(6) * resistance
    print("\n无自环单目标附加检查：通勤时间 H(a,c)+H(c,a) = 2M R_eff")
    print(f"  H(a,c)={h_ac}, H(c,a)={h_ca}, 通勤时间={commute}, 2MR_eff={identity}")
    assert commute == identity

    print("\n边界错误：停留权重必须是非负精确有理数")
    try:
        expected_hitting_times(net, [0], stay_weights={2: Fraction(-1)})
    except StayWeightError as exc:
        print(f"  捕获 {type(exc).__name__}: {exc}（node={exc.node}）")
    else:  # pragma: no cover
        raise AssertionError("负停留权重未被拒绝")
    print("  普通电阻自环仍由 Network 拒绝；停留权重不进入 Laplacian。")
    print("\n演示完成。")


if __name__ == "__main__":
    main()
