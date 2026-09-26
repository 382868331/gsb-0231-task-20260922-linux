"""固定输入演示：加权随机游走的期望首次到达步数（本轮迭代新功能）。

运行：``python demo_iteration.py``（Python 3.14，仅标准库，秒级完成）。

演示内容
--------
1. 正常结果：含停留权重（自环概率）与平行边的网络，求解 Poisson 方程
   (I - P) h = 1 得到每个起点到目标集合的期望首达步数，打印逐节点
   方程残差与连通性诊断，并与手算值对照。
2. 交叉验证：单目标、无停留权重时，通勤时间 = 2 * 总电导 * 有效电阻，
   其中有效电阻由旧的 schur_eliminate 端口 DtN 独立得到。
3. 边界情况：与目标不连通的组件期望为正无穷（math.inf），而不是线性
   求解失败；随后实际触发一个拒绝边界（负停留权重 StayWeightError）。

所有数值真实计算（精确有理数），无 sleep、无预录结果。
"""

from __future__ import annotations

import math
from fractions import Fraction

from resistor_elim import Network, StayWeightError, hitting_times, schur_eliminate


def show(title: str) -> None:
    print("=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    show("1. 期望首达步数：停留权重（自环）+ 平行边 + 手算对照")
    # 路径 s -- m -- t，s 处停留权重 2（自环概率 2/3），
    # m--t 两条平行电导 1 与 1（合并为 2）。
    # 手算：h_t = 0；h_m = 1 + h_s/3；h_s = 1 + (2/3) h_s + h_m/3
    #       -> h_s = 3 + h_m，代入得 h_m = 3, h_s = 6。
    net = Network(["s", "m", "t"])
    net.add_conductor("s", "m", 1)
    net.add_conductor("m", "t", 1)
    net.add_conductor("m", "t", 1)  # 平行边电导相加
    net.set_stay_weight("s", 2)
    res = hitting_times(net, ["t"])
    print("转移概率：p(s->s)=2/3, p(s->m)=1/3；p(m->s)=1/3, p(m->t)=2/3")
    for node in net.nodes:
        print(f"  h({node!r}) = {res.expectations[node]}")
    hand = {"s": Fraction(6), "m": Fraction(3), "t": Fraction(0)}
    assert all(res.expectations[k] == v for k, v in hand.items()), "与手算不符"
    print("手算对照 h(s)=6, h(m)=3, h(t)=0：一致")
    print("Poisson 方程残差 r_u = h_u - 1 - sum_v p(u->v) h_v：")
    for node in net.nodes:
        print(f"  r({node!r}) = {res.residuals[node]}")
    print(f"连通性诊断：含目标组件={res.target_components}, 不可达={res.unreachable}")

    show("2. 通勤时间 = 2 * 总电导 * 有效电阻（与旧 DtN 接口交叉验证）")
    path = Network([1, 2, 3])
    path.add_conductor(1, 2, 1)
    path.add_conductor(2, 3, 1)
    h_13 = hitting_times(path, [3]).expectations[1]
    h_31 = hitting_times(path, [1]).expectations[3]
    commute = h_13 + h_31
    dtn = schur_eliminate(path, [1, 3]).dtn
    r_eff = 1 / dtn[0][0]
    g_total = sum((g for _, _, g in path.branches), Fraction(0))
    print(f"h(1->3)={h_13}, h(3->1)={h_31}, 通勤时间={commute}")
    print(f"有效电阻 R_eff={r_eff}（来自 schur_eliminate DtN），总电导 G={g_total}")
    print(f"2 * G * R_eff = {2 * g_total * r_eff}  ==  通勤时间 {commute}："
          f"{2 * g_total * r_eff == commute}")
    assert 2 * g_total * r_eff == commute

    show("3. 边界：与目标不连通的组件 -> 正无穷（非求解失败）")
    disc = Network(["a", "b", "x", "y", "iso"])
    disc.add_conductor("a", "b", 1)
    disc.add_conductor("x", "y", 2)
    res2 = hitting_times(disc, ["a"])
    for node in disc.nodes:
        v = res2.expectations[node]
        text = "inf（不可达）" if v == math.inf else str(v)
        print(f"  h({node!r}) = {text}")
    print(f"不可达节点诊断：{res2.unreachable}")
    assert all(math.isinf(res2.expectations[n]) for n in ("x", "y", "iso"))

    show("4. 拒绝边界：负停留权重被 StayWeightError 拒绝")
    try:
        net.set_stay_weight("s", -1)
    except StayWeightError as exc:
        print(f"  捕获 StayWeightError: {exc}（node={exc.node!r}）")
    else:
        raise AssertionError("负停留权重未被拒绝")
    print("\n演示完成：全部数值真实计算，无 sleep、无预录。")


if __name__ == "__main__":
    main()
