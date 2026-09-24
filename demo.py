"""固定输入演示：电阻网络直流消元 + R/L/C 交流相量分析。

运行：``python demo.py``（Python 3.14，仅标准库，数秒内完成）。

演示内容
--------
1. 直流（既有能力）：含内部节点的桥型网络 + 无端口组件 + 孤立节点的
   Schur 消元，与 demo 内**独立编写**的全图线性系统求解逐项比较；
   实际触发零电阻 ``ZeroResistanceError`` 与 NaN 电导拒绝。
2. 交流（本次新增）：在同一数据模型上加电容/电感，用修正结点方程在
   非零频率点求端口复导纳——单 RC、串联 RLC 与解析式独立对照；
   纯电阻交流结果与直流 DtN 一致、负频率共轭、并联谐振开路等边界；
   实际触发零频 ``FrequencyError``、LC 串联谐振 ``SingularSystemError``
   以及对含 C/L 网络调用直流接口的拒绝。

所有数值真实计算（直流精确有理，交流实/虚部均为精确有理分数），
无 sleep、无预录结果。
"""

from __future__ import annotations

import cmath
from fractions import Fraction

from resistor_elim import (
    Network,
    QComplex,
    ZeroResistanceError,
    ConductanceError,
    FrequencyError,
    NetworkError,
    SingularSystemError,
    ac_admittance,
    schur_eliminate,
)


def independent_full_solve(net: Network, vport: dict) -> dict:
    """demo 专用独立参考：固定端口电势，对非浮接组件做全图 KCL 求解。

    与被测核心实现相互独立（自行拼装方程、自行高斯消元），仅用于演示
    “简化前（全图）”与“简化后（DtN）”的一致性。
    """
    nodes = list(net.nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    k = [[Fraction(0) for _ in nodes] for _ in nodes]
    for u, v, g in net.branches:
        i, j = idx[u], idx[v]
        k[i][j] -= g
        k[j][i] -= g
        k[i][i] += g
        k[j][j] += g

    # 无向连通组件（独立 BFS）
    adj = {v: set() for v in nodes}
    for u, v, _ in net.branches:
        adj[u].add(v)
        adj[v].add(u)
    seen, comps = set(), []
    for start in nodes:
        if start in seen:
            continue
        stack, comp = [start], {start}
        seen.add(start)
        while stack:
            x = stack.pop()
            for y in adj[x] - comp:
                comp.add(y)
                stack.append(y)
        comps.append(comp)

    voltage = dict(vport)
    for comp in comps:
        cports = [p for p in vport if p in comp]
        if not cports:
            continue  # 浮接组件：电势只确定到常数
        interior = [v for v in nodes if v in comp and v not in vport]
        if not interior:
            continue
        ii = [idx[v] for v in interior]
        a = [[k[i][j] for j in ii] for i in ii]
        b = [
            -sum(k[ii[r]][idx[p]] * vport[p] for p in cports)
            for r in range(len(interior))
        ]
        # 独立高斯-若尔当消元
        m = [row[:] + [rhs] for row, rhs in zip(a, b)]
        for col in range(len(m)):
            pivot = next(r for r in range(col, len(m)) if m[r][col] != 0)
            m[col], m[pivot] = m[pivot], m[col]
            piv = m[col][col]
            m[col] = [x / piv for x in m[col]]
            for r in range(len(m)):
                if r != col and m[r][col] != 0:
                    f = m[r][col]
                    m[r] = [x - f * y for x, y in zip(m[r], m[col])]
        for v, row in zip(interior, m):
            voltage[v] = row[-1]
    return voltage


def fmt(x: Fraction) -> str:
    return f"{x} ({float(x):.6f})"


def print_matrix(name, mat) -> None:
    print(name)
    for row in mat:
        print("   [" + ", ".join(fmt(x).split(" ")[0] for x in row) + "]")


def main() -> None:
    print("=" * 68)
    print("演示 A：直流 —— 定向结点网络的电阻等效消元（Schur / DtN）")
    print("=" * 68)
    dc_demo()
    print()
    print("=" * 68)
    print("演示 B：交流 —— R/L/C 修正结点方程求端口复导纳（非零频率）")
    print("=" * 68)
    ac_demo()


def dc_demo() -> None:

    # -------------------------------------------------------------- 建网
    # 端口 p, q；内部 a, b（带桥臂 a-b 的非平凡网络）
    net = Network(["p", "q", "a", "b", "x", "y", "z"])
    net.add_resistor("p", "a", 2)
    net.add_resistor("a", "q", 3)
    net.add_resistor("p", "b", 4)
    net.add_resistor("b", "q", 5)
    net.add_resistor("a", "b", 6)
    net.add_resistor("x", "y", 7)   # 无端口组件 → floating
    # z 完全孤立
    print(f"网络：{len(net)} 个节点，{len(net.branches)} 条电阻支路")
    print("端口：p, q；内部：a, b；无端口组件：(x,y)；孤立节点：z")

    # -------------------------------------------------------------- 消元
    red = schur_eliminate(net, ["p", "q"])
    print()
    print_matrix("简化后端口 DtN 矩阵 S（精确分数）：", red.dtn)

    sym_ok = all(red.dtn[i][j] == red.dtn[j][i] for i in range(2) for j in range(2))
    rowsum = [sum(row) for row in red.dtn]
    print(f"对称性检查：{'通过' if sym_ok else '失败'}；各行和：{rowsum}")
    print("浮接组件（显式标记，不并入端口模型）：", red.floating_components)

    # ----------------------------------------------- 简化前 vs 简化后
    vport = {"p": Fraction(12), "q": Fraction(0)}
    sol = red.solve(vport)

    full_v = independent_full_solve(net, vport)
    print()
    print("给定 V(p)=12, V(q)=0：")
    print("  内部电势恢复（简化后 DtN 回溯 vs 简化前全图独立求解）：")
    for node in ["a", "b"]:
        v_after = sol.potentials[node]
        v_before = full_v[node]
        flag = "一致" if v_after == v_before else "不一致!"
        print(f"    V({node}) = {fmt(v_after)}   全图参考 = {fmt(v_before)}  [{flag}]")

    # 全图参考的端口电流（按支路欧姆定律独立累加）
    ref_ip = Fraction(0)
    for u, w, g in net.branches:
        if u == "p":
            ref_ip += g * (full_v["p"] - full_v[w])
        if w == "p":
            ref_ip += g * (full_v[w] - full_v["p"])
    print("  端口注入电流：")
    print(f"    I(p) 简化后 = {fmt(sol.port_currents['p'])}   "
          f"全图参考 = {fmt(ref_ip)}  "
          f"[{'一致' if sol.port_currents['p'] == ref_ip else '不一致!'}]")
    print(f"    I(q) 简化后 = {fmt(sol.port_currents['q'])}（流回电源，为负）")

    r_eff = Fraction(12) / sol.port_currents["p"]
    print(f"  等效输入电阻 R_eq = V/I = {fmt(r_eff)} Ω")

    print("  功率守恒（精确）：")
    print(f"    端口注入功率 Σ V_p I_p = {fmt(sol.port_power)} W")
    print(f"    支路耗散功率 Σ g(ΔV)² = {fmt(sol.dissipated_power)} W")
    assert sol.port_power == sol.dissipated_power
    print("    二者精确相等；浮接组件 (x,y)、z 不参与耗散（支路功率为 None）")
    for (u, v, _), p in zip(net.branches, sol.branch_power):
        if u in ("x", "y", "z") or v in ("x", "y", "z"):
            assert p is None

    # -------------------------------------------------------------- 拒绝边界
    print()
    print("-" * 68)
    print("拒绝边界：零电阻不在正电导无源子集内，必须被明确拒绝")
    try:
        net.add_resistor("p", "q", 0)
    except ZeroResistanceError as exc:
        print(f"  捕获 {type(exc).__name__}（可定位 edge={exc.edge}）：")
        print(f"    {exc}")
    else:  # pragma: no cover
        raise AssertionError("零电阻未被拒绝，违反契约")
    print(f"  网络状态未被污染：支路数仍为 {len(net.branches)}")

    # 另一个数值边界：NaN 电导
    try:
        net.add_conductor("p", "q", float("nan"))
    except ConductanceError as exc:
        print(f"  另：NaN 电导同样被拒绝 → {type(exc).__name__}: {exc}")

    print()
    print("直流演示完成：简化前后端口行为精确一致，拒绝边界实际触发。")


# ---------------------------------------------------------------------- #
# 交流演示辅助
# ---------------------------------------------------------------------- #
def cfmt(z) -> str:
    """精确 QComplex 的人类可读形式（实部、虚部均为精确分数 + 小数近似）。"""
    re_f, im_f = z.re, z.im
    tail = ""
    try:
        tail = f"  ≈ {complex(float(re_f), float(im_f)):.6g}"
    except (OverflowError, ValueError):  # pragma: no cover
        pass
    return f"{re_f}{'+' if im_f >= 0 else '-'}j*{abs(im_f)}{tail}"


def independent_series_admittance(parts, omega: Fraction):
    """demo 专用独立解析式：若干 R/L/C 元件串联的支路导纳（内置 complex）。

    与被测核心完全独立（直接按 Z_R=R、Z_L=jωL、Z_C=1/(jωC) 求和后取倒）。
    """
    z = 0j
    w = float(omega)
    for kind, val in parts:
        v = float(val)
        if kind == "r":
            z += v
        elif kind == "l":
            z += 1j * w * v
        elif kind == "c":
            z += 1.0 / (1j * w * v)
    return 1.0 / z


def ac_demo() -> None:
    # ------------------------------------------------------ 1) 单 RC（并联）
    print("1) 单 RC 并联，R=2Ω，C=1/3 F，角频率 ω=5 rad/s")
    net = Network(["p", "q"])
    net.add_resistor("p", "q", 2)
    net.add_capacitor("p", "q", Fraction(1, 3))
    res = ac_admittance(net, ["p", "q"], 5)
    y11 = res.y[0][0]
    analytic = 1.0 / 2.0 + 1j * 5.0 * (1.0 / 3.0)
    print(f"   Y11 = {cfmt(y11)} S")
    print(f"   解析式 1/R + jωC = 1/2 + j·5/3 ≈ {analytic:.6g} S   "
          f"[{'一致' if cmath.isclose(complex(float(y11.re), float(y11.im)), analytic, rel_tol=1e-12) else '不一致!'}]")

    # ------------------------------------------- 2) 串联 RLC + 支路量恢复
    print()
    print("2) 串联 RLC：R=2Ω，L=3H，C=1/6 F，ω=5；V(p)=1, V(q)=0")
    net2 = Network(["p", "a", "b", "q"])
    net2.add_resistor("p", "a", 2)
    net2.add_inductor("a", "b", 3)
    net2.add_capacitor("b", "q", Fraction(1, 6))
    res2 = ac_admittance(net2, ["p", "q"], 5)
    y_s = res2.y[0][0]
    z_analytic = independent_series_admittance(
        [("r", 2), ("l", 3), ("c", Fraction(1, 6))], 5
    )
    print(f"   串联导纳 Y = {cfmt(y_s)} S   解析式 ≈ {z_analytic:.6g}   "
          f"[{'一致' if cmath.isclose(complex(float(y_s.re), float(y_s.im)), z_analytic, rel_tol=1e-12) else '不一致!'}]")
    sol = res2.solve({"p": 1, "q": 0})
    print(f"   回路电流（R/L/C 三支路相同）= {cfmt(sol.branch_currents[0])} A")
    print(f"   内部相量电压 V(a) = {cfmt(sol.potentials['a'])} V，"
          f"V(b) = {cfmt(sol.potentials['b'])} V")
    vr = 2 * sol.branch_currents[0]
    vl = sol.potentials["a"] - sol.potentials["b"]
    vc = sol.potentials["b"]
    kvl = vr + vl + vc
    print(f"   KVL 校验：V_R + V_L + V_C = {cfmt(kvl)} = 1  "
          f"[{'通过' if kvl == QComplex(1) else '失败'}]")
    print(f"   平均有功功率：端口 {float(sol.average_port_power):.6g} W，"
          f"电阻耗散 {float(sol.average_dissipated_power):.6g} W（精确相等）")
    assert sol.average_port_power == sol.average_dissipated_power

    # ------------------------------------------- 3) 纯电阻交流 == 直流 DtN
    print()
    print("3) 边界：纯电阻网络在交流下退化为直流 DtN（虚部严格为 0）")
    net3 = Network(["p", "q", "m"])
    net3.add_resistor("p", "m", 2)
    net3.add_resistor("m", "q", 6)
    y3 = ac_admittance(net3, ["p", "q"], Fraction(99, 7)).y
    dtn = schur_eliminate(net3, ["p", "q"]).dtn
    ok3 = y3[0][0] == QComplex(dtn[0][0])
    print(f"   Y11 = {cfmt(y3[0][0])} S；直流 DtN[0][0] = {dtn[0][0]} S  "
          f"[{'精确一致' if ok3 else '不一致!'}]")

    # ------------------------------------------- 4) 负频率共轭、谐振开路
    print()
    print("4) 边界：Y(-ω) = conj(Y(ω))；理想 LC 并联在谐振点导纳为 0")
    net4 = Network(["p", "q"])
    net4.add_resistor("p", "q", 3)
    net4.add_capacitor("p", "q", Fraction(2, 5))
    net4.add_inductor("p", "q", 2)
    yp = ac_admittance(net4, ["p", "q"], 4).y[0][0]
    ym = ac_admittance(net4, ["p", "q"], -4).y[0][0]
    print(f"   Y(+4) = {cfmt(yp)}；Y(-4) = {cfmt(ym)}  "
          f"[{'共轭关系成立' if ym == yp.conj() else '失败'}]")
    net4b = Network(["p", "q"])
    net4b.add_inductor("p", "q", 1)
    net4b.add_capacitor("p", "q", 1)
    yr = ac_admittance(net4b, ["p", "q"], 1).y[0][0]
    print(f"   理想 L//C（L=C=1）在 ω=1：Y = {cfmt(yr)} S（并联谐振 ⇒ 开路，零电流）")
    assert yr == QComplex(0)

    # ------------------------------------------- 5) 浮置组件仍显式标记
    print()
    print("5) 边界：无端口的 LC 组件为 floating，不并入端口模型")
    net5 = Network(["p", "q", "x", "y", "z"])
    net5.add_resistor("p", "q", 4)
    net5.add_capacitor("x", "y", 2)
    net5.add_inductor("x", "y", 3)
    res5 = ac_admittance(net5, ["p", "q"], 6)
    sol5 = res5.solve([1, 0])
    print(f"   floating 组件：{res5.floating_components}")
    print(f"   端口 Y11 = {cfmt(res5.y[0][0])} S（仅电阻 1/4）；"
          f"浮置电容/电感电流 = {sol5.capacitor_currents[0]}/{sol5.inductor_currents[0]}")

    # ------------------------------------------- 6) 实际错误
    print()
    print("-" * 68)
    print("6) 实际触发的错误边界：")

    def show(title, fn, exc_type):
        try:
            fn()
        except exc_type as exc:
            print(f"   {title}\n      → {type(exc).__name__}: {exc}")
        else:  # pragma: no cover
            raise AssertionError(title)

    net6 = Network(["p", "q"])
    net6.add_resistor("p", "q", 1)
    show("零角频率（零频极限不在契约内）：",
         lambda: ac_admittance(net6, ["p", "q"], 0), FrequencyError)
    show("非有限角频率 ω=NaN：",
         lambda: ac_admittance(net6, ["p", "q"], float("nan")), FrequencyError)

    net6b = Network(["p", "a", "q"])
    net6b.add_inductor("p", "a", 1)
    net6b.add_capacitor("a", "q", 1)
    show("理想 LC 串联在谐振点 ω=1（内部系统奇异，不得掩盖）：",
         lambda: ac_admittance(net6b, ["p", "q"], 1), SingularSystemError)

    net6c = Network(["p", "q"])
    net6c.add_capacitor("p", "q", 1)
    show("对含电容的网络调用直流接口：",
         lambda: schur_eliminate(net6c, ["p", "q"]), NetworkError)
    print("   （谐振报错后网络仍可用：偏离谐振点 ω=2 时 "
          f"Y = {cfmt(ac_admittance(net6b, ['p', 'q'], 2).y[0][0])} S）")

    print()
    print("交流演示完成：解析对照、边界行为与实际错误均已真实计算并触发。")


if __name__ == "__main__":
    main()
