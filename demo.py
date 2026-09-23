"""固定输入演示：电阻网络的定向结点等效消元。

运行：``python demo.py``（Python 3.14，仅标准库，数秒内完成）。

演示内容
--------
1. 正常结果：一个含内部节点的桥型二端口网络 + 一个无端口组件 + 一个孤立
   节点；Schur 消元得到端口 DtN，检查对称性/行和 0，给定端口电势后恢复
   内部电势、端口电流、功率，并与 demo 内**独立编写**的全图线性系统求解
   （独立拼装 KCL + 独立高斯消元）逐项比较简化前后。
2. 拒绝边界：实际尝试加入零电阻，捕获并打印
   :class:`resistor_elim.ZeroResistanceError`。

所有数值真实计算（精确有理数），无 sleep、无预录结果。
"""

from __future__ import annotations

from fractions import Fraction

from resistor_elim import Network, ZeroResistanceError, schur_eliminate


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
    print("演示：定向结点网络的电阻等效消元（Schur / Dirichlet-to-Neumann）")
    print("=" * 68)

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
    from resistor_elim import ConductanceError

    try:
        net.add_conductor("p", "q", float("nan"))
    except ConductanceError as exc:
        print(f"  另：NaN 电导同样被拒绝 → {type(exc).__name__}: {exc}")

    print()
    print("演示完成：简化前后端口行为精确一致，拒绝边界实际触发。")


if __name__ == "__main__":
    main()
