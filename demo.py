"""固定输入演示：电阻网络等效消元 + R/L/C 相量端口复导纳。

运行：``python demo.py``（Python 3.14，仅标准库，1 秒内完成）。

演示内容
--------
A. 直流/电阻（旧能力，行为不变）：含内部节点的桥型网络 + 浮置组件 + 孤立
   节点；Schur 消元得到 DtN，与 demo 内**独立编写**的全图线性系统求解
   逐项比较；实际触发零电阻、NaN 电导拒绝。
B. 交流相量（新能力）：R/L/C 网络在给定非零有限角频率下的端口复导纳，
   与解析式（串 RC、串 RLC）和 demo 内**独立编写**的复数全图 KCL 求解
   精确对照；展示浮置 LC 组件、负频率共轭；实际触发零频 ``FrequencyError``
   与纯 LC 谐振 ``SingularSystemError``（带谐振组件定位）。

所有数值真实计算（精确有理 / 精确复有理），无 sleep、无预录结果。
"""

from __future__ import annotations

from fractions import Fraction

from resistor_elim import (
    CFrac,
    CapacitanceError,
    ConductanceError,
    FrequencyError,
    Network,
    SingularSystemError,
    ZeroResistanceError,
    ac_schur_eliminate,
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


def cfmt(z: CFrac) -> str:
    """精确复分数的人类可读形式（浮点仅用于括号内近似）。"""
    c = complex(z)
    return f"{z}  ≈ {c.real:.6f}{'+' if c.imag >= 0 else '-'}j{abs(c.imag):.6f}"


def independent_ac_full_solve(net: Network, vport: dict, omega: Fraction) -> dict:
    """demo 专用独立相量参考：固定端口相量电势，全图复 KCL 求解。

    复数用 (re, im) 二元组自行四则运算，消元自行编写，不调用被测核心。
    """
    w = Fraction(omega)
    nodes = list(net.nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    zero = (Fraction(0), Fraction(0))

    def adm(kind, value):
        if kind == "R":
            return (value, Fraction(0))
        if kind == "C":
            return (Fraction(0), w * value)
        return (Fraction(0), -Fraction(1) / (w * value))

    y = [[zero for _ in nodes] for _ in nodes]
    adj = {v: set() for v in nodes}
    for u, v, kind, value in net.branches_ac:
        i, j = idx[u], idx[v]
        yb = adm(kind, value)
        y[i][j] = (y[i][j][0] - yb[0], y[i][j][1] - yb[1])
        y[j][i] = (y[j][i][0] - yb[0], y[j][i][1] - yb[1])
        y[i][i] = (y[i][i][0] + yb[0], y[i][i][1] + yb[1])
        y[j][j] = (y[j][j][0] + yb[0], y[j][j][1] + yb[1])
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
            for z in adj[x] - comp:
                comp.add(z)
                stack.append(z)
        comps.append(comp)

    def cdiv(a, b):
        d = b[0] * b[0] + b[1] * b[1]
        return ((a[0] * b[0] + a[1] * b[1]) / d,
                (a[1] * b[0] - a[0] * b[1]) / d)

    voltage = {k: (v.re, v.im) for k, v in vport.items()}
    for comp in comps:
        cports = [p for p in vport if p in comp]
        if not cports:
            continue
        interior = [v for v in nodes if v in comp and v not in vport]
        if not interior:
            continue
        ii = [idx[v] for v in interior]
        a = [[y[i][j] for j in ii] for i in ii]
        b = []
        for node in interior:
            rhs = zero
            for p in cports:
                yp = y[idx[node]][idx[p]]
                vp = voltage[p]
                rhs = (
                    rhs[0] - (yp[0] * vp[0] - yp[1] * vp[1]),
                    rhs[1] - (yp[0] * vp[1] + yp[1] * vp[0]),
                )
            b.append(rhs)
        m = [row[:] + [rhs] for row, rhs in zip(a, b)]
        for col in range(len(m)):
            pivot = next(
                r for r in range(col, len(m))
                if m[r][col][0] != 0 or m[r][col][1] != 0
            )
            m[col], m[pivot] = m[pivot], m[col]
            piv = m[col][col]
            m[col] = [cdiv(x, piv) for x in m[col]]
            for r in range(len(m)):
                if r == col:
                    continue
                f = m[r][col]
                if f[0] != 0 or f[1] != 0:
                    m[r] = [
                        (
                            x[0] - (f[0] * z[0] - f[1] * z[1]),
                            x[1] - (f[0] * z[1] + f[1] * z[0]),
                        )
                        for x, z in zip(m[r], m[col])
                    ]
        for v, row in zip(interior, m):
            voltage[v] = row[-1]
    return {k: CFrac(v[0], v[1]) for k, v in voltage.items()}


def print_matrix(name, mat) -> None:
    print(name)
    for row in mat:
        print("   [" + ", ".join(fmt(x).split(" ")[0] for x in row) + "]")


def main() -> None:
    print("=" * 68)
    print("A. 直流/电阻：定向结点网络的 Schur 等效消元（旧能力，行为不变）")
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
    try:
        net.add_conductor("p", "q", float("nan"))
    except ConductanceError as exc:
        print(f"  另：NaN 电导同样被拒绝 → {type(exc).__name__}: {exc}")

    # ================================================================== #
    # B. 交流相量：R/L/C 网络在给定非零有限频率下的端口复导纳（新能力）
    # ================================================================== #
    print()
    print("=" * 68)
    print("B. 交流相量（jω）：R/L/C 网络的修正结点分析与端口复导纳")
    print("=" * 68)

    # --- B1. 正常结果：含 R/L/C、内部节点、桥臂的二端口网络 ---------- #
    # 端口 p,q；内部 a,b；另含无端口 LC 组件 (x,y) 与孤立节点 z
    acnet = Network(["p", "q", "a", "b", "x", "y", "z"])
    acnet.add_resistor("p", "a", 2)        # 电导
    acnet.add_capacitor("a", "q", Fraction(1, 3))
    acnet.add_inductor("p", "b", 4)
    acnet.add_resistor("b", "q", 5)
    acnet.add_capacitor("a", "b", Fraction(1, 6))
    acnet.add_inductor("x", "y", 1)        # 无端口 LC（在 ω=1 谐振）
    acnet.add_capacitor("x", "y", 1)
    w = Fraction(3, 2)
    print(f"网络：{len(acnet)} 节点，{len(acnet.branches_ac)} 条 R/L/C 支路；ω={w}")
    red = ac_schur_eliminate(acnet, ["p", "q"], w)
    print("端口复导纳矩阵 Y(jω)（实部=电导，虚部：+容性 / -感性）：")
    for row in red.admittance:
        print("   [" + " , ".join(str(z) for z in row) + "]")
    sym_ok = all(
        red.admittance[i][j] == red.admittance[j][i] for i in range(2) for j in range(2)
    )
    rowsum = [sum(r, CFrac()) for r in red.admittance]
    print(f"复对称检查：{'通过' if sym_ok else '失败'}；各行和：{rowsum}")
    print("浮置组件（无端口；谐振 LC 也只标记、不强求激励）：",
          red.floating_components)

    # --- 与独立全图复 KCL 求解对照内部相量电势与端口电流 ------------- #
    vport = {"p": CFrac(6, 4), "q": CFrac()}   # 6 + 4j, 0
    sol = red.solve(vport)
    full = independent_ac_full_solve(acnet, vport, w)
    print()
    print(f"给定 V(p)=6+4j, V(q)=0（ω={w}）：")
    for node in ["a", "b"]:
        flag = "一致" if sol.potentials[node] == full[node] else "不一致!"
        print(f"    V({node}) 消元回溯 = {cfmt(sol.potentials[node])}")
        print(f"           全图参考   = {cfmt(full[node])}   [{flag}]")
    # 端口电流独立累加
    ref_ip = CFrac()
    for u, vv, kind, value in acnet.branches_ac:
        if kind == "R":
            yb = CFrac(value)
        elif kind == "C":
            yb = CFrac(0, w * value)
        else:
            yb = CFrac(0, -Fraction(1) / (w * value))
        if u == "p":
            ref_ip = ref_ip + yb * (full["p"] - full[vv])
        if vv == "p":
            ref_ip = ref_ip + yb * (full[vv] - full["p"])
    flag = "一致" if sol.port_currents["p"] == ref_ip else "不一致!"
    print(f"    I(p) = {cfmt(sol.port_currents['p'])}   全图参考核对 [{flag}]")
    print(f"    端口复功率 1/2·V·conj(I) = {cfmt(sol.port_power)} VA")
    print(f"    电阻平均耗散（=复功率实部）= {fmt(sol.dissipated_power)} W")
    assert sol.port_power.re == sol.dissipated_power

    # --- B2. 解析式对照：串 RC、串 RLC、纯电阻 ----------------------- #
    print()
    print("解析式对照（手算分数，精确相等）：")
    # 串 RC：R=2, C=1/4, ω=2 ⇒ Z=2-2j ⇒ Y=1/4+1/4 j
    rc = Network(["p", "m", "q"])
    rc.add_resistor("p", "m", 2)
    rc.add_capacitor("m", "q", Fraction(1, 4))
    y_rc = ac_schur_eliminate(rc, ["p", "q"], 2).admittance[0][0]
    print(f"    串 RC  Y = {cfmt(y_rc)}   解析 1/(2-2j) = 1/4+1/4j")
    assert y_rc == CFrac(Fraction(1, 4), Fraction(1, 4))
    # 串 RLC：R=L=C=1, ω=1 谐振点 ⇒ Z=1 ⇒ Y=1（纯实）
    rlc = Network(["p", "x", "y", "q"])
    rlc.add_resistor("p", "x", 1)
    rlc.add_inductor("x", "y", 1)
    rlc.add_capacitor("y", "q", 1)
    y_rlc = ac_schur_eliminate(rlc, ["p", "q"], 1).admittance[0][0]
    print(f"    串 RLC Y = {cfmt(y_rlc)}   解析 1/(1+j- j) = 1")
    assert y_rlc == CFrac(1)
    # 纯电阻：虚部恒 0，与 DC Schur 结果一致
    pr = Network(["p", "m", "q"])
    pr.add_resistor("p", "m", 2)
    pr.add_resistor("m", "q", 6)
    y_pr = ac_schur_eliminate(pr, ["p", "q"], 11).admittance[0][0]
    print(f"    纯电阻 Y = {cfmt(y_pr)}   解析 1/(2+6) = 1/8（任意频率虚部 0）")
    assert y_pr == CFrac(Fraction(1, 8))

    # 负频率：Y(-ω) = conj(Y(ω))
    yp = ac_schur_eliminate(acnet, ["p", "q"], w).admittance[0][0]
    yn = ac_schur_eliminate(acnet, ["p", "q"], -w).admittance[0][0]
    assert yn == yp.conjugate()
    print(f"    负频率核对：Y(-ω) = conj(Y(ω))  [{yn == yp.conjugate()}]")

    # --- B3. 实际错误：零频、谐振 ------------------------------------ #
    print()
    print("-" * 68)
    print("错误边界（实际触发，非预录文字）：")
    try:
        ac_schur_eliminate(rc, ["p", "q"], 0)
    except FrequencyError as exc:
        print(f"  零频被拒绝 → {type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("零频未被拒绝")

    try:
        ac_schur_eliminate(rc, ["p", "q"], float("nan"))
    except FrequencyError as exc:
        print(f"  NaN 频率被拒绝 → {type(exc).__name__}: {exc}")

    # 纯 LC 串联谐振：p-L-m-C-q，ω=1 导纳极点
    lc = Network(["p", "m", "q"])
    lc.add_inductor("p", "m", 1)
    lc.add_capacitor("m", "q", 1)
    try:
        ac_schur_eliminate(lc, ["p", "q"], 1)
    except SingularSystemError as exc:
        print(f"  纯 LC 谐振明确报告 → {type(exc).__name__}")
        print(f"    谐振组件 = {exc.resonant_components}")
        print(f"    {exc}")
    else:  # pragma: no cover
        raise AssertionError("谐振奇异性未被报告")
    # 网络状态未被污染
    assert len(lc.branches_ac) == 2

    # 串联 0.01Ω 阻尼后同一频率可正常求解：p-R-d-L-m-C-q，
    # ω=1 时 LC 串联阻抗为 0，总阻抗 0.01 ⇒ Y=100（有限）
    damped = Network(["p", "d", "m", "q"])
    damped.add_resistor("p", "d", Fraction(1, 100))
    damped.add_inductor("d", "m", 1)
    damped.add_capacitor("m", "q", 1)
    y_d = ac_schur_eliminate(damped, ["p", "q"], 1).admittance[0][0]
    print(f"  串入 0.01Ω 阻尼后同频率可解：Y = {cfmt(y_d)}（极点变为有限值 100）")
    assert y_d == CFrac(100)

    print()
    print("演示完成：DC 与 AC 结果均与独立参考/解析式精确一致，错误边界实际触发。")


if __name__ == "__main__":
    main()
