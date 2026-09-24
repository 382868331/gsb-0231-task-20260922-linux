"""独立参考实现（仅供测试）。

本模块**不导入也不调用** ``resistor_elim`` 的任何核心算法：拉普拉斯由
支路表独立拼装，线性求解用独立编写的高斯消元（带行交换的增广矩阵消元），
期望值另有串/并联解析式。被测库与参考之间只允许比较结论，不允许共享计算。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Hashable, Sequence


# --------------------------------------------------------------------- #
# 独立的分数高斯消元（与 resistor_elim.linalg 的实现相互独立编写）
# --------------------------------------------------------------------- #
def gauss_solve(a: list[list[Fraction]], b: list[Fraction]) -> list[Fraction]:
    """解线性方程组 a x = b。假设系统非奇异；独立的前向消元+回代实现。"""
    n = len(a)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    row = 0
    for col in range(n):
        pivot = next((r for r in range(row, n) if m[r][col] != 0), None)
        if pivot is None:
            raise ZeroDivisionError("参考求解器遇到奇异矩阵")
        m[row], m[pivot] = m[pivot], m[row]
        piv = m[row][col]
        m[row] = [x / piv for x in m[row]]
        for r in range(n):
            if r == row:
                continue
            f = m[r][col]
            if f:
                m[r] = [x - f * y for x, y in zip(m[r], m[row])]
        row += 1
    return [m[i][n] for i in range(n)]


# --------------------------------------------------------------------- #
# 独立的网络参考模型：节点表 + (u, v, g) 支路表
# --------------------------------------------------------------------- #
class RefNetwork:
    def __init__(self, nodes: Sequence[Hashable], branches):
        self.nodes = tuple(nodes)
        self.idx = {v: i for i, v in enumerate(self.nodes)}
        # branches: (u, v, conductance: Fraction)
        self.branches = [(u, v, Fraction(g)) for u, v, g in branches]

    def laplacian(self) -> list[list[Fraction]]:
        n = len(self.nodes)
        k = [[Fraction(0) for _ in range(n)] for _ in range(n)]
        for u, v, g in self.branches:
            i, j = self.idx[u], self.idx[v]
            k[i][j] -= g
            k[j][i] -= g
            k[i][i] += g
            k[j][j] += g
        return k

    def connected_components(self) -> list[set[Hashable]]:
        adj: dict[Hashable, set[Hashable]] = {v: set() for v in self.nodes}
        for u, v, _ in self.branches:
            adj[u].add(v)
            adj[v].add(u)
        seen: set[Hashable] = set()
        comps: list[set[Hashable]] = []
        for start in self.nodes:
            if start in seen:
                continue
            stack, comp = [start], {start}
            seen.add(start)
            while stack:
                x = stack.pop()
                for y in adj[x] - seen:
                    seen.add(y)
                    comp.add(y)
                    stack.append(y)
            comps.append(comp)
        return comps

    def solve_with_port_potentials(
        self,
        ports: Sequence[Hashable],
        vport: dict[Hashable, Fraction],
    ) -> dict[Hashable, Fraction]:
        """全图线性系统参考：固定端口电势，独立求解全部非浮接节点电势。

        对不含端口的组件不求解（电势只确定到常数）。
        """
        k = self.laplacian()
        port_set = set(ports)
        comps = self.connected_components()
        voltage: dict[Hashable, Fraction] = {}
        for comp in comps:
            cports = [p for p in ports if p in comp]
            if not cports:
                continue
            interior = [v for v in self.nodes if v in comp and v not in port_set]
            voltage.update(vport)
            if not interior:
                continue
            # K_II V_I = - K_IP V_P，独立拼装并求解
            ii = [self.idx[v] for v in interior]
            ip = [self.idx[p] for p in cports]
            a = [[k[i][j] for j in ii] for i in ii]
            b = [
                -sum(k[ii[r]][ip[c]] * vport[cports[c]] for c in range(len(ip)))
                for r in range(len(ii))
            ]
            x = gauss_solve(a, b)
            for v, val in zip(interior, x):
                voltage[v] = val
        return voltage

    def port_currents(
        self,
        ports: Sequence[Hashable],
        voltage: dict[Hashable, Fraction],
    ) -> dict[Hashable, Fraction]:
        """按支路欧姆定律独立统计每个端口的注入电流。"""
        cur = {p: Fraction(0) for p in ports}
        for u, v, g in self.branches:
            if u in cur:
                cur[u] += g * (voltage[u] - voltage[v])
            if v in cur:
                cur[v] += g * (voltage[v] - voltage[u])
        return cur

    def branch_quantities(self, voltage: dict[Hashable, Fraction]):
        out = []
        for u, v, g in self.branches:
            if u not in voltage or v not in voltage:
                out.append(None)
                continue
            du = voltage[u] - voltage[v]
            out.append((g * du, g * du * du))  # 电流(u->v)，耗散功率
        return out


# --------------------------------------------------------------------- #
# 串并联解析式
# --------------------------------------------------------------------- #
def series_dtn(resistances: Sequence[Fraction]) -> list[list[Fraction]]:
    """若干电阻串联在两个端口之间的二端口 DtN（电导 = 1/sum R）。"""
    g = Fraction(1) / sum(Fraction(r) for r in resistances)
    return [[g, -g], [-g, g]]


def parallel_dtn(conductances: Sequence[Fraction]) -> list[list[Fraction]]:
    """若干电导并联在两个端口之间的二端口 DtN。"""
    g = sum(Fraction(x) for x in conductances)
    return [[g, -g], [-g, g]]


# ===================================================================== #
# 相量（正弦稳态）独立参考：独立的复数元组运算 + 独立复高斯消元
#
# 不导入 resistor_elim 的任何内容：复数用 (re, im) 的 Fraction 二元组
# 自行定义四则运算，相量导纳矩阵由支路表独立拼装。
# ===================================================================== #
def cadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def csub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def cmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def cdiv(a, b):
    d = b[0] * b[0] + b[1] * b[1]
    return ((a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d)


def cconj(a):
    return (a[0], -a[1])


def cgauss_solve(a, b):
    """独立复高斯-若尔当消元：解 a x = b；奇异时抛 ZeroDivisionError。"""
    n = len(a)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = next(
            (r for r in range(col, n) if m[r][col][0] != 0 or m[r][col][1] != 0),
            None,
        )
        if pivot is None:
            raise ZeroDivisionError("参考相量求解器遇到奇异矩阵")
        m[col], m[pivot] = m[pivot], m[col]
        piv = m[col][col]
        m[col] = [cdiv(x, piv) for x in m[col]]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col]
            if f[0] != 0 or f[1] != 0:
                m[r] = [csub(x, cmul(f, y)) for x, y in zip(m[r], m[col])]
    return [m[i][n] for i in range(n)]


class RefACNetwork:
    """独立相量网络参考：节点表 + ``(u, v, kind, value)`` 支路表。

    kind 为 ``"R"``（value=电导）、``"C"``（电容）或 ``"L"``（电感）。
    """

    def __init__(self, nodes, branches):
        self.nodes = tuple(nodes)
        self.idx = {v: i for i, v in enumerate(self.nodes)}
        self.branches = [(u, v, k, Fraction(val)) for u, v, k, val in branches]

    def _branch_admittance(self, kind, value, w):
        if kind == "R":
            return (value, Fraction(0))
        if kind == "C":
            return (Fraction(0), w * value)
        return (Fraction(0), -Fraction(1) / (w * value))  # 电感

    def admittance(self, omega):
        w = Fraction(omega)
        n = len(self.nodes)
        zero = (Fraction(0), Fraction(0))
        y = [[zero for _ in range(n)] for _ in range(n)]
        for u, v, kind, value in self.branches:
            i, j = self.idx[u], self.idx[v]
            yb = self._branch_admittance(kind, value, w)
            y[i][j] = csub(y[i][j], yb)
            y[j][i] = csub(y[j][i], yb)
            y[i][i] = cadd(y[i][i], yb)
            y[j][j] = cadd(y[j][j], yb)
        return y

    def connected_components(self):
        adj = {v: set() for v in self.nodes}
        for u, v, _k, _val in self.branches:
            adj[u].add(v)
            adj[v].add(u)
        seen, comps = set(), []
        for start in self.nodes:
            if start in seen:
                continue
            stack, comp = [start], {start}
            seen.add(start)
            while stack:
                x = stack.pop()
                for y in adj[x] - seen:
                    seen.add(y)
                    comp.add(y)
                    stack.append(y)
            comps.append(comp)
        return comps

    def port_admittance(self, ports, omega):
        """独立全图 Schur：逐组件固定端口、消去内部节点，拼全局端口导纳。"""
        w = Fraction(omega)
        y = self.admittance(w)
        port_set = set(ports)
        n_p = len(ports)
        dtn = [[(Fraction(0), Fraction(0)) for _ in range(n_p)] for _ in range(n_p)]
        for comp in self.connected_components():
            cports = [p for p in ports if p in comp]
            if not cports:
                continue
            interior = [v for v in self.nodes if v in comp and v not in port_set]
            pi = [self.idx[p] for p in cports]
            ii = [self.idx[v] for v in interior]
            if interior:
                yii = [[y[i][j] for j in ii] for i in ii]
                # 用单位右端求 Y_II^{-1}
                ident = [
                    [(Fraction(1) if r == c else Fraction(0), Fraction(0))
                     for c in range(len(ii))]
                    for r in range(len(ii))
                ]
                m = [row[:] + ident[r] for r, row in enumerate(yii)]
                for col in range(len(ii)):
                    pivot = next(
                        r for r in range(col, len(ii))
                        if m[r][col][0] != 0 or m[r][col][1] != 0
                    )
                    m[col], m[pivot] = m[pivot], m[col]
                    piv = m[col][col]
                    m[col] = [cdiv(x, piv) for x in m[col]]
                    for r in range(len(ii)):
                        if r != col and (m[r][col][0] or m[r][col][1]):
                            f = m[r][col]
                            m[r] = [
                                csub(x, cmul(f, z)) for x, z in zip(m[r], m[col])
                            ]
                k = len(ii)
                yii_inv = [[m[r][k + c] for c in range(k)] for r in range(k)]
            else:
                yii_inv = []
            for a, pa in enumerate(cports):
                ga = ports.index(pa)
                for b, pb in enumerate(cports):
                    gb = ports.index(pb)
                    ia, ib = self.idx[pa], self.idx[pb]
                    val = y[ia][ib]
                    if interior:
                        for t in range(len(ii)):
                            for s in range(len(ii)):
                                # - Y_PI Y_II^-1 Y_IP
                                val = csub(
                                    val,
                                    cmul(cmul(y[ia][ii[t]], yii_inv[t][s]),
                                         y[ii[s]][ib]),
                                )
                    dtn[ga][gb] = cadd(dtn[ga][gb], val)
        return dtn

    def solve_with_port_potentials(self, ports, vport, omega):
        """固定端口相量电势，独立求解全部非浮置节点的相量电势。"""
        w = Fraction(omega)
        y = self.admittance(w)
        comps = self.connected_components()
        voltage = dict(vport)
        for comp in comps:
            cports = [p for p in ports if p in comp]
            if not cports:
                continue
            interior = [v for v in self.nodes if v in comp and v not in ports]
            if not interior:
                continue
            ii = [self.idx[v] for v in interior]
            a = [[y[i][j] for j in ii] for i in ii]
            zero = (Fraction(0), Fraction(0))
            b = [zero for _ in interior]
            for r, node in enumerate(interior):
                for p in cports:
                    b[r] = csub(b[r], cmul(y[self.idx[node]][self.idx[p]], vport[p]))
            x = cgauss_solve(a, b)
            for v, val in zip(interior, x):
                voltage[v] = val
        return voltage

    def port_currents(self, ports, voltage, omega):
        """按支路相量欧姆定律独立统计每个端口的注入电流。"""
        w = Fraction(omega)
        cur = {p: (Fraction(0), Fraction(0)) for p in ports}
        for u, v, kind, value in self.branches:
            yb = self._branch_admittance(kind, value, w)
            if u in cur:
                cur[u] = cadd(cur[u], cmul(yb, csub(voltage[u], voltage[v])))
            if v in cur:
                cur[v] = cadd(cur[v], cmul(yb, csub(voltage[v], voltage[u])))
        return cur


# --------------------------------------------------------------------- #
# RLC 解析式（二端器件串/并联，直接给出复导纳 (re, im)）
# --------------------------------------------------------------------- #
def analytic_parallel_rc(g, c, omega):
    """电阻电导 g 与电容 C 并联：Y = g + jωC。"""
    w = Fraction(omega)
    return (Fraction(g), w * Fraction(c))


def analytic_series_rc(r, c, omega):
    """电阻 R 与电容 C 串联：Z = R + 1/(jωC)，返回 Y = 1/Z。"""
    w = Fraction(omega)
    z = (Fraction(r), -Fraction(1) / (w * Fraction(c)))
    return cdiv((Fraction(1), Fraction(0)), z)


def analytic_series_rlc(r, l, c, omega):
    """R、L、C 串联：Z = R + jωL + 1/(jωC)，返回 Y = 1/Z。"""
    w = Fraction(omega)
    z = (
        Fraction(r),
        w * Fraction(l) - Fraction(1) / (w * Fraction(c)),
    )
    return cdiv((Fraction(1), Fraction(0)), z)


def analytic_resistor_admittance(g):
    """纯电阻二端导纳（实数）。"""
    return (Fraction(g), Fraction(0))
