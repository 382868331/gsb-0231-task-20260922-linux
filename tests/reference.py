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


def hitting_time_lower_bounds(
    nodes: Sequence[Hashable],
    branches,
    target: Hashable,
    horizons: Sequence[int],
    start: Hashable | None = None,
    stays: dict[Hashable, Fraction] | None = None,
) -> dict[int, Fraction]:
    """独立有限步递推：返回 sum_{t=0}^H P(T>t)，它是 E[T] 的尾和下界。

    这里自行由支路电导和停留权重归一化转移概率，不调用被测随机游走模块。
    """
    stays = stays or {}
    adj = {v: {} for v in nodes}
    for u, v, g in branches:
        g = Fraction(g)
        adj[u][v] = adj[u].get(v, Fraction(0)) + g
        adj[v][u] = adj[v].get(u, Fraction(0)) + g

    active = [v for v in nodes if v != target]
    q = {v: Fraction(1) for v in active}
    wanted = start if start is not None else nodes[0]
    result = {}
    cumulative = Fraction(0)
    for horizon in range(max(horizons) + 1):
        cumulative += q[wanted]
        if horizon in horizons:
            result[horizon] = cumulative
        nxt = {}
        for v in active:
            total = sum(adj[v].values(), Fraction(0)) + Fraction(stays.get(v, 0))
            nxt[v] = (
                Fraction(stays.get(v, 0)) / total * q[v]
                + sum(g / total * q[y] for y, g in adj[v].items() if y != target)
            )
        q = nxt
    return result
