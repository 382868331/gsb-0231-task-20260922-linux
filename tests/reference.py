"""独立参考实现（仅供测试）。

本模块**不导入也不调用** ``resistor_elim`` 的任何核心算法：拉普拉斯由
支路表独立拼装，线性求解用独立编写的高斯消元（带行交换的增广矩阵消元），
期望值另有串/并联解析式；交流部分用 Python 内置 ``complex`` 浮点独立
拼装修正结点方程。被测库与参考之间只允许比较结论，不允许共享计算。
"""

from __future__ import annotations

import cmath
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
# 交流（R/L/C + MNA）独立参考：Python 内置 complex，独立拼装与消元
# ===================================================================== #
def _gauss_solve_complex(a, b):
    """解复线性方程组 a x = b（独立的前向消元+回代，带行交换）。"""
    n = len(a)
    m = [list(a[r]) + [b[r]] for r in range(n)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if abs(m[r][col]) > 0.0), None)
        if pivot is None:
            raise ZeroDivisionError("参考交流求解器遇到奇异矩阵")
        m[col], m[pivot] = m[pivot], m[col]
        piv = m[col][col]
        m[col] = [x / piv for x in m[col]]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col]
            if f != 0.0:
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [m[i][n] for i in range(n)]


class RefACNetwork:
    """交流参考网络：节点表 + (类型, u, v, 值) 支路表。

    端口导纳由“逐端口施加单位相量电压、其余端口接地，解全图 MNA 后
    读取端口电流”的独立方法得到——与被测库的 Schur 消元路线完全不同
    （参考不做内部消元）。
    """

    def __init__(self, nodes, branches):
        # branches: ('r'|'c'|'l', u, v, value)
        self.nodes = tuple(nodes)
        self.idx = {v: i for i, v in enumerate(self.nodes)}
        self.branches = [(t, u, v, Fraction(val)) for t, u, v, val in branches]
        self._build_adj()

    def _build_adj(self):
        self.adj = {v: set() for v in self.nodes}
        for _t, u, v, _val in self.branches:
            self.adj[u].add(v)
            self.adj[v].add(u)

    def connected_components(self):
        seen, comps = set(), []
        for start in self.nodes:
            if start in seen:
                continue
            stack, comp = [start], {start}
            seen.add(start)
            while stack:
                x = stack.pop()
                for y in self.adj[x] - seen:
                    seen.add(y)
                    comp.add(y)
                    stack.append(y)
            comps.append(comp)
        return comps

    # ----------------------------------------------------------------- #
    def _component_mna(self, nodes, omega):
        """拼装一个组件的完整 MNA（节点电压变量 + 电感电流变量）。"""
        n = len(nodes)
        pos = {v: i for i, v in enumerate(nodes)}
        inds = [
            (k, u, v, val)
            for k, (t, u, v, val) in enumerate(self.branches)
            if t == "l" and u in pos
        ]
        size = n + len(inds)
        m = [[0j for _ in range(size)] for _ in range(size)]
        jw = 1j * omega
        for t, u, v, val in self.branches:
            if u not in pos:
                continue
            if t == "r":
                y = complex(1.0 / float(val))
            elif t == "c":
                y = jw * float(val)
            else:
                continue  # 电感走电流变量
            iu, iv = pos[u], pos[v]
            m[iu][iu] += y
            m[iv][iv] += y
            m[iu][iv] -= y
            m[iv][iu] -= y
        for k_local, (_gk, u, v, val) in enumerate(inds):
            row = n + k_local
            iu, iv = pos[u], pos[v]
            m[iu][row] += 1.0
            m[iv][row] -= 1.0
            m[row][iu] += 1.0
            m[row][iv] -= 1.0
            m[row][row] = -jw * float(val)
        return m, pos, inds

    def port_admittance(self, ports, omega):
        """逐列激励法独立求端口导纳矩阵（complex 浮点）。

        对每个端口 p：V_p=1，其余端口=0；对每个含端口组件分别固定端口
        电压、解内部节点+电感电流；端口电流由该端口所连支路电流求和。
        """
        ports = list(ports)
        np_ = len(ports)
        y = [[0j for _ in range(np_)] for _ in range(np_)]
        comps = self.connected_components()
        port_set = set(ports)
        for exc in range(np_):
            vport = {p: (1.0 if i == exc else 0.0) for i, p in enumerate(ports)}
            voltage = {}
            ind_currents = {}
            for comp in comps:
                cports = [p for p in ports if p in comp]
                if not cports:
                    continue  # 浮置组件
                cinterior = [v for v in self.nodes if v in comp and v not in port_set]
                mna, pos, inds = self._component_mna(
                    [v for v in self.nodes if v in comp], omega
                )
                # 把内部变量（内部节点电压 + 电感电流）作为未知量消去端口
                fixed = [pos[p] for p in cports]
                free_node = [pos[v] for v in cinterior]
                ind_rows = [len([v for v in self.nodes if v in comp]) + k
                            for k in range(len(inds))]
                free = free_node + ind_rows
                rhs = [0j for _ in free]
                for r_local, grow in enumerate(free):
                    for p in cports:
                        rhs[r_local] -= mna[grow][pos[p]] * vport[p]
                a = [[mna[grow][gcol] for gcol in free] for grow in free]
                sol = _gauss_solve_complex(a, rhs)
                for label in cports:
                    voltage[label] = vport[label]
                for v, x in zip(cinterior, sol[: len(cinterior)]):
                    voltage[v] = x
                for (k, u, v2, val), x in zip(inds, sol[len(cinterior):]):
                    ind_currents[k] = x
            # 端口电流：按支路定义独立累加
            cur = {p: 0j for p in ports}
            for k, (t, u, v, val) in enumerate(self.branches):
                if u not in cur and v not in cur:
                    continue
                if t == "r":
                    z = float(val)
                    i_uv = (voltage[u] - voltage[v]) / z
                elif t == "c":
                    i_uv = 1j * omega * float(val) * (voltage[u] - voltage[v])
                else:
                    i_uv = ind_currents[k]  # u→v
                if u in cur:
                    cur[u] += i_uv
                if v in cur:
                    cur[v] -= i_uv
            for r in range(np_):
                y[r][exc] = cur[ports[r]]
        return y

    def solve(self, ports, vport, omega):
        """给定全部端口相量电压，独立求节点电压与各支路电流。"""
        comps = self.connected_components()
        port_set = set(ports)
        voltage = dict(vport)
        ind_currents = {}
        for comp in comps:
            cports = [p for p in ports if p in comp]
            if not cports:
                continue
            cnodes = [v for v in self.nodes if v in comp]
            cinterior = [v for v in cnodes if v not in port_set]
            mna, pos, inds = self._component_mna(cnodes, omega)
            fixed = [pos[p] for p in cports]
            free_node = [pos[v] for v in cinterior]
            base_n = len(cnodes)
            ind_rows = [base_n + k for k in range(len(inds))]
            free = free_node + ind_rows
            rhs = [0j for _ in free]
            for r_local, grow in enumerate(free):
                for p in cports:
                    rhs[r_local] -= mna[grow][pos[p]] * vport[p]
            a = [[mna[grow][gcol] for gcol in free] for grow in free]
            sol = _gauss_solve_complex(a, rhs)
            for v, x in zip(cinterior, sol[: len(cinterior)]):
                voltage[v] = x
            for (k, u, v2, val), x in zip(inds, sol[len(cinterior):]):
                ind_currents[k] = x
        branch_i = []
        for k, (t, u, v, val) in enumerate(self.branches):
            if u not in voltage or v not in voltage:
                branch_i.append(None)
            elif t == "r":
                branch_i.append((voltage[u] - voltage[v]) / float(val))
            elif t == "c":
                branch_i.append(1j * omega * float(val) * (voltage[u] - voltage[v]))
            else:
                branch_i.append(ind_currents[k])
        return voltage, branch_i


# --------------------------------------------------------------------- #
# 交流解析式（串/并联）
# --------------------------------------------------------------------- #
def series_impedance_admittance(parts, omega):
    """串联支路的二端口导纳。parts 为 ``('r'|'c'|'l', value)``。"""
    z = 0j
    for t, val in parts:
        val = float(Fraction(val))
        if t == "r":
            z += val
        elif t == "l":
            z += 1j * omega * val
        elif t == "c":
            z += 1.0 / (1j * omega * val)
        else:  # pragma: no cover
            raise ValueError(t)
    y = 1.0 / z
    return [[y, -y], [-y, y]]


def parallel_admittance(parts, omega):
    """并联支路的二端口导纳。parts 为 ``('r'|'c'|'l', value)``。"""
    y = 0j
    for t, val in parts:
        val = float(Fraction(val))
        if t == "r":
            y += 1.0 / val
        elif t == "c":
            y += 1j * omega * val
        elif t == "l":
            y += 1.0 / (1j * omega * val)
        else:  # pragma: no cover
            raise ValueError(t)
    return [[y, -y], [-y, y]]


def rc_analytic(r, c, omega):
    """单 RC 并联二端口：Y = 1/R + jωC（complex 参考值）。"""
    return 1.0 / float(Fraction(r)) + 1j * float(Fraction(omega)) * float(Fraction(c))


def cmath_is_close(a, b, tol=1e-10):
    return cmath.isclose(a, b, rel_tol=tol, abs_tol=tol)
