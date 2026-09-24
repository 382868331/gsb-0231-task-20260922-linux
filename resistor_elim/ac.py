"""修正结点分析（MNA）+ Schur 消元：给定非零有限角频率求端口复导纳。

在原电阻网络上加入电容 ``C`` 与电感 ``L`` 后，在角频率 ``ω ≠ 0`` 下做
正弦稳态相量分析：

* 电阻支路导纳 ``g = 1/R``，电容支路导纳 ``jωC``——两者直接戳入结点
  导纳块（``K[i,i] += y``、``K[i,j] -= y``）；
* 每条电感引入一个支路电流变量，戳入修正方程

      V_u - V_v - jωL·i_L = 0

  并在端点 KCL 中以 ``+i_L / -i_L`` 出现。

对每个**含端口的连通组件**，把变量分为端口结点电压 ``P`` 与内部变量
``I``（内部结点电压 + 该组件全部电感电流）：

    Y = [ A  B ]     I_P = (A - B D^{-1} C) V_P = Y_port V_P
        [ C  D ]

``ω ≠ 0`` 且所有 R/L/C 严格为正时，``D`` 通常可逆；两类例外被明确
报告：不含端口的组件标记为 *floating*（不并入端口模型）；真正奇异的
``D``（例如两个端口之间仅串一个 LC 且恰好 ``ω²LC = 1`` 的零阻抗谐振
支路）抛 :class:`SingularSystemError`，绝不用容差掩盖。

全程使用实部/虚部均为 ``Fraction`` 的 :class:`~resistor_elim.cx.QComplex`
精确运算，与解析公式可逐值精确对照。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Hashable, Iterable, Mapping, Sequence

from .cx import CxMat, J, QComplex
from .errors import (
    FrequencyError,
    PortError,
    PotentialError,
    SingularSystemError,
)
from .network import Network


# ---------------------------------------------------------------------- #
# 输入校验
# ---------------------------------------------------------------------- #
def angular_frequency(value: Any) -> Fraction:
    """把角频率校验并转换为非零有限 ``Fraction``。

    接受 ``int`` / ``Fraction`` / 有限 ``float``（按二进制值精确转分数）；
    拒绝 ``bool``、复数、字符串、``NaN``/``Infinity`` 与零。负频率在数学
    上合法（结果为正频率的共轭，满足 ``Y(-ω) = conj(Y(ω))``），允许使用。
    """
    if isinstance(value, bool):
        raise TypeError("角频率不得为 bool")
    if isinstance(value, complex):
        raise FrequencyError(f"角频率必须为实数，收到 complex {value!r}")
    if isinstance(value, float):
        if math.isnan(value):
            raise FrequencyError("角频率不得为 NaN")
        if math.isinf(value):
            raise FrequencyError("角频率不得为 Infinity（必须有限）")
        w = Fraction(value)
    elif isinstance(value, Fraction):
        w = value
    elif isinstance(value, int):
        w = Fraction(value)
    else:
        raise FrequencyError(
            f"角频率必须是 int、float（有限）或 Fraction，"
            f"收到 {type(value).__name__}"
        )
    if w == 0:
        raise FrequencyError(
            "角频率不得为零：本模块只做非零频率点的相量分析，"
            "零频（直流）极限不在契约内；纯电阻直流结果请用 schur_eliminate"
        )
    return w


def phasor_scalar(value: Any, what: str) -> QComplex:
    """把一个端口相量转为 :class:`QComplex`。

    实数接受 ``int`` / ``Fraction`` / 有限 ``float``；复数接受内置
    ``complex``（实虚部均有限）或 :class:`QComplex`。``bool``、
    ``NaN``/``Infinity``、字符串等被拒绝。
    """
    if isinstance(value, bool):
        raise TypeError(f"{what} 不得为 bool")
    if isinstance(value, QComplex):
        return value
    if isinstance(value, complex):
        if math.isnan(value.real) or math.isnan(value.imag):
            raise ValueError(f"{what} 不得含 NaN")
        if math.isinf(value.real) or math.isinf(value.imag):
            raise ValueError(f"{what} 不得含 Infinity")
        return QComplex(Fraction(value.real), Fraction(value.imag))
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError(f"{what} 不得为 NaN")
        if math.isinf(value):
            raise ValueError(f"{what} 不得为 Infinity")
        return QComplex(Fraction(value), Fraction(0))
    if isinstance(value, (int, Fraction)):
        return QComplex(Fraction(value), Fraction(0))
    raise ValueError(
        f"{what} 必须是 int、float（有限）、complex、Fraction 或 QComplex，"
        f"收到 {type(value).__name__}"
    )


# ---------------------------------------------------------------------- #
# 结果数据结构
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class ACSchurRecord:
    """单个组件一次 MNA + Schur 消元的完整记录（全部为精确 :class:`QComplex`）。"""

    port_labels: tuple[Hashable, ...]
    interior_labels: tuple[Hashable, ...]
    #: 电感电流变量顺序：``(全局电感序号, u, v, L)``，电流正方向 u→v
    inductor_entries: tuple[tuple[int, Hashable, Hashable, Fraction], ...]
    a_pp: list[list[QComplex]]
    a_pi: list[list[QComplex]]
    a_ip: list[list[QComplex]]
    d_ii: list[list[QComplex]]
    d_inverse: list[list[QComplex]]
    #: ``内部变量 = backmap @ 端口电压``（行序：内部结点电压，再电感电流）
    backmap: list[list[QComplex]]


@dataclass
class ACComponentReduction:
    """一个连通组件的交流消元结果。"""

    labels: tuple[Hashable, ...]
    port_labels: tuple[Hashable, ...]
    interior_labels: tuple[Hashable, ...]
    inductor_entries: tuple[tuple[int, Hashable, Hashable, Fraction], ...]
    #: 组件局部端口导纳（按 ``port_labels`` 排序）
    local_y: list[list[QComplex]]
    record: ACSchurRecord | None
    floating: bool

    def is_ported(self) -> bool:
        return not self.floating


@dataclass
class ACSolution:
    """给定端口相量电压后的完整正弦稳态响应（精确复数）。"""

    network: Network
    omega: Fraction
    #: 可确定节点（端口组件的端口与内部结点）相量电压
    potentials: dict[Any, QComplex]
    #: 各端口注入网络的相量电流（流入为正）
    port_currents: dict[Any, QComplex]
    #: 与 network.branches 对齐的电阻支路电流（u→v）；浮接组件为 None
    branch_currents: list[QComplex | None]
    #: 与 network.capacitor_branches 对齐；浮接组件为 None
    capacitor_currents: list[QComplex | None]
    #: 与 network.inductor_branches 对齐的电感支路电流；浮接组件为 None
    inductor_currents: list[QComplex | None]
    #: 浮接组件节点（电压只能确定到组件常数）
    undetermined_nodes: tuple[Hashable, ...]
    #: 端口视在功率 S = 1/2·Σ V_p·conj(I_p)（相量按峰值振幅约定）
    port_apparent_power: QComplex
    #: 电阻支路平均耗散功率 1/2·Σ g|ΔV|²（精确等于端口有功功率）
    average_dissipated_power: Fraction

    @property
    def average_port_power(self) -> Fraction:
        """端口注入的平均有功功率 ``Re(S)``。"""
        return self.port_apparent_power.re


@dataclass
class AdmittanceResult:
    """整张网络在某一角频率下的端口复导纳结果。"""

    network: Network
    ports: tuple[Hashable, ...]
    omega: Fraction
    #: 端口复导纳矩阵 Y（行列顺序同 ``ports``）
    y: list[list[QComplex]]
    components: list[ACComponentReduction] = field(default_factory=list)

    @property
    def ported_components(self) -> list[ACComponentReduction]:
        return [c for c in self.components if not c.floating]

    @property
    def floating_components(self) -> list[tuple[Hashable, ...]]:
        return [c.labels for c in self.components if c.floating]

    # ------------------------------------------------------------------ #
    def _coerce_phasors(
        self, potentials: Sequence[Any] | Mapping[Any, Any]
    ) -> list[QComplex]:
        if isinstance(potentials, Mapping):
            missing = [p for p in self.ports if p not in potentials]
            unknown = [k for k in potentials if k not in set(self.ports)]
            if missing or unknown:
                raise PotentialError(
                    f"端口相量映射必须恰好覆盖全部端口；"
                    f"缺失={missing} 多余={unknown}"
                )
            raw = [potentials[p] for p in self.ports]
        else:
            if isinstance(potentials, (str, bytes)):
                raise PotentialError("端口相量必须是序列或映射，不能是字符串")
            raw = list(potentials)
            if len(raw) != len(self.ports):
                raise PotentialError(
                    f"端口相量数量 {len(raw)} 与端口数 {len(self.ports)} 不符"
                )
        out: list[QComplex] = []
        for label, value in zip(self.ports, raw):
            try:
                out.append(phasor_scalar(value, f"端口 {label!r} 的相量电压"))
            except (ValueError, TypeError) as exc:
                raise PotentialError(str(exc), node=label) from exc
        return out

    def solve(self, potentials: Sequence[Any] | Mapping[Any, Any]) -> ACSolution:
        """给定端口相量电压，恢复内部结点电压、各元件支路电流与功率。

        浮接组件不被激励：节点不出现在 :attr:`ACSolution.potentials`，
        对应支路电流为 ``None``。
        """
        vp = self._coerce_phasors(potentials)
        ip = CxMat.matvec(self.y, vp)

        node_v: dict[Any, QComplex] = {p: v for p, v in zip(self.ports, vp)}
        # 全局电感序号 → 电流相量
        ind_current: dict[int, QComplex] = {}
        floating_nodes: set[Any] = set()
        for comp in self.components:
            if comp.floating:
                floating_nodes.update(comp.labels)
                continue
            local_vp = [vp[self.ports.index(p)] for p in comp.port_labels]
            if comp.record is not None and comp.record.backmap:
                inner = CxMat.matvec(comp.record.backmap, local_vp)
            else:
                inner = []
            n_v = len(comp.interior_labels)
            for label, v in zip(comp.interior_labels, inner[:n_v]):
                node_v[label] = v
            for entry, cur in zip(comp.inductor_entries, inner[n_v:]):
                ind_current[entry[0]] = cur

        # 电阻支路电流
        branch_i: list[QComplex | None] = []
        for u, v, g in self.network.branches:
            if u in floating_nodes or v in floating_nodes:
                branch_i.append(None)
            else:
                branch_i.append(QComplex(g) * (node_v[u] - node_v[v]))

        # 电容支路电流 jωC·ΔV
        cap_i: list[QComplex | None] = []
        jw = J * self.omega
        for u, v, c in self.network.capacitor_branches:
            if u in floating_nodes or v in floating_nodes:
                cap_i.append(None)
            else:
                cap_i.append(jw * QComplex(c) * (node_v[u] - node_v[v]))

        # 电感支路电流（MNA 电流变量的解）
        ind_i: list[QComplex | None] = [
            ind_current.get(k) for k in range(len(self.network.inductor_branches))
        ]

        apparent = sum(
            (v * i.conj() for v, i in zip(vp, ip)), QComplex()
        ) / QComplex(2)
        dissipated = Fraction(0)
        for (u, v, g), cur in zip(self.network.branches, branch_i):
            if cur is not None:
                dv = node_v[u] - node_v[v]
                dissipated += Fraction(1, 2) * g * dv.abs_sq()

        return ACSolution(
            network=self.network,
            omega=self.omega,
            potentials=node_v,
            port_currents={p: i for p, i in zip(self.ports, ip)},
            branch_currents=branch_i,
            capacitor_currents=cap_i,
            inductor_currents=ind_i,
            undetermined_nodes=tuple(floating_nodes),
            port_apparent_power=apparent,
            average_dissipated_power=dissipated,
        )


# ---------------------------------------------------------------------- #
def _validate_ports(network: Network, ports: Iterable[Hashable]) -> list[Hashable]:
    if isinstance(ports, (str, bytes)):
        raise PortError("ports 必须是标签序列（单端口也请写成 [label]）")
    try:
        port_list = list(ports)
    except TypeError as exc:
        raise PortError("ports 必须是节点标签的可迭代对象") from exc
    if not port_list:
        raise PortError("至少需要指定一个端口")
    seen: set[Hashable] = set()
    for p in port_list:
        if isinstance(p, bool):
            raise PortError("端口标签不得为 bool", node=p)
        if p in seen:
            raise PortError(f"端口重复：{p!r}", node=p)
        seen.add(p)
        network.index(p)
    return port_list


def _build_component_mna(
    network: Network,
    comp_labels: tuple[Hashable, ...],
    comp_inductors: list[tuple[int, Hashable, Hashable, Fraction]],
    y_full: list[list[QComplex]],
    omega: Fraction,
) -> tuple[list[list[QComplex]], list[Hashable]]:
    """拼装一个组件的 MNA 矩阵（独立于 Schur 分块本身的纯组装步骤）。

    变量顺序：组件结点电压（按 ``comp_labels``）在前，电感支路电流在后。
    """
    n = len(comp_labels)
    m = CxMat.zeros(n + len(comp_inductors), n + len(comp_inductors))
    pos = {label: i for i, label in enumerate(comp_labels)}

    # R + C 结点导纳块（从全网络矩阵取组件子块，避免与网络模块重复记账）
    for a, ua in enumerate(comp_labels):
        ia = network.index(ua)
        for b, ub in enumerate(comp_labels):
            m[a][b] = y_full[ia][network.index(ub)]

    # 电感：KCL 端点 ±1，KVL 行 ±1 与 −jωL
    for k, (_, u, v, ell) in enumerate(comp_inductors):
        row = n + k
        iu, iv = pos[u], pos[v]
        m[iu][row] += QComplex(1)
        m[iv][row] -= QComplex(1)
        m[row][iu] += QComplex(1)
        m[row][iv] -= QComplex(1)
        m[row][row] = -J * omega * ell
    return m, list(comp_labels)


def ac_admittance(
    network: Network,
    ports: Iterable[Hashable],
    omega: Any,
) -> AdmittanceResult:
    """在角频率 ``omega`` 下求网络的端口复导纳（修正结点方程 + Schur 消元）。

    参数
    ----
    network:
        :class:`Network`，可含电阻、电容（:meth:`~Network.add_capacitor`）、
        电感（:meth:`~Network.add_inductor`），元件值均严格为正。
    ports:
        端口节点标签序列（非空、无重复、均已注册）；顺序即导纳矩阵的
        行列顺序。
    omega:
        非零有限角频率（``int`` / ``Fraction`` / 有限 ``float``）。
        零频被明确拒绝（直流极限不在契约内）。

    返回
    ----
    AdmittanceResult
        端口复导纳矩阵（精确 :class:`QComplex`）、逐组件消元记录、
        浮置组件清单，以及 :meth:`~AdmittanceResult.solve` 相量求解入口。

    异常
    ----
    FrequencyError
        ``omega`` 为零、非有限或类型非法。
    SingularSystemError
        含端口组件的内部 MNA 块奇异（例如零阻抗 LC 串联谐振）。
    NetworkError 子类
        端口非法等（与直流接口同一套错误语义）。

    本函数不修改 ``network``。
    """
    if not isinstance(network, Network):
        raise TypeError("network 必须是 resistor_elim.Network 实例")
    port_list = _validate_ports(network, ports)
    w = angular_frequency(omega)  # 先校验频率，再做任何计算
    port_set = set(port_list)

    y_full = network.nodal_admittance(w)

    # 把电感按组件归类（记录其在 network.inductor_branches 中的全局序号）
    inds_by_comp: dict[int, list[tuple[int, Hashable, Hashable, Fraction]]] = {}
    comp_index_of_node: dict[Hashable, int] = {}
    comps = network.components()
    for ci, labels in enumerate(comps):
        for label in labels:
            comp_index_of_node[label] = ci
    for gi, (u, v, ell) in enumerate(network.inductor_branches):
        ci = comp_index_of_node[u]
        assert comp_index_of_node[v] == ci  # 组件归类对每条边自洽
        inds_by_comp.setdefault(ci, []).append((gi, u, v, ell))

    y_port = CxMat.zeros(len(port_list), len(port_list))
    results: list[ACComponentReduction] = []

    for ci, comp_labels in enumerate(comps):
        comp_ports = [p for p in port_list if p in comp_labels]
        comp_inds = inds_by_comp.get(ci, [])

        if not comp_ports:
            results.append(
                ACComponentReduction(
                    labels=comp_labels,
                    port_labels=(),
                    interior_labels=(),
                    inductor_entries=(),
                    local_y=[],
                    record=None,
                    floating=True,
                )
            )
            continue

        comp_interior = [x for x in comp_labels if x not in port_set]
        mna, var_order = _build_component_mna(
            network, comp_labels, comp_inds, y_full, w
        )
        pos = {label: i for i, label in enumerate(var_order)}
        n_nodes = len(var_order)
        p_rows = [pos[p] for p in comp_ports]
        i_rows = [pos[x] for x in comp_interior] + [
            n_nodes + k for k in range(len(comp_inds))
        ]

        a_pp = CxMat.submatrix(mna, p_rows, p_rows)
        a_pi = CxMat.submatrix(mna, p_rows, i_rows)
        a_ip = CxMat.submatrix(mna, i_rows, p_rows)
        d_ii = CxMat.submatrix(mna, i_rows, i_rows)

        if i_rows:
            try:
                d_inv = CxMat.inverse(d_ii)
            except SingularSystemError as exc:
                raise SingularSystemError(
                    f"组件 {comp_labels} 的内部修正结点系统在 ω={w} 处奇异"
                    "（典型原因：仅含 L/C 的零阻抗串联谐振支路 ω²LC=1，"
                    "或退化的浮置内部结构）；请在该支路中加入电阻、"
                    "避开谐振频率或重新划分端口",
                    edge=tuple(comp_ports),
                ) from exc
            pi_dinv = [[sum((a_pi[r][t] * d_inv[t][s] for t in range(len(i_rows))),
                            QComplex())
                        for s in range(len(i_rows))]
                       for r in range(len(p_rows))]
            local_y = [
                [
                    a_pp[r][c]
                    - sum(
                        (pi_dinv[r][s] * a_ip[s][c] for s in range(len(i_rows))),
                        QComplex(),
                    )
                    for c in range(len(p_rows))
                ]
                for r in range(len(p_rows))
            ]
            backmap = [
                [
                    -sum((d_inv[t][s] * a_ip[s][c] for s in range(len(i_rows))),
                         QComplex())
                    for c in range(len(p_rows))
                ]
                for t in range(len(i_rows))
            ]
            record = ACSchurRecord(
                port_labels=tuple(comp_ports),
                interior_labels=tuple(comp_interior),
                inductor_entries=tuple(comp_inds),
                a_pp=a_pp,
                a_pi=a_pi,
                a_ip=a_ip,
                d_ii=d_ii,
                d_inverse=d_inv,
                backmap=backmap,
            )
        else:
            local_y = a_pp
            record = ACSchurRecord(
                port_labels=tuple(comp_ports),
                interior_labels=(),
                inductor_entries=tuple(comp_inds),
                a_pp=a_pp,
                a_pi=[],
                a_ip=[],
                d_ii=[],
                d_inverse=[],
                backmap=[],
            )

        for a, pa in enumerate(comp_ports):
            ga = port_list.index(pa)
            for b, pb in enumerate(comp_ports):
                gb = port_list.index(pb)
                y_port[ga][gb] = local_y[a][b]

        results.append(
            ACComponentReduction(
                labels=comp_labels,
                port_labels=tuple(comp_ports),
                interior_labels=tuple(comp_interior),
                inductor_entries=tuple(comp_inds),
                local_y=local_y,
                record=record,
                floating=False,
            )
        )

    return AdmittanceResult(
        network=network,
        ports=tuple(port_list),
        omega=w,
        y=y_port,
        components=results,
    )
