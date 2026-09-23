"""Schur（星-网/高斯）消元：内部结点等效消除与端口 DtN 矩阵。

对每个**与端口相连的连通组件**，把节点划分为端口 ``P`` 与内部 ``I``：

    K = [ K_PP  K_PI ]
        [ K_IP  K_II ]

内部节点满足 KCL ``K_IP V_P + K_II V_I = 0``，故

    V_I = -K_II^{-1} K_IP V_P
    I_P = (K_PP - K_PI K_II^{-1} K_IP) V_P = S V_P

``S`` 即端口 Dirichlet-to-Neumann（电流/电势）矩阵。``-K_II^{-1} K_IP``
作为消元记录保留，给定端口电势即可精确恢复内部电势、支路电流与功率。

不含任何端口的组件无法被激励（电势只能确定到组件内的常数），单独标记为
``floating``，绝不并入或丢弃；孤立节点同样显式保留。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Hashable, Iterable, Mapping, Sequence

from .errors import PortError, PotentialError
from .linalg import Mat
from .network import Network, potential_scalar


@dataclass(frozen=True)
class SchurRecord:
    """单个组件一次 Schur 消元的完整记录（全部为精确 ``Fraction``）。"""

    port_labels: tuple[Hashable, ...]
    interior_labels: tuple[Hashable, ...]
    k_pp: list[list[Fraction]]
    k_pi: list[list[Fraction]]
    k_ii: list[list[Fraction]]
    kii_inverse: list[list[Fraction]]
    #: ``backmap = -K_II^{-1} K_IP``：内部电势 = backmap @ 端口电势
    backmap: list[list[Fraction]]

    def interior_potentials(self, port_v: Sequence[Fraction]) -> list[Fraction]:
        if len(port_v) != len(self.port_labels):
            raise PotentialError(
                f"端口电势数量 {len(port_v)} 与本组件端口数 "
                f"{len(self.port_labels)} 不符"
            )
        return Mat.matvec(self.backmap, list(port_v))


@dataclass
class ComponentReduction:
    """一个连通组件的消元结果。"""

    labels: tuple[Hashable, ...]
    port_labels: tuple[Hashable, ...]
    interior_labels: tuple[Hashable, ...]
    #: 组件局部端口 DtN（按 ``port_labels`` 排序）
    local_dtn: list[list[Fraction]]
    record: SchurRecord | None
    floating: bool

    def is_ported(self) -> bool:
        return not self.floating


@dataclass
class Solution:
    """给定端口电势后的完整电路响应（精确有理）。"""

    network: Network
    #: 所有可确定节点（端口组件中的端口与内部节点）电势
    potentials: dict[Any, Fraction]
    #: 每个端口注入网络的电流（流入为正）
    port_currents: dict[Any, Fraction]
    #: 与 network.branches 对齐；浮接组件中的支路为 None
    branch_currents: list[Fraction | None]
    #: 与 network.branches 对齐；支路耗散功率 g*(du-dv)^2；浮接组件为 None
    branch_power: list[Fraction | None]
    #: 浮接组件节点（电势不可由端口激励确定）
    undetermined_nodes: tuple[Hashable, ...]
    #: 端口注入功率 sum_p V_p I_p
    port_power: Fraction
    #: 端口组件所有支路的耗散功率之和（精确等于 port_power）
    dissipated_power: Fraction

    def determined_branch_power(self) -> Fraction:
        return sum((p for p in self.branch_power if p is not None), Fraction(0))


@dataclass
class Reduction:
    """整张网络的等效消元结果。"""

    network: Network
    ports: tuple[Hashable, ...]
    dtn: list[list[Fraction]]
    components: list[ComponentReduction] = field(default_factory=list)

    @property
    def ported_components(self) -> list[ComponentReduction]:
        return [c for c in self.components if not c.floating]

    @property
    def floating_components(self) -> list[tuple[Hashable, ...]]:
        return [c.labels for c in self.components if c.floating]

    # ------------------------------------------------------------------ #
    def _coerce_potentials(
        self, potentials: Sequence[Any] | Mapping[Any, Any]
    ) -> list[Fraction]:
        if isinstance(potentials, Mapping):
            missing = [p for p in self.ports if p not in potentials]
            unknown = [k for k in potentials if k not in set(self.ports)]
            if missing or unknown:
                raise PotentialError(
                    f"端口电势映射必须恰好覆盖全部端口；"
                    f"缺失={missing} 多余={unknown}"
                )
            raw = [potentials[p] for p in self.ports]
        else:
            if isinstance(potentials, (str, bytes)):
                raise PotentialError("端口电势必须是序列或映射，不能是字符串")
            raw = list(potentials)
            if len(raw) != len(self.ports):
                raise PotentialError(
                    f"端口电势数量 {len(raw)} 与端口数 {len(self.ports)} 不符"
                )
        out: list[Fraction] = []
        for label, value in zip(self.ports, raw):
            try:
                out.append(potential_scalar(value, f"端口 {label!r} 的电势"))
            except (ValueError, TypeError) as exc:
                raise PotentialError(str(exc), node=label) from exc
        return out

    def solve(self, potentials: Sequence[Any] | Mapping[Any, Any]) -> Solution:
        """给定端口电势，精确恢复内部电势、电流与功率。

        浮接组件节点不出现在 ``potentials`` 中（其电势只能确定到组件常数），
        对应支路电流/功率返回 ``None``；它们既不耗散功率也不接受端口注入。
        """
        vport = self._coerce_potentials(potentials)
        iport = Mat.matvec(self.dtn, vport)

        node_v: dict[Any, Fraction] = {}
        for p, v in zip(self.ports, vport):
            node_v[p] = v
        comp_of: dict[Any, ComponentReduction] = {}
        for comp in self.components:
            if comp.floating:
                continue
            local_v = [vport[self.ports.index(p)] for p in comp.port_labels]
            inner_v = comp.record.interior_potentials(local_v) if comp.record else []
            for label, v in zip(comp.port_labels, local_v):
                node_v[label] = v
            for label, v in zip(comp.interior_labels, inner_v):
                node_v[label] = v
            for label in comp.labels:
                comp_of[label] = comp

        floating_nodes = tuple(
            label
            for comp in self.components
            if comp.floating
            for label in comp.labels
        )

        branch_i: list[Fraction | None] = []
        branch_p: list[Fraction | None] = []
        for u, v, g in self.network.branches:
            if u in floating_nodes or v in floating_nodes:
                branch_i.append(None)
                branch_p.append(None)
                continue
            du = node_v[u] - node_v[v]
            i = g * du  # 自 u 流向 v
            branch_i.append(i)
            branch_p.append(g * du * du)

        port_currents = {p: i for p, i in zip(self.ports, iport)}
        port_power = sum((v * i for v, i in zip(vport, iport)), Fraction(0))
        dissipated = sum(
            (p for p in branch_p if p is not None), Fraction(0)
        )
        return Solution(
            network=self.network,
            potentials=node_v,
            port_currents=port_currents,
            branch_currents=branch_i,
            branch_power=branch_p,
            undetermined_nodes=floating_nodes,
            port_power=port_power,
            dissipated_power=dissipated,
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
        network.index(p)  # 未注册 → NodeError（可定位）
    return port_list


def schur_eliminate(
    network: Network,
    ports: Iterable[Hashable],
) -> Reduction:
    """对网络做定向结点等效消元。

    参数
    ----
    network:
        :class:`Network`（节点数 ≤ 30，正电导子集）。
    ports:
        端口节点标签序列（非空、无重复、均为已注册节点）。顺序即全局
        DtN 矩阵的行列顺序。

    返回
    ----
    Reduction
        全局端口 DtN（未被任何电导连接的端口行为零行零列）、逐组件消元
        记录（含 Schur 补因子，用于内部量恢复）以及浮接组件清单。

    本函数不修改 ``network``；输入非法时抛出 :class:`NetworkError` 子类，
    不会产生部分结果。
    """
    if not isinstance(network, Network):
        raise TypeError("network 必须是 resistor_elim.Network 实例")
    port_list = _validate_ports(network, ports)
    port_set = set(port_list)
    n = len(network)
    k_full = network.laplacian()

    component_results: list[ComponentReduction] = []
    # 全局 DtN：先置零，按组件把局部 DtN 嵌入到对应端口行列
    dtn = [[Fraction(0) for _ in port_list] for _ in port_list]

    for comp_labels in network.components():
        # 保持 ports 参数给定的全局顺序
        comp_ports = [p for p in port_list if p in comp_labels]
        is_floating = not comp_ports
        if is_floating:
            component_results.append(
                ComponentReduction(
                    labels=comp_labels,
                    port_labels=(),
                    interior_labels=(),
                    local_dtn=[],
                    record=None,
                    floating=True,
                )
            )
            continue

        comp_interior = [x for x in comp_labels if x not in port_set]
        p_idx = [network.index(p) for p in comp_ports]
        i_idx = [network.index(x) for x in comp_interior]

        k_pp = Mat.submatrix(k_full, p_idx, p_idx)
        k_pi = Mat.submatrix(k_full, p_idx, i_idx)
        k_ip = Mat.submatrix(k_full, i_idx, p_idx)
        k_ii = Mat.submatrix(k_full, i_idx, i_idx)

        if comp_interior:
            # 连通组件内每个内部节点至少有一条电导相连，且至少存在一个
            # 端口节点，故 K_II 正定、精确可逆。
            kii_inv = Mat.inverse(k_ii)
            local_dtn = [
                [
                    k_pp[r][c]
                    - sum(
                        (
                            k_pi[r][t] * kii_inv[t][s] * k_ip[s][c]
                            for t in range(len(i_idx))
                            for s in range(len(i_idx))
                        ),
                        Fraction(0),
                    )
                    for c in range(len(p_idx))
                ]
                for r in range(len(p_idx))
            ]
            backmap = [
                [
                    -sum(
                        (kii_inv[t][s] * k_ip[s][c] for s in range(len(i_idx))),
                        Fraction(0),
                    )
                    for c in range(len(p_idx))
                ]
                for t in range(len(i_idx))
            ]
            record = SchurRecord(
                port_labels=tuple(comp_ports),
                interior_labels=tuple(comp_interior),
                k_pp=k_pp,
                k_pi=k_pi,
                k_ii=k_ii,
                kii_inverse=kii_inv,
                backmap=backmap,
            )
        else:
            local_dtn = k_pp
            record = SchurRecord(
                port_labels=tuple(comp_ports),
                interior_labels=(),
                k_pp=k_pp,
                k_pi=[],
                k_ii=[],
                kii_inverse=[],
                backmap=[],
            )

        for a, pa in enumerate(comp_ports):
            ga = port_list.index(pa)
            for b, pb in enumerate(comp_ports):
                gb = port_list.index(pb)
                dtn[ga][gb] = local_dtn[a][b]

        component_results.append(
            ComponentReduction(
                labels=comp_labels,
                port_labels=tuple(comp_ports),
                interior_labels=tuple(comp_interior),
                local_dtn=local_dtn,
                record=record,
                floating=False,
            )
        )

    return Reduction(
        network=network,
        ports=tuple(port_list),
        dtn=dtn,
        components=component_results,
    )
