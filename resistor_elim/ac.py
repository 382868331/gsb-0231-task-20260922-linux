"""相量（正弦稳态）修正结点分析 + Schur 消元：端口复导纳矩阵。

在角频率 ``ω``（非零有限）下，无源 R/L/C 支路都有相量导纳形式：

    电阻 g : Y = g          电容 C : Y = jωC        电感 L : Y = 1/(jωL)

因此**修正结点方程（MNA）**在本库子集（无独立源、无受控源、非零频率，
电感也有有限导纳）中无需附加电流源行，直接退化为结点导纳系统

    Y(jω) V = I

对每个含端口的连通组件按端口 ``P`` / 内部 ``I`` 分块做 Schur 补：

    Y = [ Y_PP  Y_PI ]      I_P = (Y_PP - Y_PI Y_II^{-1} Y_IP) V_P
        [ Y_IP  Y_II ]

即端口复导纳（Dirichlet-to-Neumann）矩阵。全程使用精确有理复数
:class:`resistor_elim.linalg.CFrac`（实部、虚部分别为 ``Fraction``），
谐振奇异性由**精确零主元**判定，不使用数值容差。

不含端口的组件（浮置）沿用直流模块的契约：单独标记为 ``floating``，
不并入端口模型。纯 L/C 组件在谐振频率下端口导纳发散（``Y_II`` 奇异），
抛 :class:`SingularSystemError` 明确报告谐振组件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Hashable, Iterable, Mapping, Sequence

from .errors import PortError, PotentialError, SingularSystemError
from .linalg import CFrac, CMat
from .network import Network, frequency_scalar, phasor_scalar


@dataclass(frozen=True)
class ACSchurRecord:
    """单个组件一次相量 Schur 消元的完整记录（全部为精确 :class:`CFrac`）。"""

    port_labels: tuple[Hashable, ...]
    interior_labels: tuple[Hashable, ...]
    y_pp: list[list[CFrac]]
    y_pi: list[list[CFrac]]
    y_ii: list[list[CFrac]]
    yii_inverse: list[list[CFrac]]
    #: ``backmap = -Y_II^{-1} Y_IP``：内部相量电势 = backmap @ 端口电势
    backmap: list[list[CFrac]]

    def interior_potentials(self, port_v: Sequence[CFrac]) -> list[CFrac]:
        if len(port_v) != len(self.port_labels):
            raise PotentialError(
                f"端口电势数量 {len(port_v)} 与本组件端口数 "
                f"{len(self.port_labels)} 不符"
            )
        return CMat.matvec(self.backmap, list(port_v))


@dataclass
class ACComponentReduction:
    """一个连通组件的相量消元结果。"""

    labels: tuple[Hashable, ...]
    port_labels: tuple[Hashable, ...]
    interior_labels: tuple[Hashable, ...]
    #: 组件局部端口导纳（按 ``port_labels`` 排序）
    local_admittance: list[list[CFrac]]
    record: ACSchurRecord | None
    floating: bool

    def is_ported(self) -> bool:
        return not self.floating


@dataclass
class ACSolution:
    """给定端口相量电势后的完整正弦稳态响应（精确有理复数）。

    相量约定为**振幅相量**（峰值，非 RMS）。支路复功率
    ``S = 1/2 · ΔV · conj(I)``：电阻支路为非负实数（平均耗散），
    电容/电感支路为纯虚数（吞吐无功、平均耗散为 0）。
    """

    network: Network
    omega: Fraction
    #: 可确定节点（端口组件中的端口+内部节点）的相量电势
    potentials: dict[Any, CFrac]
    #: 每个端口注入网络的相量电流（流入为正）
    port_currents: dict[Any, CFrac]
    #: 与 network.branches_ac 对齐；浮置组件支路为 None
    branch_currents: list[CFrac | None]
    #: 与 network.branches_ac 对齐；支路复功率 1/2·ΔV·conj(I)；浮置为 None
    branch_power: list[CFrac | None]
    #: 浮置组件节点（电势只能确定到组件常数）
    undetermined_nodes: tuple[Hashable, ...]
    #: 端口复功率 1/2·Σ V_p·conj(I_p)（等于端口组件全部支路复功率之和）
    port_power: CFrac
    #: 电阻支路平均耗散功率之和（= port_power 的实部，精确相等）
    dissipated_power: Fraction

    def determined_branch_power(self) -> CFrac:
        return sum((p for p in self.branch_power if p is not None), CFrac())


@dataclass
class ACReduction:
    """整张网络在一个频率点上的相量等效消元结果。"""

    network: Network
    ports: tuple[Hashable, ...]
    omega: Fraction
    #: 全局端口复导纳矩阵，行列顺序同 ``ports``
    admittance: list[list[CFrac]]
    components: list[ACComponentReduction] = field(default_factory=list)

    @property
    def ported_components(self) -> list[ACComponentReduction]:
        return [c for c in self.components if not c.floating]

    @property
    def floating_components(self) -> list[tuple[Hashable, ...]]:
        return [c.labels for c in self.components if c.floating]

    # ------------------------------------------------------------------ #
    def _coerce_potentials(
        self, potentials: Sequence[Any] | Mapping[Any, Any]
    ) -> list[CFrac]:
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
        out: list[CFrac] = []
        for label, value in zip(self.ports, raw):
            try:
                out.append(phasor_scalar(value, f"端口 {label!r} 的相量电势"))
            except (ValueError, TypeError) as exc:
                raise PotentialError(str(exc), node=label) from exc
        return out

    def solve(self, potentials: Sequence[Any] | Mapping[Any, Any]) -> ACSolution:
        """给定端口相量电势，恢复内部相量电势、支路电流与复功率。

        浮置组件节点的电势不可确定，对应支路电流/功率返回 ``None``。
        """
        vport = self._coerce_potentials(potentials)
        iport = CMat.matvec(self.admittance, vport)

        node_v: dict[Any, CFrac] = {}
        for p, v in zip(self.ports, vport):
            node_v[p] = v
        for comp in self.components:
            if comp.floating:
                continue
            local_v = [vport[self.ports.index(p)] for p in comp.port_labels]
            inner_v = comp.record.interior_potentials(local_v) if comp.record else []
            for label, v in zip(comp.port_labels, local_v):
                node_v[label] = v
            for label, v in zip(comp.interior_labels, inner_v):
                node_v[label] = v

        floating_nodes = set(
            label
            for comp in self.components
            if comp.floating
            for label in comp.labels
        )

        w = self.omega
        branch_i: list[CFrac | None] = []
        branch_s: list[CFrac | None] = []
        for u, v, kind, value in self.network.branches_ac:
            if u in floating_nodes or v in floating_nodes:
                branch_i.append(None)
                branch_s.append(None)
                continue
            if kind == "R":
                y = CFrac(value)                      # 电导
            elif kind == "C":
                y = CFrac(Fraction(0), w * value)    # jωC
            else:  # "L"，value 为电感值
                y = CFrac(Fraction(0), -Fraction(1) / (w * value))
            dv = node_v[u] - node_v[v]
            i = y * dv                               # 自 u 流向 v
            s = dv * i.conjugate() / 2               # 支路吸收的复功率（振幅相量）
            branch_i.append(i)
            branch_s.append(s)

        port_power = sum(
            (v * i.conjugate() / 2 for v, i in zip(vport, iport)), CFrac()
        )
        dissipated = sum(
            (
                s.re
                for (_, _, kind, _), s in zip(self.network.branches_ac, branch_s)
                if s is not None and kind == "R"
            ),
            Fraction(0),
        )
        return ACSolution(
            network=self.network,
            omega=self.omega,
            potentials=node_v,
            port_currents={p: i for p, i in zip(self.ports, iport)},
            branch_currents=branch_i,
            branch_power=branch_s,
            undetermined_nodes=tuple(
                label
                for comp in self.components
                if comp.floating
                for label in comp.labels
            ),
            port_power=port_power,
            dissipated_power=dissipated,
        )


# ---------------------------------------------------------------------- #
def _resonant_error(
    resonant: list[tuple[Hashable, ...]], omega: Fraction
) -> SingularSystemError:
    labels = ", ".join(str(tuple(c)) for c in resonant)
    err = SingularSystemError(
        f"角频率 ω={omega} 下相量系统奇异：组件 [{labels}] 的内部结点导纳 "
        f"Y_II 不可逆——该无耗散纯 L/C 组件在当前频率谐振（串联阻抗为零、"
        f"导纳极点，或并联谐振）；请改用其他频率点或在组件中加入阻尼（电阻）"
    )
    err.resonant_components = resonant  # type: ignore[attr-defined]
    return err


def ac_schur_eliminate(
    network: Network,
    ports: Iterable[Hashable],
    omega: Any,
) -> ACReduction:
    """在角频率 ``omega`` 下对网络做相量 Schur 消元，求端口复导纳。

    参数
    ----
    network:
        :class:`Network`（节点数 ≤ 30；R/L/C 均为严格正的无源元件）。
    ports:
        端口节点标签序列（非空、无重复、均已注册）；顺序即端口导纳矩阵的
        行列顺序。
    omega:
        角频率 ``ω``（rad/s）：``int``/``Fraction`` 或有限非零 ``float``。
        本迭代只支持**非零有限**频率点；零频抛 :class:`FrequencyError`。
        负频率在数学上允许（结果为正频率的共轭）。

    返回
    ----
    ACReduction
        全局端口复导纳 ``Y(jω)``、逐组件消元记录（可由端口相量电势
        回溯内部电势/支路电流/复功率）与浮置组件清单。

    抛出
    ----
    FrequencyError
        ``omega`` 为零、NaN/Infinity 或类型不符。
    SingularSystemError
        含端口的纯 L/C 组件在该频率谐振（``Y_II`` 精确奇异）；异常对象带
        ``resonant_components`` 属性列出谐振组件标签。
    NetworkError 子类
        端口非法等输入错误。本函数不修改 ``network``，失败不留部分结果。
    """
    if not isinstance(network, Network):
        raise TypeError("network 必须是 resistor_elim.Network 实例")

    # 复用直流模块的端口校验（同一套契约）
    from .reduction import _validate_ports

    port_list = _validate_ports(network, ports)
    port_set = set(port_list)
    w = frequency_scalar(omega)
    y_full = network.admittance_matrix(w)

    component_results: list[ACComponentReduction] = []
    admittance = [[CFrac() for _ in port_list] for _ in port_list]
    resonant: list[tuple[Hashable, ...]] = []

    for comp_labels in network.components():
        comp_ports = [p for p in port_list if p in comp_labels]
        if not comp_ports:
            component_results.append(
                ACComponentReduction(
                    labels=comp_labels,
                    port_labels=(),
                    interior_labels=(),
                    local_admittance=[],
                    record=None,
                    floating=True,
                )
            )
            continue

        comp_interior = [x for x in comp_labels if x not in port_set]
        p_idx = [network.index(p) for p in comp_ports]
        i_idx = [network.index(x) for x in comp_interior]

        y_pp = CMat.submatrix(y_full, p_idx, p_idx)
        y_pi = CMat.submatrix(y_full, p_idx, i_idx)
        y_ip = CMat.submatrix(y_full, i_idx, p_idx)
        y_ii = CMat.submatrix(y_full, i_idx, i_idx)

        if comp_interior:
            try:
                yii_inv = CMat.inverse(y_ii)
            except SingularSystemError:
                # 精确零主元：纯 L/C 组件谐振，端口导纳为极点。
                resonant.append(tuple(comp_labels))
                continue
            n_p, n_i = len(p_idx), len(i_idx)
            local_y = [
                [
                    y_pp[r][c]
                    - sum(
                        (
                            y_pi[r][t] * yii_inv[t][s] * y_ip[s][c]
                            for t in range(n_i)
                            for s in range(n_i)
                        ),
                        CFrac(),
                    )
                    for c in range(n_p)
                ]
                for r in range(n_p)
            ]
            backmap = [
                [
                    -sum(
                        (yii_inv[t][s] * y_ip[s][c] for s in range(n_i)),
                        CFrac(),
                    )
                    for c in range(n_p)
                ]
                for t in range(n_i)
            ]
            record = ACSchurRecord(
                port_labels=tuple(comp_ports),
                interior_labels=tuple(comp_interior),
                y_pp=y_pp,
                y_pi=y_pi,
                y_ii=y_ii,
                yii_inverse=yii_inv,
                backmap=backmap,
            )
        else:
            local_y = y_pp
            record = ACSchurRecord(
                port_labels=tuple(comp_ports),
                interior_labels=(),
                y_pp=y_pp,
                y_pi=[],
                y_ii=[],
                yii_inverse=[],
                backmap=[],
            )

        for a, pa in enumerate(comp_ports):
            ga = port_list.index(pa)
            for b, pb in enumerate(comp_ports):
                gb = port_list.index(pb)
                admittance[ga][gb] = local_y[a][b]

        component_results.append(
            ACComponentReduction(
                labels=comp_labels,
                port_labels=tuple(comp_ports),
                interior_labels=tuple(comp_interior),
                local_admittance=local_y,
                record=record,
                floating=False,
            )
        )

    if resonant:
        raise _resonant_error(resonant, w)

    return ACReduction(
        network=network,
        ports=tuple(port_list),
        omega=w,
        admittance=admittance,
        components=component_results,
    )
