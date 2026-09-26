"""电导加权随机游走的期望首次到达时间（Poisson 方程）。

普通无向边电导 ``g_ij`` 给出离开节点的转移权重；可选停留权重 ``s_i``
给出自环权重。对非目标节点 ``i``，令 ``w_i = s_i + sum_j g_ij``，则

    P(i, j) = g_ij / w_i,  P(i, i) = s_i / w_i,
    h_i = 1 + sum_j P(i,j) h_j.

目标集合吸收，故 ``h_t = 0``。乘以 ``w_i`` 并把自环项移到左侧后：

    sum_j g_ij (h_i - h_j) = w_i.

自环只增加右侧的一步时间，不改变普通边拉普拉斯算子本身。

这是随机游走的 Poisson/离散 Dynkin 方程；有效电阻只用于附加的通勤时间
恒等式检查，不被直接当成到达时间。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Hashable, Mapping, Sequence

from .errors import NodeError, PortError, StayWeightError
from .linalg import Mat
from .network import Network, exact_scalar


@dataclass(frozen=True)
class HittingTimeResult:
    """期望首次到达时间、转移概率、残差与连通性诊断。"""

    network: Network
    targets: tuple[Hashable, ...]
    expectations: dict[Hashable, Fraction | float]
    transition_probabilities: dict[Hashable, dict[Hashable, Fraction]]
    residuals: dict[Hashable, Fraction | None]
    max_abs_residual: Fraction
    target_components: list[tuple[Hashable, ...]]
    reachable_nodes: tuple[Hashable, ...]
    unreachable_nodes: tuple[Hashable, ...]
    unreachable_components: list[tuple[Hashable, ...]]
    zero_total_weight_nodes: tuple[Hashable, ...]


def _validate_targets(
    network: Network,
    targets: Sequence[Hashable],
) -> list[Hashable]:
    if isinstance(targets, (str, bytes)):
        raise PortError("targets 必须是标签序列（单目标也请写成 [label]）")
    try:
        target_list = list(targets)
    except TypeError as exc:
        raise PortError("targets 必须是节点标签的可迭代对象") from exc
    if not target_list:
        raise PortError("至少需要指定一个吸收目标")
    seen: set[Hashable] = set()
    for target in target_list:
        if isinstance(target, bool):
            raise PortError("目标标签不得为 bool", node=target)
        if target in seen:
            raise PortError(f"目标重复：{target!r}", node=target)
        seen.add(target)
        network.index(target)
    return target_list


def _coerce_stay_weights(
    network: Network,
    stay_weights: Mapping[Hashable, object] | None,
) -> list[Fraction]:
    values = list(network.stay_weights)
    if stay_weights is None:
        return values
    if not isinstance(stay_weights, Mapping):
        raise TypeError("stay_weights 必须是 {节点: 非负权重} 映射或 None")
    unknown = [node for node in stay_weights if node not in network.nodes]
    if unknown:
        raise NodeError(f"未知停留权重节点：{unknown!r}", node=unknown[0])
    for node, raw in stay_weights.items():
        idx = network.index(node)
        values[idx] = exact_scalar(
            raw,
            f"节点 {node!r} 的停留权重",
            StayWeightError,
        )
        if values[idx] < 0:
            raise StayWeightError(
                f"节点 {node!r} 的停留权重必须非负，收到 {values[idx]}",
                node=node,
            )
    return values


def expected_hitting_times(
    network: Network,
    targets: Sequence[Hashable],
    *,
    stay_weights: Mapping[Hashable, int | Fraction] | None = None,
) -> HittingTimeResult:
    """求每个起点首次到达吸收目标集合的期望步数。

    与至少一个目标连通的非目标节点返回精确 ``Fraction``；目标自身返回 0。
    与目标集合不连通的组件不会送入奇异线性求解，其起点期望为
    ``math.inf``，并在 ``unreachable_components`` 中显式列出。
    """
    if not isinstance(network, Network):
        raise TypeError("network 必须是 resistor_elim.Network 实例")

    target_list = _validate_targets(network, targets)
    stays = _coerce_stay_weights(network, stay_weights)
    target_set = set(target_list)
    labels = list(network.nodes)
    index = {label: i for i, label in enumerate(labels)}
    components = network.components()

    target_components: list[tuple[Hashable, ...]] = []
    unreachable_components: list[tuple[Hashable, ...]] = []
    reachable: set[Hashable] = set()
    for component in components:
        if any(label in target_set for label in component):
            target_components.append(component)
            reachable.update(component)
        else:
            unreachable_components.append(component)

    expectations: dict[Hashable, Fraction | float] = {
        label: (Fraction(0) if label in target_set else float("inf"))
        for label in labels
    }
    residuals: dict[Hashable, Fraction | None] = {
        label: (Fraction(0) if label in target_set else None)
        for label in labels
    }

    transitions: dict[Hashable, dict[Hashable, Fraction]] = {}
    zero_total_nodes: list[Hashable] = []
    for i, label in enumerate(labels):
        if label in target_set:
            transitions[label] = {label: Fraction(1)}
            continue
        neighbors = network.neighbors(label)
        edge_total = sum(neighbors.values(), Fraction(0))
        total = edge_total + stays[i]
        if total == 0:
            zero_total_nodes.append(label)
            transitions[label] = {}
            continue
        row = {label: stays[i] / total} if stays[i] != 0 else {}
        for neighbor, conductance in neighbors.items():
            row[neighbor] = conductance / total
        transitions[label] = row

    for component in target_components:
        unknown = [label for label in component if label not in target_set]
        if not unknown:
            continue
        unknown_idx = [index[label] for label in unknown]
        unknown_column = {global_index: col for col, global_index in enumerate(unknown_idx)}
        size = len(unknown)
        matrix = Mat.zeros(size, size)
        rhs = [Fraction(0) for _ in unknown]

        for row, label in enumerate(unknown):
            i = index[label]
            neighbors = network.neighbors(label)
            total = sum(neighbors.values(), Fraction(0)) + stays[i]
            matrix[row][row] = total - stays[i]
            rhs[row] = total
            for neighbor, conductance in neighbors.items():
                if neighbor in target_set:
                    continue
                column = unknown_column[index[neighbor]]
                if index[neighbor] != i:
                    matrix[row][column] -= conductance

        values = Mat.solve(matrix, rhs)
        for label, value in zip(unknown, values):
            expectations[label] = value

        for row, label in enumerate(unknown):
            residuals[label] = (
                sum(
                    (
                        matrix[row][column] * values[column]
                        for column in range(size)
                    ),
                    Fraction(0),
                )
                - rhs[row]
            )

    finite_residuals = [abs(value) for value in residuals.values() if value is not None]
    max_residual = max(finite_residuals, default=Fraction(0))

    return HittingTimeResult(
        network=network,
        targets=tuple(target_list),
        expectations=expectations,
        transition_probabilities=transitions,
        residuals=residuals,
        max_abs_residual=max_residual,
        target_components=target_components,
        reachable_nodes=tuple(label for label in labels if label in reachable),
        unreachable_nodes=tuple(
            label for label in labels if label not in reachable
        ),
        unreachable_components=unreachable_components,
        zero_total_weight_nodes=tuple(zero_total_nodes),
    )
