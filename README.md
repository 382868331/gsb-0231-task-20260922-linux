# schur_elim — 定向结点电阻网络的等效消元模块

对无源（无电源、无受控源）电阻图构建结点电导拉普拉斯矩阵，用 **Schur 补
（逐结点高斯消元）** 消去每个“与端口相连的连通组件”内部的全部结点，得到
端口上的 **Dirichlet-to-Neumann（DtN）矩阵**；不给定任何端口的连通组件
（含孤立结点）标记为 **floating** 单列，绝不丢弃或掩盖。消元过程逐行记录，
给定端口电势后可据此精确恢复内部电势、支路电流与端口注入电流、功率。

- **Python 版本**：Python 3.14.7（使用 3.10+ 的语法特性；仅按 3.14.7 验证）
- **依赖**：仅 Python 标准库（`fractions`、`math`、`dataclasses`、`typing`、
  `unittest`）。安装 Python 后无需安装任何其他包，全程离线，不访问网络、
  账号、硬件或外部服务。

## 运行命令

```bash
# 必要测试（独立参考实现驱动，见 tests/）
python -m unittest discover -s tests -v

# 固定输入演示：真实计算正常结果 + 实际触发零电阻/零电导拒绝
python demo.py
```

演示在普通笔记本上约 0.01 秒完成（要求预算为约 8 秒内），无 sleep、
无预录结果、无压力测试。

## 快速上手

```python
from fractions import Fraction
from schur_elim import ResistorNetwork, reduce_network

net = ResistorNetwork.build(
    nodes=["p", "x", "q"],
    edges=[("p", "x", Fraction(1)), ("x", "q", Fraction(1))],
    ports=["p", "q"],
)
reduced = reduce_network(net)

# 两个 1 S 电导串联 => 端口等效电导 1/2 S，DtN 为 [[1/2,-1/2],[-1/2,1/2]]
print(reduced.dtm)

sol = reduced.solve_boundary({"p": 1, "q": 0})
sol.voltages["x"]        # Fraction(1, 2)  内部电势
sol.port_currents        # {"p": 1/2 A, "q": -1/2 A}
sol.branch_currents      # {("p","x"): 1/2 A, ("x","q"): 1/2 A}
sol.power_injected       # 1/2 W（端口注入）
sol.power_dissipated     # 1/2 W（电阻耗散 Σ g (ΔV)²）
sol.power_balance_residual  # Fraction(0)，精确为零
```

## 公开接口与数据结构

### `ResistorNetwork.build(nodes, edges, ports) -> ResistorNetwork`

- `nodes: Sequence[str]`：结点标签，1–30 个，**唯一、非空字符串**。
- `edges: Sequence[(u, v, g)]`：无向电阻边。`g` 为电导（西门子），接受
  `int | float | fractions.Fraction`，内部一律转成 `Fraction`。
  - 不允许自环；同一对结点只允许一条边（并联请先把电导相加合并，重复边
    直接报 `DuplicateEdgeError`，避免任选一条掩盖建模错误）。
  - 端点必须是已声明结点，否则 `UnknownNodeError`（错误信息含 `edge #k`
    可定位）。
- `ports: Sequence[str]`：端口端子，**至少一个**，必须是已声明结点且不重复。
- 网络对象不可变（frozen dataclass）。
- 方法：`laplacian()` 返回 n×n 精确拉普拉斯的全新副本；
  `connected_components()` 返回连通组件（含孤立结点，按结点序确定）。

### `reduce_network(net) -> ReducedNetwork`

纯函数，不修改输入；要么整体成功返回，要么抛异常，**不存在可观察的部分
结果**。字段：

- `ports: tuple[str, ...]`：全局端口顺序，也是 DtN 的行列顺序。
- `dtm: tuple[tuple[Fraction, ...], ...]`：p×p 端口 DtN 矩阵，满足
  `I_ports = S · V_ports`。多个端口组件按块对角叠加。
- `components: tuple[ComponentInfo, ...]`：每个连通组件一条记录：
  - `kind`：`"ported"` 或 `"floating"`；
  - `nodes / port_nodes / internal_nodes`；
  - `dtm_block`：该组件的端口 DtN 块（floating 为 `None`）；
  - `steps: tuple[EliminationStep, ...]`：消元轨迹，每步是被消去变量
    （主元）在该阶段关于“剩余内部变量 + 端口变量”的一行精确方程。
- 查询：`ported_components()`、`floating_components()`。

### `ReducedNetwork.solve_boundary(voltages) -> BoundarySolution`

- `voltages: Mapping[str, int|float|Fraction]`：必须**恰好覆盖全部端口**；
  多给非端口结点或缺端口均报 `BoundaryConditionError`。
- 反向回放消元轨迹恢复内部电势，再由 DtN 计算端口电流，并计算全部支路
  电流（方向按边声明时的端点顺序）与功率。
- floating 组件的电势只在“每组件一个加性常数”意义下有定义；本实现选定
  规范 **V = 0**（其内支路电流因此为 0）。该取法在返回值与 README 中
  显式说明，不静默掩盖其存在。

### 异常层级

全部继承 `CircuitError(ValueError)`（类型错误用 `TypeError`）：
`DuplicateNodeError`、`DuplicateEdgeError`、`DuplicatePortError`、
`UnknownNodeError`、`SelfLoopError`、`ZeroConductanceError`、
`NegativeConductanceError`、`InfiniteConductanceError`、`NodeLimitError`、
`NoPortsError`、`BoundaryConditionError`、`EliminationSingularError`。

## 输入与错误语义（固定契约）

- **数值**：电导与端口电势拒绝 `NaN`、`+∞`、`-∞`
  （`InfiniteConductanceError`）。电导 `+∞` 即零电阻连接——**零电阻不在
  本数学子集内，明确拒绝**，不会合并成短路线；电导 `0` 是开路，不是边，
  报 `ZeroConductanceError`；负电导（有源网络）报 `NegativeConductanceError`。
- **布尔拒绝**：`bool` 虽是 `int` 子类，但电导、端口电势传 `True/False`
  一律 `TypeError`；结点数等整数形参同样拒绝布尔。
- **非法输入可定位、不静默纠正**：错误信息带边编号（`edge #k`）、结点名
  或缺失端口名；不自动去重、不自动合并并联边、不自动忽略孤立组件。
- **原子性**：`build` / `reduce_network` 是无状态的整体操作，失败时没有
  部分变更；`ResistorNetwork` 不可变。
- **搜索预算 vs 无解**：本模块是确定性的精确线性消元，没有启发式搜索，
  因此不存在“预算耗尽”这一状态；数学上无法消元（内部块奇异）时抛
  `EliminationSingularError`，属于“证明不可解”。在当前输入契约下（组件
  含至少一个端口且边电导严格为正）内部块必正定，该异常不会被正常输入
  触发；保留它是为了不把奇异情形伪装成数值结果。

## 算法

1. 按边累加构建拉普拉斯 `L`：对角为结点入射电导和，非对角为 `-g`。
2. 无向图连通分量分类（BFS）：无端口组件（含孤立结点）直接标 floating。
3. 对每个端口组件，按 `[内部结点 | 端口结点]` 分块
   `A = [[A_II, A_IB],[A_BI, A_BB]]`，对 `A_II` 做逐主元高斯消元并把
   端口列随行变换携带；每步主元行存入 `steps`。
4. 回代求 `X = A_II^{-1} A_IB`，Schur 补
   `S = A_BB − A_BI X` 即该组件的 DtN 块；全局 DtN 按端口索引叠加。
5. 恢复：端口电势固定后按消元逆序回代；电流 `I = S V`；
   功率 `P_in = V_portsᵀ I_ports`，`P_diss = Σ_edges g (ΔV)²`。
   无源网络下二者恒等（测试中以精确分数断言为 0）。

## 误差口径（重要）

- **核心计算零数值误差**：全部使用 `fractions.Fraction` 精确有理运算，
  DtN、恢复电势、电流、功率均为精确分数；对称性、行和为 0、功率守恒
  在测试中按**精确相等**断言，不存在容差。
- **float 输入**：若传入 Python `float`，按该二进制浮点数的**精确有理
  值**转换（例如字面量 `0.1` 实际是
  `3602879701896397/36028797018963968`，不会被四舍五入成十进制 1/10）。
  若需要十进制语义，请直接传入 `Fraction("0.1")` 或整数电导。
- 仅在**展示**时若调用方自行 `float(Fraction)`，才引入一次 IEEE-754
  binary64 舍入，相对误差约 ≤ 2⁻⁵³（≈1.1e-16）；库本身不用宽容差掩盖
  任何边界判定（奇异、零电导等均按精确 0 判定）。

## 输入上限

- 结点数：**最多 30**（第 31 个结点抛 `NodeLimitError`），30 为合法边界，
  测试覆盖恰好 30 结点的链。
- 边数：无向简单图上限 n(n−1)/2 = 435。
- 端口数：1–30 且互不相同。
- 精确分数的分子分母位数无人为截断；上述规模内运算时间远低于秒级。

## 设计取舍

- 选精确有理算术而非浮点：DtN 对称、行和 0、功率守恒可严格断言；代价是
  分数位增长，故规模限定在 30 结点。
- 端口是“端子结点”而非二端对黑箱：DtN 直接定义在被标记结点上，多端口、
  多组件自然支持。
- 并联边要求调用方显式合并：避免在重复边/建模错误上“任选一个”。
- floating 组件保留为显式结果，并在求解时公开声明 gauge 取 0。
- 不做也不宣称完整电路行业标准（SPICE 等）兼容：不支持零电阻、受控源、
  非互易元件、交流频域、非线性元件、地节点约定或瞬态分析；本模块只实现
  本题契约描述的有限无源电阻直流子集。

## 已知限制

- 仅直流、互易、严格正电导的无向电阻网络。
- floating 组件的绝对电势不可观，返回规范零；若需要其物理电势须自行
  引入参考端子。
- 单条边不能表达并联电导（需先相加）；无向边不区分重复方向。
- 精确分数在超大规模或复杂有理输入下会变慢/变大，30 结点契约之外不作
  性能承诺。

## 目录

```
schur_elim/__init__.py        公开 API 导出
schur_elim/elimination.py     核心：校验、拉普拉斯、Schur 消元、恢复
tests/_reference.py           独立参考（自有拉普拉斯+高斯消元，不导入核心）
tests/test_elimination.py     必要测试（解析解 + 独立参考 + 性质断言）
demo.py                       固定输入演示
README.md
```

## 测试独立性说明

测试的期望值来自两类来源，均**不调用被测核心**：

1. 串并联/电桥的闭式解析结果（手工推导的分数值）；
2. `tests/_reference.py`：独立实现的拉普拉斯拼装与 Fraction 高斯消元，
   对每个端口做单位电压求解得到参考 DtN 与全图解。

“消元轨迹回放满足内部结点 KCL”“功率残差为 0”等往返一致性检查仅作为
补充断言。
