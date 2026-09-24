# resistor_elim —— 定向结点网络的电阻消元与 R/L/C 相量复导纳

对无源网络做**定向结点（Schur / 星-网）消元**：

- **直流/电阻**：给定正电导电阻图与端口结点，逐连通组件消除内部节点，
  输出端口 Dirichlet-to-Neumann（DtN，电流/电势）矩阵；全程精确
  `Fraction`。无端口组件显式标记为 `floating`，绝不丢弃或掩盖。
- **交流相量（本次新增）**：在电阻之外加入**正电容、正电感**，用
  **修正结点方程（MNA）**在给定**非零有限角频率 ω** 下求端口**复导纳**
  `Y(jω)`；全程精确复有理（实部/虚部分别为 `Fraction`），谐振奇异性由
  精确零主元判定。给定端口相量电势可恢复内部相量电势、支路电流与复功率。

消元记录完整保留：直流下给定端口电势可精确恢复内部电势、支路电流与
功率；交流下同理恢复相量量与复功率。

- **Python 版本**：Python 3.14.7（3.10+ 亦可运行；开发与验收使用 3.14.7）
- **依赖**：仅标准库（`fractions`、`dataclasses` 等）。安装 Python 后
  无需安装任何第三方包，无网络/账号/数据库/云 API 需求。
- **数值内核**：直流全程 `fractions.Fraction` 精确有理运算；交流全程使用
  库内自实现的精确复分数 `CFrac`（实部、虚部各为一个 `Fraction`）。
  拉普拉斯/结点导纳拼装、分块 Schur 补、有理（复）高斯消元、内部量恢复
  全部在本仓库从零实现，不使用第三方同类实现。

## 快速开始

```python
from fractions import Fraction
from resistor_elim import Network, schur_eliminate

net = Network(["p", "q", "m"])          # 最多 30 个节点
net.add_resistor("p", "m", 2)           # 电阻 R=2（电导 1/2）
net.add_resistor("m", "q", 6)           # 电阻 R=6
# 也可用 net.add_conductor("m", "q", Fraction(1, 6))

red = schur_eliminate(net, ports=["p", "q"])
print(red.dtn)        # [[1/8, -1/8], [-1/8, 1/8]]（串联 ⇒ g=1/(2+6)）
print(red.floating_components)          # 无端口组件元组列表

sol = red.solve({"p": 8, "q": 0})       # 给定端口电势
sol.potentials["m"]      # 6 —— 精确恢复内部电势
sol.port_currents        # {"p": 1, "q": -1} —— 注入电流
sol.port_power           # 8 —— Σ V_p I_p
sol.dissipated_power     # 8 —— Σ g(ΔV)²，精确相等
```

### 交流相量（R/L/C，新能力）

```python
from fractions import Fraction
from resistor_elim import Network, ac_schur_eliminate, CFrac

net = Network(["p", "m", "q"])
net.add_resistor("p", "m", 2)             # R = 2 Ω
net.add_capacitor("m", "q", Fraction(1, 4))  # C = 1/4 F

red = ac_schur_eliminate(net, ports=["p", "q"], omega=2)   # ω 非零有限
y = red.admittance[0][0]                  # CFrac：1/4 + 1/4 j
#   Z = R + 1/(jωC) = 2 - 2j ⇒ Y = 1/(2-2j) = 1/4 + j/4

sol = red.solve({"p": 1, "q": 0})         # 相量电势，可传 int/Fraction/complex
sol.potentials["m"]                        # CFrac(1/2, -1/2)
sol.port_currents["p"]                     # CFrac(1/4, 1/4)
sol.port_power          # 端口复功率 1/2·V·conj(I)（振幅相量约定，CFrac）
sol.dissipated_power    # 电阻平均耗散 = port_power.re（精确 Fraction）
```

元件相量导纳（振幅相量、`e^{jωt}` 约定）：

| 元件 | 添加方法 | 值 | 支路导纳 Y |
| --- | --- | --- | --- |
| 电阻 | `add_resistor(u,v,R)` / `add_conductor(u,v,g)` | R>0 / g>0 | `g = 1/R`（实） |
| 电容 | `add_capacitor(u,v,C)` | C>0 | `jωC`（+虚） |
| 电感 | `add_inductor(u,v,L)` | L>0 | `1/(jωL) = -j/(ωL)`（-虚） |

平行同类元件按基尔霍夫定律合并：电导/电容直接相加，电感按 `1/L` 相加。
`omega` 接受 `int`/`Fraction` 或有限非零 `float`（按 `Fraction(ω)` 精确
转换）；负频率在数学上允许，且 `Y(-ω) = conj(Y(ω))`。

## 准确命令

```bash
python -m unittest discover -s tests -v   # 必要测试（unittest 自动统计用例数）
python demo.py                             # 固定输入演示（约 0.1 秒）
```

`demo.py` 真实计算并打印两部分：

- **A. 直流**：含内部节点、浮接组件与孤立节点的网络的消元结果（DtN、
  对称性/行和检查、内部电势/电流/功率恢复），与 demo 内**独立编写**的
  全图线性系统求解逐项比较；随后实际触发零电阻 `ZeroResistanceError` 与
  `NaN` 电导拒绝。
- **B. 交流相量**：含 R/L/C、内部节点、桥臂与浮置谐振 LC 组件的网络在
  `ω=3/2` 下的端口复导纳，与独立复 KCL 全图求解对照内部相量电势/电流/
  复功率，并用**手算解析式**核对单 RC、串 RLC、纯电阻与负频率共轭；
  随后实际触发零频/NaN 频率 `FrequencyError` 与纯 LC 谐振
  `SingularSystemError`（带谐振组件定位），并展示加阻尼后极点变为有限值。

无 sleep、无预录结果、无压力测试。

## 公开数据结构与接口

| 名称 | 说明 |
| --- | --- |
| `Network(nodes)` | 网络。节点标签任意可哈希、非 `bool`、互不相同，数量 1–30。 |
| `Network.add_conductor(u, v, g)` | 加电导 `g`（`int`/`Fraction`，**严格正**）。平行边电导相加。 |
| `Network.add_resistor(u, v, r)` | 加电阻 `r`（严格正）；`r=0` 抛 `ZeroResistanceError`。 |
| `Network.laplacian()` | 结点拉普拉斯 `K`（`Fraction` 矩阵，按节点顺序）。 |
| `Network.components()` | 无向连通组件；**孤立节点自成组件显式出现**。 |
| `schur_eliminate(network, ports)` | 消元，返回 `Reduction`。不修改原网络。 |
| `Reduction.dtn` | 全局端口 DtN 矩阵，行列顺序同 `ports`。 |
| `Reduction.components` | 逐组件 `ComponentReduction`（含局部 DtN 与 `SchurRecord`）。 |
| `Reduction.floating_components` | 无端口组件的标签元组列表（含孤立节点）。 |
| `Reduction.solve(potentials)` | 序列（按端口序）或 `{端口: 电势}` 映射，返回 `Solution`。 |
| `Solution.potentials` | 可确定节点（端口组件中的端口+内部节点）电势。 |
| `Solution.port_currents` | 各端口注入电流（流入为正）。 |
| `Solution.branch_currents` / `branch_power` | 与 `network.branches` 对齐；浮接组件支路为 `None`。 |
| `Solution.port_power` / `dissipated_power` | 端口注入功率 / 支路耗散功率（精确相等）。 |
| `Solution.undetermined_nodes` | 浮接组件节点（电势只能确定到组件常数）。 |
| `SchurRecord` | 保留 `K_PP, K_PI, K_II, K_II^{-1}, backmap=-K_II^{-1}K_IP`。 |
| `Network.add_capacitor(u, v, C)` | 加电容（**严格正** int/Fraction；平行电容相加）。 |
| `Network.add_inductor(u, v, L)` | 加电感（**严格正**；平行电感按 `1/L` 相加）。 |
| `Network.branches_ac` | 全部支路记录 `(u, v, kind, value)`；kind 为 `R/C/L`。 |
| `Network.capacitance_between` / `inductance_between` | 并联合并后的 C 与等效 L（无则为 0）。 |
| `Network.admittance_matrix(ω)` | 该频率下的相量结点导纳 `Y(jω)`（`CFrac` 矩阵）。 |
| `ac_schur_eliminate(network, ports, omega)` | 相量消元，返回 `ACReduction`。不修改原网络。 |
| `ACReduction.omega` / `.admittance` | 分析频率 `Fraction` 与全局端口复导纳（行列同 `ports`）。 |
| `ACReduction.components` / `.floating_components` | 逐组件 `ACComponentReduction` / 无端口组件清单。 |
| `ACReduction.solve(phasors)` | 给定端口相量电势（实数或有限 `complex`/`CFrac`），返回 `ACSolution`。 |
| `ACSolution.potentials` / `.port_currents` | 可确定节点相量电势 / 各端口相量注入电流（`CFrac`）。 |
| `ACSolution.branch_currents` / `.branch_power` | 与 `branches_ac` 对齐；浮置组件为 `None`；功率为 1/2·ΔV·conj(I)。 |
| `ACSolution.port_power` / `.dissipated_power` | 端口复功率（`CFrac`）/ 电阻平均耗散（= 复功率实部，`Fraction`）。 |
| `ACSolution.undetermined_nodes` | 浮置组件节点（相量电势只能确定到组件常数）。 |
| `ACSchurRecord` | 保留 `Y_PP, Y_PI, Y_II, Y_II^{-1}, backmap=-Y_II^{-1}Y_IP`。 |

异常层级：`NetworkError`（基类，携带可定位的 `node`/`edge`）
⊢ `NodeError`、`ConductanceError`、`ResistanceError`、
`CapacitanceError`、`InductanceError`、`FrequencyError`、`PortError`、
`PotentialError`、`LimitError`
⊢ `ZeroResistanceError`（是 `ResistanceError` 子类）、`SingularSystemError`。

## 数学背景

### 直流/电阻

每个含端口的连通组件按端口 `P` / 内部 `I` 分块：

```
K = [ K_PP K_PI ]      内部 KCL: K_IP V_P + K_II V_I = 0
    [ K_IP K_II ]
V_I = -K_II^{-1} K_IP V_P
I_P = (K_PP - K_PI K_II^{-1} K_IP) V_P = S V_P
```

`S` 即端口 DtN。`K_II` 在“组件连通且至少有一个端口节点、所有电导严格
为正”的前提下正定，故精确可逆；实现不使用数值容差。

### 交流相量（修正结点方程）

采用 `e^{jωt}` 振幅相量约定。在本子集（无独立源/受控源，ω≠0，电感
导纳有限）中，MNA 不需要附加电流变量，直接是结点导纳系统
`Y(jω) V = I`。逐组件同样做精确复 Schur 补：

```
Y = [ Y_PP Y_PI ]      I_P = (Y_PP - Y_PI Y_II^{-1} Y_IP) V_P = Y_port V
    [ Y_IP Y_II ]
```

**奇异（谐振）语义**：含端口的纯 L/C 无耗散组件在某频率满足
`det Y_II = 0`（精确零主元）时，端口导纳为极点，抛
`SingularSystemError`，异常的 `resonant_components` 属性列出全部谐振
组件标签；**无端口组件即使谐振也只标记为 floating**（无法被激励）。
加入任意正电阻（阻尼）使 `Y_II` 的实部严格正定后极点消失，系统在所有
频率可解。

## 输入上限与错误语义

- 节点数硬上限 **30**（`resistor_elim.network.MAX_NODES`），超限抛
  `LimitError`。边数受稠密矩阵规模约束（30 节点完全图 435 条）。
- 电导/电阻/电容/电感只接受 `int` 与 `Fraction`：`float` 一律拒绝
  （其中 `NaN`/`±Infinity` 给出专门报错，分别为 `ConductanceError`、
  `ResistanceError`、`CapacitanceError`、`InductanceError`），以免
  二进制浮点被误当精确值。需要小数量级请写 `Fraction(17, 10)`。
- **角频率 ω** 接受 `int`/`Fraction` 与有限非零 `float`（按
  `Fraction(ω)` 精确转换）；零频抛 `FrequencyError`（本迭代不做直流
  极限），`NaN`/`±Infinity` 同样抛 `FrequencyError`，`bool` 拒绝。
- 端口电势：直流接受 int/Fraction 与有限 float；交流相量额外接受有限
  `complex`（实虚部按 `Fraction` 精确转换），`NaN`/`±Infinity` 抛
  `PotentialError`。
- 任何位置的 `bool`（`True/False`，虽为 `int` 子类）都被拒绝。
- 边引用未注册节点 → `NodeError`（异常带节点名）；自环、非正元件值、
  重复节点标签、空/重复/未知端口、电势数量或键不符均报错，
  **不静默纠正**。
- 失败不留未说明的部分变更：`add_*` 在任何变更前完成全部校验；
  `schur_eliminate`/`ac_schur_eliminate` 不修改输入网络；谐振报错时
  网络支路状态不变。
- 本模块没有“搜索”类算法（是直接的精确消元），因此不存在搜索预算
  概念；矩阵奇异性由精确的零主元判定并抛 `SingularSystemError`（直流
  为孤立内部节点，交流为纯 L/C 谐振极点），与“预算耗尽/未找到”不混淆。

## 误差口径（重要）

- **核心结果零浮点误差**：直流拉普拉斯、Schur 补、逆、内部电势、电流、
  功率全部是精确 `Fraction`；交流导纳矩阵、Schur 补、逆、相量电势/电流、
  复功率全部是精确 `CFrac`（实部虚部各为 `Fraction`）。测试用
  `assertEqual` 做精确等式断言（复对称、行和为 0、复功率实部 == 电阻
  耗散、与独立参考逐值相等），**不使用任何容差**，因此不存在“用宽容差
  掩盖边界错误”。
- 仅在这些地方出现浮点：(1) demo 打印时 `float(...)`/`complex(...)`
  仅为人类可读的近似显示；(2) 用户主动传入 `float` 频率/电势或
  `complex` 相量时，按其二进制值精确转成有理分数（如 `0.1` →
  `3602879701896397/36028797018963968`），若需要十进制 1/10 请显式传
  `Fraction(1, 10)`。
- 大规模网络下有理整数位宽会增长（精确性的代价）；30 节点上限内
  运算仍然很快（测试套件与 demo 均在 1 秒内）。

## 测试与独立性说明

- `tests/reference.py` 是**独立参考**：自行拼装支路表拉普拉斯/相量导纳、
  自行编写（复）高斯消元，并给出串/并联电阻与单 RC、串 RLC 解析式；
  它不导入被测核心（复数运算也用独立的 `(re, im)` 二元组函数实现）。
- 期望值来自：串/并联解析式、单 RC/串 RLC/纯电阻解析公式、手算分数值、
  独立全图（复）线性系统参考。“简化后再求解与简化前一致”的往返只作为
  补充性质，不作为期望值来源。
- 交流结果用解析式（单 RC、串 RLC、纯电阻）、手算相量值与确定性枚举的
  独立参考对照；谐振/浮置/零频等边界单独断言。

## 设计取舍

- **精确有理优先于浮点速度**：规模 ≤30 时稠密（复）分数消元成本可接受，
  换来可验证的零误差与严格的奇异判定（谐振是精确零主元，不是“很小”）。
- **显式区分浮置组件**：无端口组件无法被激励（电势只确定到组件常数），
  直流与交流都单独标记、支路量返回 `None`，而不是任取一个节点接地或
  静默丢弃；即使浮置组件在该频率谐振也不报错——孤立节点同样显式保留。
- **谐振是硬错误而非 ±∞**：含端口的无耗散 L/C 组件在谐振频率端口导纳
  为极点，明确抛 `SingularSystemError` 并定位组件，而不是输出无穷大；
  用户可改频率或加阻尼电阻。
- **未被任何元件连接的端口**：导纳矩阵对应为零行零列（该端口无电流），
  不伪造连接关系。
- 平行同类元件直接按基尔霍夫定律合并；无向处理，支路方向仅决定
  `branch_currents` 的符号约定（自 `u` 流向 `v` 为正）。

## 已知限制 / 非目标

- 元件子集为**无源、线性、正值的 R/L/C**：无零/负电阻电容电感、无独立
  源/受控源、无互感/变压器、无二极管、无温度系数；不宣称 SPICE 等行业
  标准兼容。修正结点方程在本子集下无需附加 MNA 电流变量。
- **本迭代只给单个非零有限频率点**的相量复导纳：不做零频（直流）极限
  （抛 `FrequencyError`）、不做频率扫描/频响曲线、不做被动降阶（模型
  降阶/Pade）、不做全频带正实性认证。
- 谐振只做**精确**判定：有理数频率下精确命中的无耗散极点才报奇异；
  不提供数值容差意义上的“近谐振”定位，也不估计品质因数。
- 零电阻/零电感（理想短路线）被明确拒绝：需要时请先在网络定义中合并
  等势节点。
- 稠密矩阵 + 精确（复）分数，不针对超大图/稀疏高性能场景。
- 不提供网站、CLI 产品、报表或文件管理功能；仓库终端命令仅用于
  测试与演示。
