# resistor_elim —— 定向结点网络：电阻等效消元与 RLC 相量分析

两条能力线，共用同一套节点/支路数据模型与错误语义：

1. **直流（精确有理）**：对无源电阻网络做**定向结点（Schur / 星-网）
   消元**——给定正电导电阻图与端口结点，逐连通组件消除内部节点，输出
   端口的 Dirichlet-to-Neumann（DtN，电流/电势）矩阵；无端口组件显式
   标记为 `floating`，绝不丢弃或掩盖。消元记录完整保留，给定端口电势
   可精确恢复内部电势、支路电流与功率。
2. **交流（精确复数）**：在电阻网络上加入电容 `C`、电感 `L`，在给定的
   **非零有限角频率 ω** 下用**修正结点方程（MNA）**做正弦稳态相量分析，
   Schur 消去内部变量后输出**端口复导纳矩阵 Y**；给定端口相量电压可
   恢复内部相量电压、R/C/L 各支路相量电流与平均功率。浮置/奇异系统
   （如理想 LC 串联谐振）被明确报告。

- **Python 版本**：Python 3.14.7（3.10+ 亦可运行；开发与验收使用 3.14.7）
- **依赖**：仅标准库（`fractions`、`dataclasses`、`cmath` 等）。安装
  Python 后无需安装任何第三方包，无网络/账号/数据库/云 API 需求。
- **数值内核**：直流全程 `fractions.Fraction` 精确有理；交流全程使用
  实部/虚部均为 `Fraction` 的精确复数 `QComplex`（本仓库 `cx.py`），
  不使用浮点容差，核心算法（拉普拉斯/导纳拼装、MNA 戳元、分块 Schur
  补、有理/复高斯消元、内部量恢复）全部在本仓库从零实现。

## 快速开始

```python
from fractions import Fraction
from resistor_elim import Network, schur_eliminate, ac_admittance

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

# ---- 交流：加入电容/电感，在非零频率点求端口复导纳 ----
ac = Network(["p", "a", "q"])
ac.add_resistor("p", "a", 2)
ac.add_capacitor("a", "q", 3)           # 电容 C=3（严格正）
res = ac_admittance(ac, ["p", "q"], omega=5)
res.y[0][0]              # 精确复数：R、C 串联 ⇒ 450/901 + j·15/901 S
phasor = res.solve({"p": 1, "q": 0})    # 相量电压（支持 complex/QComplex）
phasor.potentials["a"]                  # 内部相量电压（精确复数）
phasor.capacitor_currents               # 各电容支路相量电流
phasor.average_dissipated_power         # 电阻平均耗散功率（== 端口有功）
```

## 准确命令

```bash
python -m unittest discover -s tests -v   # 必要测试（unittest 自动统计用例数）
python demo.py                             # 固定输入演示（约 0.1 秒）
```

`demo.py` 真实计算并打印两段内容：

- **直流段**：一个含内部节点、浮接组件与孤立节点的网络的消元结果
  （DtN、对称性/行和检查、内部电势/电流/功率恢复），与 demo 内
  **独立编写**的全图线性系统求解逐项比较；随后实际触发零电阻
  `ZeroResistanceError`，并演示 `NaN` 电导被拒绝。
- **交流段**：单 RC、串联 RLC 的复导纳与**独立解析式**逐项对照；
  纯电阻交流结果与直流 DtN 精确一致、负频率共轭、理想 LC 并联谐振
  开路、浮置组件标记等边界；随后**实际触发**零频 `FrequencyError`、
  `NaN` 频率拒绝、理想 LC 串联谐振 `SingularSystemError`，以及对含
  C/L 的网络调用直流接口的拒绝。

无 sleep、无预录结果、无压力测试。

## 公开数据结构与接口

| 名称 | 说明 |
| --- | --- |
| `Network(nodes)` | 网络。节点标签任意可哈希、非 `bool`、互不相同，数量 1–30。 |
| `Network.add_conductor(u, v, g)` | 加电导 `g`（`int`/`Fraction`，**严格正**）。平行边电导相加。 |
| `Network.add_resistor(u, v, r)` | 加电阻 `r`（严格正）；`r=0` 抛 `ZeroResistanceError`。 |
| `Network.add_capacitor(u, v, c)` | 加电容 `c`（严格正，`int`/`Fraction`）；平行电容容量相加。 |
| `Network.add_inductor(u, v, l)` | 加电感 `l`（严格正）；每条电感在 MNA 中占独立电流变量，逐条保留。 |
| `Network.laplacian()` | **直流**电阻拉普拉斯 `K`（`Fraction`）。不含 C/L。 |
| `Network.branches` / `capacitor_branches` / `inductor_branches` | 三类支路的添加记录（`(u,v,值)`）。 |
| `Network.components()` | 无向连通组件（R/C/L 都算边）；**孤立节点自成组件显式出现**。 |
| `schur_eliminate(network, ports)` | **直流**消元，返回 `Reduction`。网络含 C/L 时明确拒绝（见下）。 |
| `ac_admittance(network, ports, omega)` | **交流**：非零有限角频率下求端口复导纳，返回 `AdmittanceResult`。 |
| `Reduction.dtn` | 直流全局端口 DtN 矩阵，行列顺序同 `ports`。 |
| `Reduction.components` / `floating_components` | 逐组件 `ComponentReduction` / 无端口组件标签元组列表。 |
| `Reduction.solve(potentials)` | 直流：序列或 `{端口: 电势}`，返回 `Solution`。 |
| `AdmittanceResult.omega` / `y` | 角频率（`Fraction`）/ 端口复导纳矩阵（`QComplex`）。 |
| `AdmittanceResult.components` / `floating_components` | 逐组件 `ACComponentReduction`（含 `ACSchurRecord`）/ 浮置组件。 |
| `AdmittanceResult.solve(phasors)` | 给定端口相量电压（实数或 `complex`/`QComplex`），返回 `ACSolution`。 |
| `ACSolution.potentials` | 端口组件中端口+内部节点的相量电压（`QComplex`）。 |
| `ACSolution.port_currents` | 各端口注入相量电流（流入为正）。 |
| `ACSolution.branch_currents` / `capacitor_currents` / `inductor_currents` | 与三类支路记录对齐；浮置组件支路为 `None`。 |
| `ACSolution.port_apparent_power` | 端口视在功率 `S = ½·Σ V·conj(I)`（峰值相量约定）。 |
| `ACSolution.average_port_power` / `average_dissipated_power` | 端口平均有功 / 电阻平均耗散（精确相等；L、C 不耗散）。 |
| `ACSolution.undetermined_nodes` | 浮置组件节点。 |
| `QComplex` | 精确复数（实部/虚部均为 `Fraction`），支持四则运算、`conj()`、`abs_sq()`。 |
| `SchurRecord` / `ACSchurRecord` | 保留分块矩阵、内部块逆与 `backmap`（内部量 = backmap @ 端口量）。 |

异常层级：`NetworkError`（基类，携带可定位的 `node`/`edge`）
⊢ `NodeError`、`ConductanceError`、`ResistanceError`、
`CapacitanceError`、`InductanceError`
⊢ `ZeroResistanceError`、`PortError`、`PotentialError`、
`FrequencyError`、`LimitError`、`SingularSystemError`。

## 交流数学背景（修正结点方程）

在角频率 ω 下采用相量约定 `x(t) = Re{X·e^{jωt}}`（相量按**峰值振幅**）：

- 电阻导纳 `g = 1/R`；电容导纳 `jωC`——直接戳入结点导纳块
  （`K[i,i] += y`、`K[i,j] -= y`）；
- 电感阻抗 `jωL`，其导纳在 ω=0 发散，故**不**并入结点块，而是为每条
  电感引入支路电流变量 `i_L`（正方向 u→v），戳入：

```
KCL @ u,v : 结点方程中出现 +i_L / −i_L
KVL 支路行 : V_u − V_v − jωL·i_L = 0
```

每个含端口组件的变量按端口结点电压 `P` 与内部变量 `I`（内部结点电压
＋该组件全部电感电流）分块，复 Schur 补给出端口复导纳：

```
Y = [ A  B ]     I_P = (A − B D^{−1} C) V_P = Y_port V_P
    [ C  D ]
内部变量 = −D^{−1} C V_P
```

复导纳矩阵满足**复对称** `Y = Yᵀ`（注意不是 Hermitian 共轭对称）与
行和为 0；无损耗网络实部为 0；`Y(−ω) = conj(Y(ω))`。这些关系在测试
中做**精确**断言。

**奇异/浮置语义**：`ω ≠ 0` 且所有元件严格为正时内部块 `D` 通常可逆。
两类例外被明确报告而非掩盖：(1) 不含端口的组件标记为 `floating`
（含仅由 C/L 挂到悬空内部节点的情形，其输入导纳为 0、内部节点等势
随动）；(2) 真正奇异的 `D`——例如两端口间仅一条理想 LC 串联且恰在
`ω²LC = 1` 的零阻抗谐振点——抛 `SingularSystemError`（精确零主元判定，
无容差）。并联谐振（L//C 在 `ω²LC=1`）不奇异：导纳恰为 0，正常返回。

## 输入上限与错误语义

- 节点数硬上限 **30**（`resistor_elim.network.MAX_NODES`），超限抛
  `LimitError`。边数受稠密矩阵规模约束（另含每条电感一个 MNA 电流变量）。
- **元件值**（电导/电阻/电容/电感）只接受 `int` 与 `Fraction`：`float`
  一律拒绝（其中 `NaN`/`±Infinity` 给出专门报错），以免二进制浮点被误当
  精确值。需要小数量级请写 `Fraction(17, 10)`。所有元件值必须**严格正**。
- **角频率 ω** 接受 `int`/`Fraction` 与**有限** `float`（按 `Fraction(x)`
  精确转换）；拒绝零（`FrequencyError`，零频极限不在契约内）、
  `NaN`/`±Infinity`、`bool`、复数与字符串。负频率合法，用于检验
  `Y(−ω)=conj(Y(ω))`。
- **端口相量电压**接受 `int`/`Fraction`/有限 `float`（实数相量）、内置
  `complex`（实虚部均须有限）或 `QComplex`；`NaN`/`Infinity`/`bool`/
  字符串抛 `PotentialError`。直流端口电势规则同旧版。
- 任何位置的 `bool`（`True/False`，虽为 `int` 子类）都被拒绝。
- 边引用未注册节点 → `NodeError`（异常带节点名）；自环、非正元件值、
  重复节点标签、空/重复/未知端口、相量数量或键不符均报错，
  **不静默纠正**。
- 失败不留未说明的部分变更：`add_*` 在任何变更前完成全部校验；
  `schur_eliminate` / `ac_admittance` 不修改输入网络（即便计算中途遇到
  奇异性，网络仍可在其他频率继续使用）。
- **直流/交流入口分工**：`schur_eliminate` 只处理纯电阻网络；网络含
  电容或电感时它明确抛 `NetworkError` 并指向 `ac_admittance`（因为 ω=0
  时 C 开路、L 短路，零频极限不在本次契约内）。纯电阻网络的交流结果
  与直流 DtN 逐值相等。
- 本模块没有“搜索”类算法（是直接的精确消元），因此不存在搜索预算
  概念；奇异性由精确的零主元判定并抛 `SingularSystemError`，
  与“预算耗尽/未找到”不存在混淆。

## 误差口径（重要）

- **核心结果零浮点误差**：
  - 直流：拉普拉斯、Schur 补、逆、电势、电流、功率全部是精确 `Fraction`；
  - 交流：导纳、MNA、Schur 补、相量电压/电流的实部与虚部全部是精确
    `Fraction`（`QComplex`）。
- 测试对解析结果（单 RC、串/并 RLC、纯电阻退化、负频率共轭、理想 LC
  谐振、功率守恒等）用 `assertEqual` 做**精确等式**断言，不用容差；
  仅在与独立 `complex` 浮点参考（确定性小图枚举）比较时使用
  `cmath.isclose` 的显式容差（1e-9），该容差只描述浮点参考自身的舍入。
- 仅在这些位置出现浮点：(1) demo/打印时 `float(x)` 仅为人类可读近似；
  (2) 用户主动传入有限 `float`（ω、端口电势）时按其二进制值精确转成
  有理分数（如 `0.1` → `3602879701896397/36028797018963968`），需要
  十进制 1/10 请显式传 `Fraction(1,10)`。
- 大规模网络下有理整数位宽会增长（精确性的代价）；30 节点上限内
  运算仍然很快（测试套件与 demo 均在 1 秒内）。

## 测试与独立性说明

- `tests/reference.py` 是**独立参考**：
  - 直流部分自行拼装支路表拉普拉斯、自行编写高斯消元，并给出串/并联
    解析式；
  - 交流部分（`RefACNetwork`）用 Python 内置 `complex` 浮点**独立拼装
    MNA、独立高斯消元**，用“逐端口施加单位相量电压”的全图激励法求
    导纳（不做内部消元），并独立给出串/并联 RLC 解析式。
  - 它不导入被测核心，被测库与参考之间只比较结论。
- 期望值来自：R/L/C 串/并联解析式、手算分数值（如串联 RLC 的
  `(50−j345)/4861`、功率 `225/901`）、独立全图系统参考。24 个确定性
  LCG 小图（含 100+ 电感/电容 MNA 变量）与独立参考逐值比对导纳、节点
  电压、支路电流；“简化后再求解与简化前一致”的往返只作为补充性质，
  不作为期望值来源。

## 设计取舍

- **精确（有理/精确复数）优先于浮点速度**：规模 ≤30 时稠密消元成本
  可接受，换来可验证的零误差与严格的奇异判定；交流结果能与解析式
  逐值 `assertEqual`。
- **电感用 MNA 电流变量，而非结点导纳 `1/(jωL)`**：后者在 ω=0 必然
  发散；MNA 形式在每个非零频率点定义干净，且与电阻/电容的结点戳元
  自然统一。每条电感单独保留电流变量（平行电感不合并）。
- **直流/交流入口显式分工**：`schur_eliminate` 拒绝含 C/L 的网络，
  不在直流接口里偷偷“C 开路、L 短路”——零频极限是非目标，明确指引
  用户改用 `ac_admittance`。
- **显式区分浮接组件**：无端口组件无法被激励（相量电压只确定到组件
  常数），单独标记、支路量返回 `None`，而不是任取一个节点接地或
  静默丢弃——孤立节点同样显式保留。
- **奇异点显式报错而非返回 ±∞**：理想 LC 串联谐振使内部块精确奇异，
  抛 `SingularSystemError`；理想 LC 并联谐振导纳恰为 0（有限且可逆），
  正常返回 0。
- **未被任何支路连接的端口**：导纳矩阵对应为零行零列（该端口无电流），
  不伪造连接关系。
- 平行电阻电导相加、平行电容容量相加（基尔霍夫并联）；无向处理，
  支路方向仅决定支路电流的符号约定（自 `u` 流向 `v` 为正）。

## 已知限制 / 非目标

- 本次迭代只交付**单个非零有限频率点**的端口复导纳与相量求解。
  **明确不做**：被动降阶/模型化简、全频带正实性（positive-real）认证、
  零频（直流）极限、频率扫描的专门接口。
- 元件子集为无源、线性、R/L/C 且元件值严格正：无零/负元件值、无电源/
  受控源/互感/变压器、无二极管/非线性、无分布参数；不宣称 SPICE 等
  行业标准兼容。
- 零电阻/零电感（理想短路线）被明确拒绝：需要时请先在网络定义中
  合并等势节点。
- 理想（无耗）LC 串联在精确谐振频率点 `ω²LC=1` 无有限导纳解，抛
  `SingularSystemError`；实际电路加任意正电阻即可在该频率得到有限值。
- 稠密矩阵 + 精确分数复数，不针对超大图/稀疏高性能场景；30 节点下
  分数位宽与耗时均可接受（套件 < 1 秒）。
- 相量采用峰值振幅约定（平均功率含 1/2 因子）；不提供有效值（RMS）
  约定的换算开关。
- 不提供网站、CLI 产品、报表或文件管理功能；仓库终端命令仅用于
  测试与演示。
