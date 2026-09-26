# resistor_elim —— 定向结点网络的电阻等效消元

对无源电阻网络做**定向结点（Schur / 星-网）消元**：给定正电导电阻图与
端口结点，逐连通组件消除内部节点，输出端口的
Dirichlet-to-Neumann（DtN，电流/电势）矩阵；无端口组件显式标记为
`floating`，绝不丢弃或掩盖。消元记录完整保留，给定端口电势可精确恢复
内部电势、支路电流与功率。

- **Python 版本**：Python 3.14.7（3.10+ 亦可运行；开发与验收使用 3.14.7）
- **依赖**：仅标准库（`fractions`、`dataclasses` 等）。安装 Python 后
  无需安装任何第三方包，无网络/账号/数据库/云 API 需求。
- **数值内核**：全程 `fractions.Fraction` 精确有理运算，不使用第三方
  同类实现，核心算法（拉普拉斯拼装、分块 Schur 补、有理高斯消元、
  内部量恢复）全部在本仓库从零实现。

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

## 准确命令

```bash
python -m unittest discover -s tests -v   # 必要测试（unittest 自动统计用例数）
python demo.py                             # 固定输入演示（约 0.05 秒）
python demo_iteration.py                   # 随机游走期望首达步数演示（约 0.1 秒）
```

## venv 准备

仓库根目录的 `.venv` 是指向运行环境虚拟环境的符号链接；如需自建：

```bash
python3.14 -m venv .venv && source .venv/bin/activate
```

无需 `pip install` 任何内容（仅标准库，无第三方依赖）。

## 随机游走：期望首次到达步数（本轮新增）

把正电导归一成每个节点的转移概率，求到指定**吸收目标集合**的期望
首次到达步数。建立并精确求解 Poisson 方程 `(I - P) h = 1`，**不是**
把有效电阻直接当作时间。

```python
from fractions import Fraction
from resistor_elim import Network, hitting_times

net = Network(["s", "m", "t"])
net.add_conductor("s", "m", 1)
net.add_conductor("m", "t", 1)
net.add_conductor("m", "t", 1)        # 平行边电导相加 -> p(m->t)=2/3
net.set_stay_weight("s", 2)           # 停留权重 -> 自环概率 p(s->s)=2/3

res = hitting_times(net, targets=["t"])
res.expectations["s"]     # 6（精确 Fraction）
res.residuals["s"]        # 0 —— Poisson 方程逐节点残差
res.unreachable           # () —— 与目标不连通的节点清单
```

- **转移概率**：`p(u->v) = g_uv / (stay_u + Σ_w g_uw)`，
  `p(u->u) = stay_u / (stay_u + Σ_w g_uw)`；平行边先按电导相加。
- **停留权重（自环）**：`Network.set_stay_weight(u, w)` 设置每节点
  **非负**权重（`int`/`Fraction`，默认 0；`stay_weight(u)` /
  `stay_weights()` 读取）。它只是随机游走的权重标注：不进入
  `laplacian()`、不改变 `schur_eliminate` 结果；自环**电阻支路**
  仍然被 `add_conductor` 拒绝（旧规则不变）。
- **目标集合吸收**：目标节点期望为 0；空/重复/未知目标抛
  `TargetError`（`PortError` 子类）。
- **不连通分量**：与目标不连通的节点期望为 `math.inf`
  （`resistor_elim.INFINITY`），残差为 `None`，并在
  `unreachable` / `target_components` 中给出连通性诊断——这**不是**
  线性求解失败，不抛 `SingularSystemError`。
- **结果结构** `HittingTimeResult`：`expectations`（每起点期望，
  精确 `Fraction` 或 `math.inf`）、`residuals`（方程残差，精确 0；
  目标/不可达节点为 `None`）、`reachable_transient`、`unreachable`、
  `target_components`，以及 `expectation(node)` / `is_reachable(node)`。
- **验证口径**：路径图/星形图手算分数值；独立有限步概率递推
  （`E[min(T,k)] = Σ P(T>j)` 尾和）给出单调递增下界并逼近期望；
  单目标无自环时验证通勤时间 `= 2 * 总电导 * 有效电阻`，有效电阻
  由旧的 `schur_eliminate` DtN 独立得到。旧端口消元接口与回归测试
  全部保持原状。

`demo.py` 真实计算并打印：一个含内部节点、浮接组件与孤立节点的网络的
消元结果（DtN、对称性/行和检查、内部电势/电流/功率恢复），与 demo 内
**独立编写**的全图线性系统求解逐项比较；随后实际触发一个拒绝边界
（零电阻 `ZeroResistanceError`，并演示 `NaN` 电导被拒绝）。无 sleep、
无预录结果、无压力测试。

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

异常层级：`NetworkError`（基类，携带可定位的 `node`/`edge`）
⊢ `NodeError`、`ConductanceError`、`ResistanceError`
⊢ `ZeroResistanceError`、`PortError`、`PotentialError`、
`LimitError`、`SingularSystemError`、`StayWeightError`；
`TargetError` 是 `PortError` 的子类。

## 数学背景

每个含端口的连通组件按端口 `P` / 内部 `I` 分块：

```
K = [ K_PP K_PI ]      内部 KCL: K_IP V_P + K_II V_I = 0
    [ K_IP K_II ]
V_I = -K_II^{-1} K_IP V_P
I_P = (K_PP - K_PI K_II^{-1} K_IP) V_P = S V_P
```

`S` 即端口 DtN。`K_II` 在“组件连通且至少有一个端口节点、所有电导严格
为正”的前提下正定，故精确可逆；实现不使用数值容差。

## 输入上限与错误语义

- 节点数硬上限 **30**（`resistor_elim.network.MAX_NODES`），超限抛
  `LimitError`。边数受稠密矩阵规模约束（30 节点完全图 435 条）。
- 电导/电阻只接受 `int` 与 `Fraction`：`float` 一律拒绝（其中
  `NaN`/`±Infinity` 给出专门报错），以免二进制浮点被误当精确值。
  需要小数量级请写 `Fraction(17, 10)`。
- 端口电势额外接受**有限** `float`，按 `Fraction(x)` 精确转换；
  `NaN`/`±Infinity` 抛 `PotentialError`。
- 任何位置的 `bool`（`True/False`，虽为 `int` 子类）都被拒绝。
- 边引用未注册节点 → `NodeError`（异常带节点名）；自环、非正电导、
  重复节点标签、空/重复/未知端口、电势数量或键不符均报错，
  **不静默纠正**。
- 失败不留未说明的部分变更：`add_*` 在任何变更前完成全部校验；
  `schur_eliminate` 不修改输入网络。
- 本模块没有“搜索”类算法（是直接的精确消元），因此不存在搜索预算
  概念；矩阵奇异性由精确的零主元判定并抛 `SingularSystemError`，
  与“预算耗尽/未找到”不存在混淆。

## 误差口径（重要）

- **核心结果零浮点误差**：拉普拉斯、Schur 补、逆、内部电势、电流、
  功率全部是精确 `Fraction`；测试用 `assertEqual` 做精确等式断言
  （对称、行和为 0、注入功率 == 耗散功率、与独立参考逐值相等），
  **不使用任何容差**，因此不存在“用宽容差掩盖边界错误”。
- 仅在两处出现浮点：(1) demo 打印时 `float(x)` 仅为人类可读的
  近似显示；(2) 用户主动传入 `float` 端口电势时，按其二进制值精确
  转成有理分数（如 `0.1` → `3602879701896397/36028797018963968`），
  若需要十进制 1/10 请显式传 `Fraction(1, 10)`。
- 大规模网络下有理整数位宽会增长（精确性的代价）；30 节点上限内
  运算仍然很快（测试套件与 demo 均在 1 秒内）。

## 测试与独立性说明

- `tests/reference.py` 是**独立参考**：自行拼装支路表拉普拉斯、自行
  编写高斯消元，并给出串/并联解析式；它不导入被测核心。
- 期望值来自：串/并联解析式、手算分数值（如分压网络）、独立全图
  线性系统参考。“简化后再求解与简化前一致”的往返只作为补充性质，
  不作为期望值来源。

## 设计取舍

- **精确有理优先于浮点速度**：规模 ≤30 时稠密分数消元成本可接受，
  换来可验证的零误差与严格的奇异判定。
- **显式区分浮接组件**：无端口组件无法被激励（电势只确定到组件
  常数），单独标记、支路量返回 `None`，而不是任取一个节点接地或
  静默丢弃——孤立节点同样显式保留。
- **未被电导连接的端口**：DtN 对应为零行零列（该端口无电流），
  不伪造连接关系。
- 平行边电导直接相加（基尔霍夫并联）；无向处理，支路方向仅决定
  `branch_currents` 的符号约定（自 `u` 流向 `v` 为正）。

## 已知限制 / 非目标

- 仅实现**无源、线性、正电导电阻子集**：无零/负电阻、无电源/受控源、
  无电容电感、无二极管、无交流相量、无温度系数；不宣称 SPICE 等行业
  标准兼容。
- 零电阻（理想短路线）被明确拒绝：需要时请在网络定义中先合并等势节点。
- 稠密矩阵 + 精确分数，不针对超大图/稀疏高性能场景。
- 不提供网站、CLI 产品、报表或文件管理功能；仓库终端命令仅用于
  测试与演示。
