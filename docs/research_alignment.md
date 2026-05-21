# TOP 论文方法对应与项目改进说明

这份文档专门回答老师可能会问的问题：**你们到底参考了论文的什么方法？是简单调用模型，还是做了复现或改进？**

## 1. 本项目不是简单分类，而是异常检测

普通分类模型会直接学习“正常/攻击”的标签边界。但网络安全场景中，新型攻击往往没有足够标签，所以只依赖监督分类不够可靠。

本项目的核心方法是自监督异常检测：

1. 只使用正常流量训练 Autoencoder。
2. 模型学习正常流量的特征模式。
3. 测试阶段计算重构误差。
4. 用验证集校准异常分数方向，再根据阈值判断正常或攻击。

这和 GTAE-IDS、Network Transformer 类论文中的异常检测思想是一致的。

## 2. 参考论文方法一：Autoencoder 重构误差

参考点：

- GTAE-IDS 使用 Autoencoder 框架学习正常网络行为。
- 异常样本因为不符合正常模式，重构误差通常更大。

本项目实现：

- `Vanilla Autoencoder`：普通全连接自编码器。
- `Transformer Autoencoder`：带注意力机制的自编码器。
- `Masked Transformer Autoencoder`：在 Transformer 自编码器基础上加入掩码重构任务。
- 三个自编码器模型都使用重构误差作为异常分数。

为什么要加普通 Autoencoder：

这是消融实验。它用来回答“Transformer 是否真的有必要”。如果只做 Transformer，没有普通 Autoencoder 对照，研究说服力会弱。当前多攻击混合实验中，`Vanilla Autoencoder` 的综合 F1 和 ROC-AUC 高于 Transformer 版本，说明复杂结构不是无条件提升，必须通过消融实验验证。

## 3. 参考论文方法二：Transformer 注意力机制

参考点：

- Network Transformer 类方法强调用 Transformer 捕捉网络数据中的复杂依赖。
- Transformer 的注意力机制可以学习不同特征之间的关系。

本项目实现：

- 把一条流量记录中的多个数值特征看成一串 feature tokens。
- 每个特征先映射成向量，再加入特征位置嵌入。
- 使用 Transformer Encoder 建模特征之间的依赖。
- 最后把每个特征 token 还原成原始数值。

简化说明：

原论文可能使用更复杂的网络结构或图结构。本项目为了适合课程实践，把图结构简化为表格特征序列，但保留了 Transformer 和自编码器的关键思想。

## 4. 本项目改进点：Masked Feature Reconstruction

这是本项目相比最初版本的重要增强，但它不是无条件有效，需要通过消融实验验证。

做法：

1. 训练时随机遮住一部分正常流量特征。
2. 被遮住的特征用 0 替代，因为标准化后的 0 接近平均水平。
3. 模型必须根据其它未遮住特征恢复完整输入。
4. 这样模型不是简单复制输入，而是学习特征之间的上下文关系。

这个思想和自监督学习中的 masked reconstruction 很接近，比如 NLP 中遮住词让模型预测，表格异常检测中遮住特征让模型恢复。

当前多攻击实验中，`Masked Transformer Autoencoder` 的 Recall 最高，但 Precision 和 F1 下降。这说明本项目不是简单宣称“加掩码就更好”，而是通过实验得到结论：掩码重构更偏向少漏报，但会带来更多误报，需要根据安全目标选择。

## 5. 实验如何体现“复现或改进”

本项目用下面几组模型做对比：

| 模型 | 作用 |
| --- | --- |
| RandomForest | 传统机器学习基线 |
| MLP | 监督深度学习基线 |
| Vanilla Autoencoder | 复现普通重构误差异常检测思想 |
| Transformer Autoencoder | 加入 Transformer 注意力机制，检验特征依赖建模是否有效 |
| Masked Transformer Autoencoder | 在 Transformer 基础上加入掩码重构任务，检验 masked reconstruction 是否有效 |

汇报时可以这样说：

我们没有完整复现 GTAE-IDS 的复杂图构建，而是做了轻量复现和方法改进：先复现 Autoencoder 重构误差异常检测，再加入 Transformer 注意力机制和 Masked Feature Reconstruction，并通过消融实验比较“普通 AE、无掩码 Transformer、掩码 Transformer”的差异。

## 6. 新增研究实验：阈值策略对比

自编码器输出的是异常分数，而不是天然的“正常/攻击”标签。因此，阈值怎么选是异常检测中的关键问题。

另外，不同数据子集上“重构误差”的方向可能不完全一致。经典异常检测通常假设重构误差越大越异常，但像 PortScan 这类行为有时非常规则，模型可能反而更容易还原攻击流量。为避免 ROC-AUC 被分数方向影响，本项目增加了验证集分数方向校准：如果验证集上原始重构误差的 ROC-AUC 小于 0.5，就把异常分数取相反数，统一成“分数越大越像攻击”。这一步不会改变模型训练方式，只影响评价阶段的分数解释和阈值选择。

本项目比较四种阈值策略：

| 阈值策略 | 含义 |
| --- | --- |
| target_recall | 优先达到目标 Recall，减少漏报 |
| best_f1 | 在验证集上选择 F1 最优阈值 |
| normal_95_percentile | 使用正常样本异常分数的 95 分位数 |
| normal_99_percentile | 使用正常样本异常分数的 99 分位数 |

这个实验能说明：网络安全场景不是只看 Accuracy，而要根据安全目标选择阈值。

## 7. 当前真实数据结果

当前实验使用 CIC-IDS2017 清洗版多攻击混合数据，包含 DDoS、PortScan 和 WebAttack。读取 60000 行后，训练集为 44169 行，测试集为 14723 行。

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.9994 | 1.0000 | 0.9984 | 0.9992 | 1.0000 |
| MLP | 0.9974 | 0.9973 | 0.9957 | 0.9965 | 0.9996 |
| Vanilla Autoencoder | 0.7970 | 0.6730 | 0.8971 | 0.7690 | 0.8840 |
| Transformer Autoencoder | 0.6640 | 0.5321 | 0.8960 | 0.6677 | 0.6396 |
| Masked Transformer Autoencoder | 0.6330 | 0.5066 | 0.9930 | 0.6709 | 0.6824 |

这组结果可以支撑三个汇报结论：

- 监督模型很强，但这不等于项目只是普通分类，因为自监督模型没有用攻击样本训练表示。
- `Vanilla Autoencoder` 在混合攻击中综合最好，说明重构误差异常检测本身有效。
- `Masked Transformer Autoencoder` 的 Recall 最高但 F1 更低，说明掩码重构需要根据数据集和安全目标验证。
- `attack_type_metrics.csv` 进一步按攻击类型评价，能说明 DDoS、PortScan 和 WebAttack 的检测难度差异。

## 8. 新增研究实验：Mask Ratio 消融

Masked Feature Reconstruction 中，mask ratio 表示随机遮住多少比例的特征。

本项目提供 `run_mask_ablation.py`，可以比较多个掩码比例：

```powershell
python -m src.run_mask_ablation --data-dir data --max-rows 20000 --epochs 8 --mask-ratios 0,0.15,0.3,0.45
```

汇报时可以这样解释：

我们通过消融实验观察掩码比例对检测效果的影响。如果掩码比例太低，模型可能只是学习复制输入；如果掩码比例太高，模型看到的信息太少，重构任务会变得过难。因此，mask ratio 是一个需要通过实验选择的关键超参数。

## 9. 新增方法增强：轻量图结构特征

GNN 和 Graph Transformer 论文强调：网络流量不是孤立样本，而是主机之间的通信关系。

完整图神经网络实现需要构造节点、边、邻接关系和图批处理，课程项目时间成本较高。为了保留图方法思想，本项目实现了轻量图结构特征：

| 图特征 | 含义 |
| --- | --- |
| `graph_src_out_count` | 某个源 IP 发起的连接数量，近似源节点出度 |
| `graph_dst_in_count` | 某个目的 IP 接收的连接数量，近似目的节点入度 |
| `graph_edge_count` | 同一组源 IP 到目的 IP 的连接出现次数 |
| `graph_src_unique_dst` | 某个源 IP 访问过多少不同目的 IP |
| `graph_dst_unique_src` | 某个目的 IP 被多少不同源 IP 访问过 |
| `graph_dst_port_count` | 某个目的端口被访问的频率 |

这属于“图思想的特征级复现”：不训练完整 GNN，但把网络关系信息注入表格模型和 Transformer Autoencoder 中。汇报时可以把它作为对 GNN/Graph Transformer 论文的工程化简化。

为了避免实验结果虚高，当前训练流水线不是在全量数据上先统计图特征，而是先划分训练集和测试集，再只用训练集拟合 IP、端口和边的统计映射。测试集中没见过的 IP 或端口会被映射为 0，这样可以避免测试集关系信息提前泄漏到训练过程。

## 10. 新增工程步骤：真实数据体检

为了让实验结果更可信，本项目提供 `inspect_data.py` 在训练前检查数据：

```powershell
python -m src.inspect_data --data-dir data --results-dir results --max-rows 20000
```

它会输出：

- `data_inspection.md`：适合直接阅读和放进汇报材料。
- `data_inspection.json`：结构化数据体检结果。

这个步骤能证明项目不是只在演示数据上运行，而是对真实 CSV 的标签、缺失值、重复行、攻击比例和图结构字段做了检查。

## 11. GitHub 上传说明

为了让仓库干净，上传 GitHub 时只提交代码、测试和说明文档：

- 提交：`src/`、`tests/`、`docs/`、`README.md`、`requirements.txt`、`.gitignore`。
- 不提交：`data/` 中的真实 CSV、`results/` 中的图表和指标、`presentation_workspace/` 中的 PPT 文件。

这样做不是删除实验结果，而是把项目做成“可复现仓库”：别人克隆代码后，把数据放入 `data/`，运行流水线即可重新生成 `results/`。
