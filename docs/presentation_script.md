# 40 分钟汇报讲稿提纲

这份讲稿不是让你逐字照读，而是帮助你把每一页 PPT 要讲什么想清楚。正式汇报时，可以根据老师要求压缩或展开。

## 一、论文选读 20 分钟

### 1. 研究背景，约 3 分钟

可以这样讲：

网络入侵检测的目标，是从大量网络流量中发现攻击行为。传统规则方法依赖人工写规则，面对新型攻击时容易漏报。人工智能方法可以从历史流量中学习规律，所以适合用于异常检测。

需要强调的点：

- 网络攻击变化快，人工规则维护成本高。
- 真实攻击样本标注困难，所以自监督学习有意义。
- 入侵检测更重视 Recall，因为漏报攻击的风险通常大于误报。

### 2. GTAE-IDS 论文，约 8 分钟

可以按“问题、方法、启发”三步讲：

1. 问题：普通表格模型忽略了网络通信关系。
2. 方法：论文把网络通信建成图，再用 Graph Transformer Autoencoder 学习正常通信模式。
3. 启发：本项目不完整复现复杂图模型，而是保留自编码器、Transformer 和异常分数的核心思想。

重点解释：

- Autoencoder（自编码器）会尝试还原输入。
- 正常流量容易被还原，攻击流量更难被还原。
- 一般情况下，重构误差越大，样本越可能异常。
- 但在 PortScan 这类非常规则的攻击中，攻击流量有时反而更容易被还原，所以本项目加入了验证集分数方向校准。

### 3. Network Transformer 论文，约 6 分钟

可以这样讲：

这篇论文强调自监督和可解释。自监督意味着模型不完全依赖攻击标签，而是从数据自身学习正常模式。可解释意味着模型输出不只是“正常/异常”，还应该帮助安全人员理解异常来源。

和本项目的联系：

- 本项目的 Vanilla AE、Transformer AE 和 Masked Transformer AE 都只用正常样本训练。
- 通过异常分数直方图展示正常和攻击样本的分布差异。
- 通过阈值调节优先保证 Recall。
- 验证集标签只用于校准异常分数方向和选择阈值，所以应表述为“自监督表示学习 + 少量标签阈值校准”。

### 4. 文献对比，约 3 分钟

可以总结为：

三篇论文都在解决同一个核心问题：如何让模型自动学习网络流量中的异常模式。区别是，有的强调图结构，有的强调 Transformer，有的强调自监督学习。本项目选择其中最适合课程实践的部分进行轻量复现。

可以补一句突出改进：

本项目加入了两层研究型增强：第一，把表格特征当成 feature token，用 Transformer 学习特征依赖；第二，加入 Masked Feature Reconstruction，并通过 mask ratio 消融验证它是否真的适合当前数据。

## 二、项目报告 20 分钟

### 1. 题目和关键问题，约 3 分钟

题目：基于自监督 Transformer 的网络流量异常检测。

关键问题：

- 攻击样本标注成本高。
- 新型攻击难以及时覆盖。
- 需要在尽量少漏报的前提下完成检测。

### 2. 数据与预处理，约 4 分钟

讲清楚下面几件事：

- 数据来自 CIC-IDS2017 或演示数据。
- 正式实验前用 `run_research_pipeline.py` 统一完成数据体检、模型训练、消融实验和提交清单生成。
- 标签被转换为二分类：`BENIGN` 和 `Attack`。
- 缺失值用中位数填充。
- 特征做标准化，让模型训练更稳定。
- 如果数据包含 IP 和端口字段，会提取轻量图结构特征，例如源主机出度、目的主机入度和边出现次数。

可以展示：

- `data_inspection.md` 中的数据规模和字段检查结果
- `label_distribution.png`
- `dataset_summary.json` 中的样本数量

图特征可以这样讲：

完整 GNN 复现工作量较大，所以我们把图思想做成特征级复现。把 IP 看成节点，把流量看成边，再统计节点度数和边出现频率，让模型获得网络关系信息。

### 3. 模型设计，约 5 分钟

讲五个模型的分工：

- RandomForest：传统机器学习基线。
- MLP：普通深度学习基线。
- Vanilla Autoencoder：普通自编码器消融对照。
- Transformer Autoencoder：加入注意力机制的自监督异常检测模型。
- Masked Transformer Autoencoder：在 Transformer 基础上加入掩码重构任务，用来验证 masked reconstruction 是否有效。

重点讲 Transformer Autoencoder：

1. 输入是一条网络流量的多个数值特征。
2. 模型把每个特征看成一个 token。
3. Transformer 学习特征之间的关系。
4. 解码器尝试重构完整特征。
5. 根据重构误差得到异常分数。
6. 经过方向校准和阈值选择后，把样本判为正常或攻击。

再补充 Masked 版本：

- Masked 版本训练时随机遮住一部分特征。
- 它的目标是让模型学习上下文，而不是简单复制输入。
- 但真实实验发现，本数据集上 `mask_ratio = 0` 的 Transformer AE 更好，所以掩码重构不是无条件提升，这正是消融实验的意义。

### 4. 实验结果，约 5 分钟

建议展示：

- `metrics.csv`
- `model_metric_comparison.png`
- 一个混淆矩阵
- `score_histogram_transformer_autoencoder.png`
- `threshold_analysis_transformer_autoencoder.png`
- 如果时间允许，再展示 `mask_ratio_ablation.png`

讲指标时可以这样说：

Accuracy 表示整体判断正确率。Precision 表示模型报出的攻击有多少是真的。Recall 表示真实攻击有多少被模型找出来。F1 是 Precision 和 Recall 的综合指标。

阈值策略可以这样讲：

自编码器输出的是异常分数，所以必须选择一个阈值。阈值低，Recall 可能更高，但误报也可能增加；阈值高，误报可能减少，但漏报风险增加。网络安全里漏报攻击更危险，所以我们加入了目标 Recall 阈值策略。

当前真实数据结果可以这样讲：

- RandomForest 和 MLP 接近满分，说明 PortScan 子集用监督分类很容易分开。
- Vanilla AE 的 F1 是 0.8535，ROC-AUC 是 0.7124。
- Transformer AE 的 F1 是 0.8660，ROC-AUC 是 0.8595，说明注意力机制确实提升了自监督模型。
- Masked Transformer AE 的 Recall 是 0.9759，但 F1 降到 0.7867，说明它更偏向少漏报，但误报增加。

### 5. 不足与改进，约 3 分钟

可以诚实说明：

- 当前实现是轻量复现，没有完整实现 Graph Transformer。
- 当前 CIC-IDS2017 清洗版缺少完整 IP 字段，所以图结构只做到端口统计和轻量图特征。
- 后续可以加入 IP、端口、会话关系，构造图结构，用 GNN 或 Graph Transformer 改进。

## 三、上传 GitHub 时怎么说明

可以这样写在仓库说明或提交备注里：

本仓库只上传代码、测试和文档。真实 CIC-IDS2017 CSV、`results/` 实验输出和 `presentation_workspace/` PPT 文件不上传，因为它们体积较大且可以由流水线重新生成。复现实验时，请先把 CSV 放入 `data/`，再运行：

```powershell
python -m src.run_research_pipeline --data-dir data --max-rows 20000 --epochs 10 --ablation-epochs 6
```
