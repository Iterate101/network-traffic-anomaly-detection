# 参考论文讲解笔记

## 1. GTAE-IDS：Graph Transformer Autoencoder

论文：Jalal Ghadermazi et al. *GTAE-IDS: Graph Transformer-Based Autoencoder Framework for Real-Time Network Intrusion Detection*. IEEE Transactions on Information Forensics and Security, 2025.

这篇论文的核心思想是：网络流量不是孤立的一行表格，而是主机之间不断通信形成的关系结构。作者把通信过程建成图，再用 Graph Transformer Autoencoder 学习正常通信结构。如果一段通信无法被模型很好重构，就可能是异常或攻击。

本项目参考点：

- 使用 Autoencoder 学习正常流量模式。
- 使用 Transformer 的注意力机制捕捉特征之间的关系。
- 使用重构误差作为异常分数。
- 增加 `Transformer Autoencoder` 与 `Vanilla Autoencoder` 的消融对比，观察注意力机制是否提升自监督异常检测效果。

本项目简化点：

- 不完整复现图构建和 Graph Transformer。
- 把每条流量记录的多个数值特征当成一个“特征序列”输入 Transformer。
- 这样实现更轻量，更适合课程演示和初学者理解。
- 当前使用的 CIC-IDS2017 清洗版只包含 `Destination Port`，缺少完整源 IP、目的 IP 和源端口字段，所以图结构部分是轻量特征级复现，不是完整图模型复现。

## 2. Self-Supervised and Interpretable Anomaly Detection Using Network Transformers

论文：Daniel L. Marino et al. *Self-Supervised and Interpretable Anomaly Detection Using Network Transformers*. IEEE Transactions on Industrial Informatics, 2025.

这篇论文强调两个关键词：自监督和可解释。自监督的意思是，模型不完全依赖人工标注攻击样本，而是从数据自身结构中学习规律。可解释的意思是，模型不只给出结果，还要帮助安全人员理解异常来源。

本项目参考点：

- 训练阶段主要学习正常流量，而不是要求大量攻击标签。
- 输出异常分数直方图，让同学能看到正常样本和攻击样本的分布差异。
- 通过 Recall 目标调节阈值，体现安全场景下“尽量少漏报攻击”的需求。
- 增加异常分数方向校准：如果验证集发现原始重构误差方向与攻击标签相反，就把分数取相反数，统一解释为“分数越高越像攻击”。
- 当前实验应准确表述为“自监督表示学习 + 少量标签阈值校准”，因为阈值选择和分数方向校准使用了验证集标签。

## 3. Self-supervised GNN for Network Intrusion Detection

论文：Renjie Xu et al. *Applying self-supervised learning to network intrusion detection for network flows with graph neural network*. Computer Networks, 2024.

这篇论文说明，GNN 可以利用网络流量中的结构关系，例如主机、端口、流之间的连接。它提醒我们：网络安全数据常常有图结构，不应该只把每条流量看成互不相关的表格行。

本项目参考点：

- 在汇报中说明未来可以把当前表格模型扩展为 GNN。
- 当前代码先完成可运行版本；如果数据提供源 IP、目的 IP、源端口和目的端口，预处理会把这些关系转成轻量图统计特征。
- 为避免测试集信息泄漏，图统计量只用训练集拟合，测试集中没见过的 IP 或端口会映射为 0。

## 4. 当前实验结果如何服务论文讲解

最新真实数据实验使用 CIC-IDS2017 清洗版 `Friday-WorkingHours-Afternoon-PortScan` 子集。结果显示：

| 模型 | F1 | ROC-AUC | 讲解价值 |
| --- | ---: | ---: | --- |
| Vanilla Autoencoder | 0.7690 | 0.8840 | 复现普通重构误差异常检测，混合攻击下综合最好 |
| Transformer Autoencoder | 0.6677 | 0.6396 | 说明 Transformer 结构已实现，但复杂模型不一定无条件提升 |
| Masked Transformer Autoencoder | 0.6709 | 0.6824 | Recall 最高，说明掩码重构更偏向少漏报，需要消融验证 |

汇报时不要说“我们完整复现了 GTAE-IDS”。更稳妥的说法是：

> 我们参考 TOP 论文中的自监督、Transformer 和图建模思想，做了课程级轻量复现；其中完整实现的是自编码器重构误差、Transformer 特征 token 建模、阈值策略分析、按攻击类型评价和 mask ratio 消融，图方法部分以端口/IP 统计特征形式简化实现。多攻击实验也说明，复杂模型不一定自动更好，必须用消融实验验证。
