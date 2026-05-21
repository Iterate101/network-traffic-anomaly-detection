# 40 分钟汇报大纲

## 论文选读 20 分钟

1. 研究背景，3 分钟  
   说明为什么传统规则检测难以应对新型攻击，以及为什么网络安全需要 AI。

2. GTAE-IDS，8 分钟  
   讲清楚图建模、Transformer、自编码器和重构误差。重点解释：模型只学习正常流量，异常流量会产生更大误差。

3. Network Transformer，6 分钟  
   讲自监督学习和可解释性。重点解释：为什么减少人工标签依赖对网络安全很重要。

4. 文献对比，3 分钟  
   总结三篇论文共同点：都试图从网络结构或流量特征中自动学习异常模式。

## 项目报告 20 分钟

1. 自拟题目与关键问题，3 分钟  
   题目：基于自监督 Transformer 的网络入侵异常检测方法研究与实践。

2. 数据集与预处理，4 分钟  
   说明 CIC-IDS2017 数据来源、二分类标签转换、缺失值处理、标准化。

3. 模型设计，5 分钟  
   介绍 RandomForest、MLP、Vanilla Autoencoder、Transformer Autoencoder、Masked Transformer Autoencoder 五个模型的对比关系。

4. 实验结果，5 分钟  
   展示 `results/metrics.csv`、混淆矩阵、异常分数分布、阈值策略和 mask ratio 消融。重点看 Recall、F1 和 ROC-AUC。

5. 不足与改进，3 分钟  
   当前版本是轻量复现，后续可以加入真实图结构、更多攻击类型、多分类识别和在线检测。

## 当前真实数据结果摘要

当前实验使用 CIC-IDS2017 清洗版多攻击混合数据，包含 DDoS、PortScan 和 WebAttack，读取 60000 行做课程规模实验。

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.9994 | 1.0000 | 0.9984 | 0.9992 | 1.0000 |
| MLP | 0.9974 | 0.9973 | 0.9957 | 0.9965 | 0.9996 |
| Vanilla Autoencoder | 0.7970 | 0.6730 | 0.8971 | 0.7690 | 0.8840 |
| Transformer Autoencoder | 0.6640 | 0.5321 | 0.8960 | 0.6677 | 0.6396 |
| Masked Transformer Autoencoder | 0.6330 | 0.5066 | 0.9930 | 0.6709 | 0.6824 |

汇报结论：

- 监督模型接近满分，说明 CIC-IDS2017 的监督特征区分度较强。
- Vanilla AE 在自监督模型里综合最好，说明重构误差异常检测本身有效。
- Masked Transformer AE 的 Recall 最高，但 Precision 和 F1 下降，说明掩码重构需要通过消融实验判断是否适用。
- 按攻击类型指标可以展示 WebAttack 等少样本攻击更难检测，增强实验可信度。
- 本项目的研究重点不是“监督分类刷高分”，而是复现和分析自监督异常检测方法。

## GitHub 上传范围

- 上传：`src/`、`tests/`、`docs/`、`README.md`、`requirements.txt`、`.gitignore`。
- 不上传：`data/` 真实 CSV、`results/` 自动生成文件、`presentation_workspace/` PPT 工作区。
