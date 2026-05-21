# 实验操作指南

这份文档用于帮助你从零跑通实验，并理解每一步为什么要这样做。

## 1. 安装依赖

```powershell
pip install -r requirements.txt
```

这一步会安装：

- `pandas`：读取和处理 CSV 表格。
- `scikit-learn`：训练 RandomForest、MLP，并计算机器学习指标。
- `torch`：训练 Transformer Autoencoder。
- `matplotlib`：生成混淆矩阵、ROC 曲线等图片。

## 2. 先用演示数据跑通

```powershell
python -m src.run_research_pipeline --demo-data --demo-samples 600 --epochs 2 --ablation-epochs 1 --batch-size 64 --mask-ratio 0.15
```

这一步的目的不是得到正式论文结果，而是确认环境、代码、图表输出都正常。

## 3. 使用真实数据

把 CIC-IDS2017 的 CSV 放入 `data/` 目录，然后推荐直接运行完整流水线：

```powershell
python -m src.run_research_pipeline --data-dir data --results-dir results --max-rows 20000 --epochs 10 --ablation-epochs 6 --mask-ratio 0.15
```

完整流水线会生成：

- `results/data_inspection.md`
- `results/data_inspection.json`
- `results/metrics.csv`
- `results/experiment_report.md`
- `results/submission_checklist.md`
- `results/mask_ratio_ablation.csv`
- 多张可视化图片

其中数据体检主要检查：

- 是否找到了 CSV 文件。
- 标签列是不是 `Label` 或类似字段。
- 正常样本和攻击样本比例是否合理。
- 是否存在大量缺失值、重复行或无穷值。
- 是否存在 IP/端口字段，能不能生成轻量图结构特征。

如果你想分步运行，也可以先体检、再训练：

```powershell
python -m src.inspect_data --data-dir data --results-dir results --max-rows 20000
python -m src.run_research_pipeline --data-dir data --results-dir results --max-rows 20000 --epochs 10 --ablation-epochs 6 --mask-ratio 0.15
```

如果电脑运行较慢，可以先把 `--max-rows` 改成 `5000`。这个参数的意思是最多读取多少行数据。

## 4. 结果怎么看

重点看 `results/metrics.csv`：

- `accuracy`：整体准确率。
- `precision`：预测为攻击的样本中，有多少是真的攻击。
- `recall`：真实攻击中，有多少被模型发现。
- `f1`：Precision 和 Recall 的综合指标。
- `roc_auc`：模型区分正常和攻击的整体能力。
- `mask_ratio`：Transformer 训练时被随机遮住的特征比例。

入侵检测场景中，Recall 特别重要。因为如果攻击被漏掉，就可能造成安全风险。

## 5. 汇报时推荐展示的图片

1. `label_distribution.png`  
   用来说明数据集中正常和攻击样本比例。

2. `feature_importance_randomforest.png`  
   用来说明传统模型主要依据哪些特征判断攻击。

3. `confusion_matrix_transformer_autoencoder.png`  
   用来说明自监督模型的误报和漏报情况。

4. `model_metric_comparison.png`  
   用来说明 RandomForest、MLP、Vanilla AE、Transformer AE 和 Masked Transformer AE 之间的消融对比。

5. `score_histogram_transformer_autoencoder.png`  
   用来解释为什么重构误差可以作为异常分数，并说明为什么需要做分数方向校准。

6. `threshold_analysis_transformer_autoencoder.png`  
   用来说明阈值选择会影响 Precision、Recall 和 F1。这个图很适合解释“为什么入侵检测更看重 Recall”。

## 6. 当前真实数据实验结论

当前已使用 CIC-IDS2017 清洗版多攻击混合数据跑通实验，包含 DDoS、PortScan 和 WebAttack。读取 60000 行后，训练集为 44169 行，测试集为 14723 行。

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.9994 | 1.0000 | 0.9984 | 0.9992 | 1.0000 |
| MLP | 0.9974 | 0.9973 | 0.9957 | 0.9965 | 0.9996 |
| Vanilla Autoencoder | 0.7970 | 0.6730 | 0.8971 | 0.7690 | 0.8840 |
| Transformer Autoencoder | 0.6640 | 0.5321 | 0.8960 | 0.6677 | 0.6396 |
| Masked Transformer Autoencoder | 0.6330 | 0.5066 | 0.9930 | 0.6709 | 0.6824 |

讲解时可以这样总结：

- 多攻击混合实验比单 PortScan 更可信，能回应“项目是否太简单”的问题。
- 监督模型接近满分，说明 CIC-IDS2017 的表格特征对传统分类很友好。
- 在自监督模型内部，`Vanilla Autoencoder` 综合 F1 和 ROC-AUC 最好，说明重构误差异常检测本身有效。
- `Masked Transformer Autoencoder` 的 Recall 最高，但 Precision 和 F1 下降，说明掩码重构更偏向少漏报，需要通过消融实验验证，不能假设一定提升。
- `attack_type_metrics.csv` 会展示不同攻击类型的 Recall/F1，适合说明 WebAttack 这类少样本攻击更难检测。
- 项目增加了异常分数方向校准：如果验证集发现“低重构误差更像攻击”，就把分数方向翻转，统一解释为“分数越高越像攻击”。

## 7. 运行 Mask Ratio 消融实验

```powershell
python -m src.run_mask_ablation --demo-data --demo-samples 600 --epochs 2 --mask-ratios 0,0.15,0.3
```

如果使用真实数据：

```powershell
python -m src.run_mask_ablation --data-dir data --max-rows 20000 --epochs 8 --mask-ratios 0,0.15,0.3,0.45
```

输出文件：

- `mask_ratio_ablation.csv`：每个掩码比例对应的指标。
- `mask_ratio_ablation.png`：不同掩码比例的 Recall、F1、ROC-AUC 对比。

这个实验的讲法是：掩码比例太低时，任务可能太简单；掩码比例太高时，输入信息损失太多。当前多攻击实验中 `mask_ratio = 0.45` 的 F1 最好，而 `mask_ratio = 0.15` 的 Recall 最高，说明掩码比例会改变“少漏报”和“少误报”的取舍。

## 8. 轻量图结构特征

如果真实 CSV 中包含下面字段，程序会自动生成图结构统计特征：

- 源 IP：`Source IP` 或 `Src IP`
- 目的 IP：`Destination IP` 或 `Dst IP`
- 源端口：`Source Port` 或 `Src Port`
- 目的端口：`Destination Port` 或 `Dst Port`

这些字段不会直接作为字符串输入模型，而是先转换成图统计量，例如源主机连接数量、目的主机被访问数量、同一 IP 对的连接次数。这样做的原因是：机器学习模型更容易处理数字特征，也更能体现网络关系。

如果想关闭这个功能，可以加：

```powershell
python -m src.run_research_pipeline --data-dir data --no-graph-features
```

## 9. GitHub 上传注意事项

上传 GitHub 时只提交代码和文档，不提交本地数据和生成文件：

- `data/` 中的 CSV 很大，不上传。
- `results/` 中的指标和图片可以重新生成，不上传。
- `presentation_workspace/` 中的 PPT 和预览图不上传。
- 仓库保留 `data/.gitkeep` 和 `results/.gitkeep`，用于说明目录用途。
