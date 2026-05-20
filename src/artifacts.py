"""实验产物生成工具。

这个文件负责把训练过程中的关键信息整理成报告材料。
简单说，模型负责“算结果”，这里负责“把结果讲清楚”。
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocess import PreparedData, label_summary


def _to_builtin(value: Any) -> Any:
    """把 numpy 类型转换成 Python 原生类型，方便保存成 JSON。"""

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def build_dataset_summary(data: PreparedData) -> dict[str, Any]:
    """生成数据集概况，帮助汇报时说明实验数据规模。"""

    return {
        "train_samples": int(data.x_train.shape[0]),
        "test_samples": int(data.x_test.shape[0]),
        "feature_count": int(data.x_train.shape[1]),
        "graph_feature_count": int(sum(name.startswith("graph_") for name in data.feature_names)),
        "train_label_summary": label_summary(data.y_train),
        "test_label_summary": label_summary(data.y_test),
        "feature_names": data.feature_names,
    }


def save_json(data: dict[str, Any], output_path: str | Path) -> Path:
    """把字典保存成 JSON 文件。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {key: _to_builtin(value) for key, value in data.items()}
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_run_config(args: Namespace, output_dir: str | Path) -> Path:
    """保存本次实验命令参数，方便之后复现实验。"""

    config = {key: _to_builtin(value) for key, value in vars(args).items()}
    return save_json(config, Path(output_dir) / "run_config.json")


def save_dataset_summary(summary: dict[str, Any], output_dir: str | Path) -> Path:
    """保存数据集概况 JSON。"""

    return save_json(summary, Path(output_dir) / "dataset_summary.json")


def _format_number(value: Any) -> str:
    """把指标数字格式化成适合报告展示的文本。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if np.isnan(number):
        return ""
    return f"{number:.4f}"


def _best_model_by_f1(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """根据 F1 选择整体表现最均衡的模型。"""

    return max(metrics, key=lambda item: float(item.get("f1", 0.0)))


def write_experiment_report(
    metrics: list[dict[str, Any]],
    dataset_summary: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """自动生成一份实验报告草稿。"""

    output_path = Path(output_dir) / "experiment_report.md"
    best_model = _best_model_by_f1(metrics)

    rows = []
    for item in metrics:
        rows.append(
            "| {model} | {accuracy} | {precision} | {recall} | {f1} | {roc_auc} |".format(
                model=item.get("model", ""),
                accuracy=_format_number(item.get("accuracy")),
                precision=_format_number(item.get("precision")),
                recall=_format_number(item.get("recall")),
                f1=_format_number(item.get("f1")),
                roc_auc=_format_number(item.get("roc_auc")),
            )
        )

    text = f"""# 实验报告草稿：基于自监督 Transformer 的网络流量异常检测

## 1. 实验目标

本实验把网络入侵检测简化为二分类任务：`BENIGN` 表示正常流量，`Attack` 表示攻击流量。实验对比传统机器学习、普通监督深度学习、普通自编码器和掩码 Transformer 自编码器，观察不同方法在 Recall、F1 和 ROC-AUC 上的表现。

## 2. 数据概况

- 训练集样本数：{dataset_summary["train_samples"]}
- 测试集样本数：{dataset_summary["test_samples"]}
- 特征数量：{dataset_summary["feature_count"]}
- 轻量图结构特征数量：{dataset_summary["graph_feature_count"]}
- 训练集标签分布：{dataset_summary["train_label_summary"]}
- 测试集标签分布：{dataset_summary["test_label_summary"]}

## 3. 方法设计

- `RandomForest` 作为传统机器学习基线，优点是训练速度快，并且能输出特征重要性。
- `MLP` 作为普通深度学习基线，用多层全连接网络学习特征和标签之间的非线性关系。
- `Vanilla Autoencoder` 作为普通自监督对照组，用来观察“只有重构误差、没有 Transformer 注意力机制”的效果。
- `Transformer Autoencoder` 把表格特征看成 feature tokens，用注意力机制学习特征之间的依赖关系。
- `Masked Transformer Autoencoder` 作为掩码重构版本，随机遮住正常流量的一部分特征，再让模型重构完整输入，用来检验 masked reconstruction 是否能提升异常检测效果。
- 自编码器训练仍然只使用正常样本；验证集标签只用于校准异常分数方向和选择阈值，属于“自监督表示学习 + 少量标签阈值校准”。
- 如果原始数据包含 IP 和端口字段，预处理阶段会提取轻量图结构特征，例如源主机出度、目的主机入度、边出现次数和端口访问频率，用来对齐 GNN/Graph Transformer 论文中的网络关系建模思想。

## 4. 实验结果

| 模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

从 F1 指标看，本次实验表现最均衡的模型是 **{best_model.get("model", "")}**。在网络安全场景中，Recall 尤其重要，因为漏报攻击可能造成更严重的风险。

## 5. 可视化产物

- `label_distribution.png`：训练集和测试集的正常/攻击样本数量。
- `feature_importance_randomforest.png`：RandomForest 判断时最看重的特征。
- `model_metric_comparison.png`：不同模型的 Recall 和 F1 对比，用来展示消融实验。
- `threshold_analysis_*.png`：不同阈值策略下 Precision、Recall、F1 的取舍。
- `confusion_matrix_*.png`：每个模型的混淆矩阵。
- `roc_curve_*.png`：每个模型的 ROC 曲线。
- `precision_recall_curve_*.png`：Precision-Recall 曲线，适合攻击样本比例不均衡的情况。
- `score_histogram_*.png`：自编码器重构误差分布。

## 6. 不足与改进

当前项目是课程场景下的轻量实现，重点复现“自监督 + Transformer + 重构误差异常检测”的核心思想，并通过普通 Autoencoder 与 Masked Transformer Autoencoder 的对比做消融实验。后续可以进一步加入 IP、端口和会话关系，构造图结构，再扩展为 Graph Transformer 或 GNN 入侵检测模型。
"""

    output_path.write_text(text, encoding="utf-8")
    return output_path
