"""模型评价与可视化工具。

这里统一计算 Accuracy、Precision、Recall、F1、ROC-AUC，并保存汇报可用的图片。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _configure_matplotlib_font() -> bool:
    """配置 Matplotlib 字体，并返回当前环境是否有可用中文字体。"""

    candidate_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    installed_fonts = {
        font_manager.FontProperties(fname=font_path).get_name()
        for font_path in font_manager.findSystemFonts()
    }

    for font_name in candidate_fonts:
        if font_name in installed_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True

    plt.rcParams["axes.unicode_minus"] = False
    return False


HAS_CHINESE_FONT = _configure_matplotlib_font()


def _label(chinese: str, english: str) -> str:
    """有中文字体时使用中文，否则使用英文，避免图片标题乱码。"""

    return chinese if HAS_CHINESE_FONT else english


def safe_name(model_name: str) -> str:
    """把模型名称转换成适合文件名使用的小写字符串。"""

    return model_name.lower().replace(" ", "_").replace("/", "_")


def classification_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray | None = None,
    threshold: float | None = None,
) -> dict[str, float | str]:
    """计算二分类常用指标。

    Precision（精确率）关注“报出来的攻击有多少是真的”。
    Recall（召回率）关注“真实攻击中有多少被找出来”，入侵检测通常很重视它。
    """

    result: dict[str, float | str] = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if threshold is not None:
        result["threshold"] = float(threshold)

    if scores is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = roc_auc_score(y_true, scores)
    else:
        result["roc_auc"] = float("nan")

    return result


def orient_anomaly_scores(
    y_valid: np.ndarray,
    valid_scores: np.ndarray,
    test_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """用验证集校准异常分数方向。

    Autoencoder 的常见假设是“重构误差越大越异常”。但在某些清洗版数据中，
    攻击流量可能更规则，反而更容易被还原。这里用验证集 ROC-AUC 判断方向：
    如果原始分数的 AUC 小于 0.5，就说明分数方向基本反了，需要取相反数。
    """

    if len(np.unique(y_valid)) < 2:
        return valid_scores, test_scores, "raw_reconstruction_error", float("nan")

    validation_auc = float(roc_auc_score(y_valid, valid_scores))
    if validation_auc >= 0.5:
        return valid_scores, test_scores, "raw_reconstruction_error", validation_auc

    # 取相反数后，仍然保持“分数越大越像攻击”的统一接口，后续阈值代码不用改变。
    return -valid_scores, -test_scores, "inverted_reconstruction_error", validation_auc


def choose_threshold_for_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_recall: float = 0.9,
) -> float:
    """根据验证集选择阈值，优先保证较高召回率。

    scores 越大表示越像攻击。函数会寻找 Recall 达到目标值的候选阈值，
    再从中选择 F1 最好的那个，避免模型为了追求召回率而把太多正常流量误报为攻击。
    """

    if len(np.unique(y_true)) < 2:
        return float(np.quantile(scores, 0.95))

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return float(np.quantile(scores, 0.95))

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for index, threshold in enumerate(thresholds):
        current_precision = precision[index]
        current_recall = recall[index]
        if current_recall < target_recall:
            continue

        current_f1 = 0.0
        if current_precision + current_recall > 0:
            current_f1 = 2 * current_precision * current_recall / (current_precision + current_recall)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)

    if best_f1 >= 0:
        return best_threshold

    # 如果没有阈值达到目标召回率，就退而求其次，选择整体 F1 最好的阈值。
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1_values))])


def choose_threshold_for_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    """选择让 F1 最高的阈值。"""

    if len(np.unique(y_true)) < 2:
        return float(np.quantile(scores, 0.95))

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return float(np.quantile(scores, 0.95))

    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1_values))])


def _normal_quantile_threshold(y_true: np.ndarray, scores: np.ndarray, quantile: float) -> float:
    """只根据正常样本分数选择阈值，模拟无攻击标签场景。"""

    normal_scores = scores[y_true == 0]
    if len(normal_scores) == 0:
        normal_scores = scores
    return float(np.quantile(normal_scores, quantile))


def build_threshold_analysis(
    model_name: str,
    y_valid: np.ndarray,
    valid_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    target_recall: float = 0.9,
) -> list[dict[str, float | str]]:
    """生成多种阈值策略的测试集效果对比。"""

    strategies = [
        (
            f"target_recall_{target_recall:.2f}",
            choose_threshold_for_recall(y_valid, valid_scores, target_recall=target_recall),
            "优先达到目标 Recall，适合更重视少漏报的安全场景。",
        ),
        (
            "best_f1",
            choose_threshold_for_f1(y_valid, valid_scores),
            "在验证集上选择 F1 最高的阈值，适合追求 Precision 和 Recall 平衡。",
        ),
        (
            "normal_95_percentile",
            _normal_quantile_threshold(y_valid, valid_scores, 0.95),
            "只参考正常样本 95 分位数，模拟攻击标签较少的场景。",
        ),
        (
            "normal_99_percentile",
            _normal_quantile_threshold(y_valid, valid_scores, 0.99),
            "只参考正常样本 99 分位数，通常更保守、误报更少。",
        ),
    ]

    rows: list[dict[str, float | str]] = []
    for strategy_name, threshold, description in strategies:
        predictions = (test_scores >= threshold).astype(int)
        metrics = classification_metrics(
            model_name,
            y_test,
            predictions,
            scores=test_scores,
            threshold=threshold,
        )
        metrics["strategy"] = strategy_name
        metrics["description"] = description
        rows.append(metrics)

    return rows


def save_metrics(metrics: list[dict[str, float | str]], output_path: str | Path) -> None:
    """保存指标表，便于直接放入汇报 PPT。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(path, index=False, encoding="utf-8-sig")

    json_path = path.with_suffix(".json")
    json_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def plot_confusion_matrix(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
) -> Path:
    """保存混淆矩阵图片。"""

    output_path = Path(output_dir) / f"confusion_matrix_{safe_name(model_name)}.png"
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    display = ConfusionMatrixDisplay(matrix, display_labels=["BENIGN", "Attack"])
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{model_name} {_label('混淆矩阵', 'Confusion Matrix')}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_roc_curve(
    model_name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    output_dir: str | Path,
) -> Path | None:
    """保存 ROC 曲线；如果测试集中只有一个类别，就不生成。"""

    if len(np.unique(y_true)) < 2:
        return None

    output_path = Path(output_dir) / f"roc_curve_{safe_name(model_name)}.png"
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    RocCurveDisplay.from_predictions(y_true, scores, ax=ax, name=model_name)
    ax.set_title(f"{model_name} {_label('ROC 曲线', 'ROC Curve')}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_precision_recall_curve(
    model_name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    output_dir: str | Path,
) -> Path | None:
    """保存 Precision-Recall 曲线，适合类别不平衡的入侵检测场景。"""

    if len(np.unique(y_true)) < 2:
        return None

    output_path = Path(output_dir) / f"precision_recall_curve_{safe_name(model_name)}.png"
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    PrecisionRecallDisplay.from_predictions(y_true, scores, ax=ax, name=model_name)
    ax.set_title(f"{model_name} {_label('Precision-Recall 曲线', 'Precision-Recall Curve')}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_score_histogram(
    model_name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    output_dir: str | Path,
) -> Path:
    """保存异常分数直方图，用来解释自编码器如何区分正常和攻击。"""

    output_path = Path(output_dir) / f"score_histogram_{safe_name(model_name)}.png"
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(scores[y_true == 0], bins=35, alpha=0.72, label="BENIGN")
    ax.hist(scores[y_true == 1], bins=35, alpha=0.72, label="Attack")
    ax.axvline(threshold, color="red", linestyle="--", label=_label("阈值", "Threshold"))
    ax.set_title(f"{model_name} {_label('异常分数分布', 'Anomaly Score Distribution')}")
    ax.set_xlabel(_label("重构误差 / 攻击分数", "Reconstruction Error / Attack Score"))
    ax.set_ylabel(_label("样本数量", "Sample Count"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_threshold_analysis(
    model_name: str,
    threshold_rows: list[dict[str, float | str]],
    output_dir: str | Path,
) -> Path:
    """保存阈值策略对比表。"""

    output_path = Path(output_dir) / f"threshold_analysis_{safe_name(model_name)}.csv"
    pd.DataFrame(threshold_rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def plot_threshold_analysis(
    model_name: str,
    threshold_rows: list[dict[str, float | str]],
    output_dir: str | Path,
) -> Path:
    """保存阈值策略对比图，展示 Precision、Recall、F1 的取舍。"""

    output_path = Path(output_dir) / f"threshold_analysis_{safe_name(model_name)}.png"
    strategies = [str(row["strategy"]) for row in threshold_rows]
    precision_values = [float(row.get("precision", 0.0)) for row in threshold_rows]
    recall_values = [float(row.get("recall", 0.0)) for row in threshold_rows]
    f1_values = [float(row.get("f1", 0.0)) for row in threshold_rows]

    x = np.arange(len(strategies))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - width, precision_values, width, label="Precision", color="#4c78a8")
    ax.bar(x, recall_values, width, label="Recall", color="#f58518")
    ax.bar(x + width, f1_values, width, label="F1", color="#54a24b")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(_label("指标值", "Score"))
    ax.set_title(f"{model_name} {_label('阈值策略对比', 'Threshold Strategy Comparison')}")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=18, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_label_distribution(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    output_dir: str | Path,
) -> Path:
    """保存训练集和测试集的标签分布图。"""

    output_path = Path(output_dir) / "label_distribution.png"
    train_counts = [int(np.sum(train_labels == 0)), int(np.sum(train_labels == 1))]
    test_counts = [int(np.sum(test_labels == 0)), int(np.sum(test_labels == 1))]

    x = np.arange(2)
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.bar(x - width / 2, train_counts, width, label=_label("训练集", "Train"))
    ax.bar(x + width / 2, test_counts, width, label=_label("测试集", "Test"))
    ax.set_xticks(x)
    ax.set_xticklabels(["BENIGN", "Attack"])
    ax.set_ylabel(_label("样本数量", "Sample Count"))
    ax.set_title(_label("标签分布", "Label Distribution"))
    ax.legend()

    for index, count in enumerate(train_counts):
        ax.text(index - width / 2, count, str(count), ha="center", va="bottom")
    for index, count in enumerate(test_counts):
        ax.text(index + width / 2, count, str(count), ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    output_dir: str | Path,
    top_k: int = 12,
) -> Path:
    """保存 RandomForest 的特征重要性表格。"""

    output_path = Path(output_dir) / "feature_importance_randomforest.csv"
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    frame.head(top_k).to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    output_dir: str | Path,
    top_k: int = 12,
) -> Path:
    """保存 RandomForest 特征重要性图片，用来解释模型判断依据。"""

    output_path = Path(output_dir) / "feature_importance_randomforest.png"
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    top_features = frame.head(top_k).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.barh(top_features["feature"], top_features["importance"], color="#2f6f9f")
    ax.set_xlabel(_label("重要性", "Importance"))
    ax.set_title(_label("RandomForest 特征重要性", "RandomForest Feature Importance"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_metric_comparison(
    metrics: list[dict[str, float | str]],
    output_dir: str | Path,
) -> Path:
    """保存模型 Recall 和 F1 对比图，方便展示消融实验结果。"""

    output_path = Path(output_dir) / "model_metric_comparison.png"
    model_names = [str(item["model"]) for item in metrics]
    recall_values = [float(item.get("recall", 0.0)) for item in metrics]
    f1_values = [float(item.get("f1", 0.0)) for item in metrics]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, recall_values, width, label="Recall", color="#2f6f9f")
    ax.bar(x + width / 2, f1_values, width, label="F1", color="#d9822b")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(_label("指标值", "Score"))
    ax.set_title(_label("模型消融对比", "Model Ablation Comparison"))
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_mask_ratio_ablation(
    metrics: list[dict[str, float | str]],
    output_dir: str | Path,
) -> Path:
    """保存 mask ratio 消融实验图。"""

    output_path = Path(output_dir) / "mask_ratio_ablation.png"
    sorted_metrics = sorted(metrics, key=lambda item: float(item.get("mask_ratio", 0.0)))
    mask_ratios = [float(item.get("mask_ratio", 0.0)) for item in sorted_metrics]
    recall_values = [float(item.get("recall", 0.0)) for item in sorted_metrics]
    f1_values = [float(item.get("f1", 0.0)) for item in sorted_metrics]
    auc_values = [float(item.get("roc_auc", 0.0)) for item in sorted_metrics]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(mask_ratios, recall_values, marker="o", label="Recall", color="#f58518")
    ax.plot(mask_ratios, f1_values, marker="o", label="F1", color="#54a24b")
    ax.plot(mask_ratios, auc_values, marker="o", label="ROC-AUC", color="#4c78a8")
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Mask Ratio")
    ax.set_ylabel(_label("指标值", "Score"))
    ax.set_title(_label("Mask Ratio 消融实验", "Mask Ratio Ablation"))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
