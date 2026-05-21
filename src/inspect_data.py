"""真实数据接入体检工具。

这个脚本不训练模型，只检查数据是否适合进入实验流程。
它的作用类似“实验前体检”：先确认 CSV、标签、缺失值和图字段，再正式训练模型。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocess import (
    DESTINATION_IP_COLUMNS,
    DESTINATION_PORT_COLUMNS,
    SOURCE_IP_COLUMNS,
    SOURCE_PORT_COLUMNS,
    add_graph_features,
    detect_label_column,
    find_csv_files,
    load_or_build_dataframe,
    prepare_features_and_labels,
)


def _detect_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """从候选字段中找出真实存在的字段名。"""

    normalized = {column.strip(): column.strip() for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _format_bytes(size: int) -> str:
    """把文件大小转换成人更容易读懂的形式。"""

    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GB"


def _series_top_counts(series: pd.Series, top_k: int = 10) -> dict[str, int]:
    """统计某一列最常见的取值。"""

    counts = series.astype(str).str.strip().value_counts().head(top_k)
    return {str(index): int(value) for index, value in counts.items()}


def _build_recommendations(report: dict[str, Any]) -> list[str]:
    """根据体检结果生成实验建议。"""

    recommendations: list[str] = []

    if report["data_source"] == "demo":
        recommendations.append("当前使用的是演示数据；正式汇报建议换成 CIC-IDS2017 或 Kaggle 清洗版真实 CSV。")

    if report["attack_samples"] == 0:
        recommendations.append("没有检测到攻击样本，无法评价 Recall、F1 和 ROC-AUC；请检查标签列。")

    if report["attack_samples"] > 0 and report["attack_type_count"] < 2:
        recommendations.append("当前只检测到一种攻击类型；若老师关注实验可信度，建议加入 DDoS、WebAttack 或 Infiltration 等更多 CSV 做混合实验。")

    if report["benign_samples"] == 0:
        recommendations.append("没有检测到正常样本，自监督 Autoencoder 无法只用正常流量训练。")

    if report["missing_value_ratio"] > 0.05:
        recommendations.append("缺失值比例超过 5%，建议在汇报中说明缺失值处理方式。")

    if report["duplicate_rows"] > 0:
        recommendations.append("数据中存在重复行，训练前会自动去重，避免评价结果过于乐观。")

    if report["graph_feature_count"] == 0:
        recommendations.append("没有生成图结构特征；如果想体现 GNN/Graph Transformer 思想，建议使用包含 IP 和端口字段的原始数据。")

    if report["feature_count_after_preprocess"] < 8:
        recommendations.append("清洗后的特征数量偏少，建议确认 CSV 是否已经丢失关键网络流量字段。")

    if not recommendations:
        recommendations.append("数据体检结果正常，可以进入模型训练与消融实验。")

    return recommendations


def inspect_dataframe(
    raw_data: pd.DataFrame,
    csv_files: list[Path],
    data_source: str,
    use_graph_features: bool = True,
) -> dict[str, Any]:
    """生成数据体检报告字典。"""

    data = raw_data.copy()
    data.columns = [str(column).strip() for column in data.columns]
    label_column = detect_label_column(data.columns)

    labels = data[label_column].astype(str).str.strip()
    binary_labels = labels.map(lambda value: 0 if value.upper() in {"BENIGN", "NORMAL"} else 1)
    attack_label_counts = {
        label: count
        for label, count in _series_top_counts(labels[binary_labels == 1], top_k=20).items()
    }

    source_ip_column = _detect_column(data.columns, SOURCE_IP_COLUMNS)
    destination_ip_column = _detect_column(data.columns, DESTINATION_IP_COLUMNS)
    source_port_column = _detect_column(data.columns, SOURCE_PORT_COLUMNS)
    destination_port_column = _detect_column(data.columns, DESTINATION_PORT_COLUMNS)

    _, graph_feature_names = add_graph_features(data) if use_graph_features else (data, [])
    features, prepared_labels = prepare_features_and_labels(data, use_graph_features=use_graph_features)

    numeric_data = data.drop(columns=[label_column], errors="ignore").apply(pd.to_numeric, errors="coerce")
    missing_cells = int(data.isna().sum().sum())
    total_cells = int(max(data.shape[0] * max(data.shape[1], 1), 1))
    infinite_cells = int(np.isinf(numeric_data.to_numpy(dtype=float)).sum())

    report: dict[str, Any] = {
        "data_source": data_source,
        "csv_file_count": len(csv_files),
        "csv_files": [
            {
                "path": str(path),
                "size": _format_bytes(path.stat().st_size) if path.exists() else "unknown",
            }
            for path in csv_files
        ],
        "raw_rows": int(len(data)),
        "raw_columns": int(len(data.columns)),
        "duplicate_rows": int(data.duplicated().sum()),
        "label_column": label_column,
        "label_top_counts": _series_top_counts(labels),
        "attack_label_counts": attack_label_counts,
        "attack_type_count": len(attack_label_counts),
        "benign_samples": int((prepared_labels == 0).sum()),
        "attack_samples": int((prepared_labels == 1).sum()),
        "binary_attack_ratio": float(binary_labels.mean()) if len(binary_labels) else 0.0,
        "missing_cells": missing_cells,
        "missing_value_ratio": float(missing_cells / total_cells),
        "infinite_cells": infinite_cells,
        "feature_count_after_preprocess": int(features.shape[1]),
        "graph_feature_count": int(len(graph_feature_names)),
        "graph_feature_names": graph_feature_names,
        "network_columns": {
            "source_ip": source_ip_column,
            "destination_ip": destination_ip_column,
            "source_port": source_port_column,
            "destination_port": destination_port_column,
        },
        "feature_names_preview": list(features.columns[:20]),
    }
    report["recommendations"] = _build_recommendations(report)
    return report


def write_inspection_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """同时保存 JSON 和 Markdown 两种数据体检报告。"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "data_inspection.json"
    markdown_path = output_path / "data_inspection.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_lines = [
        f"- `{item['path']}`：{item['size']}"
        for item in report["csv_files"]
    ] or ["- 未读取真实 CSV，本次使用演示数据。"]
    graph_lines = [
        f"- `{name}`"
        for name in report["graph_feature_names"]
    ] or ["- 未生成图结构特征。"]
    recommendation_lines = [f"- {item}" for item in report["recommendations"]]

    text = f"""# 数据体检报告

## 1. 数据来源

- 数据来源类型：{report["data_source"]}
- CSV 文件数量：{report["csv_file_count"]}

{chr(10).join(csv_lines)}

## 2. 基本规模

- 原始样本数：{report["raw_rows"]}
- 原始字段数：{report["raw_columns"]}
- 重复行数量：{report["duplicate_rows"]}
- 标签列：`{report["label_column"]}`
- 正常样本数：{report["benign_samples"]}
- 攻击样本数：{report["attack_samples"]}
- 攻击样本比例：{report["binary_attack_ratio"]:.4f}
- 攻击类型数量：{report["attack_type_count"]}

## 3. 数据质量

- 缺失单元格数量：{report["missing_cells"]}
- 缺失值比例：{report["missing_value_ratio"]:.4f}
- 无穷值数量：{report["infinite_cells"]}
- 清洗后特征数量：{report["feature_count_after_preprocess"]}

## 4. 网络关系字段

- 源 IP 字段：`{report["network_columns"]["source_ip"]}`
- 目的 IP 字段：`{report["network_columns"]["destination_ip"]}`
- 源端口字段：`{report["network_columns"]["source_port"]}`
- 目的端口字段：`{report["network_columns"]["destination_port"]}`
- 图结构特征数量：{report["graph_feature_count"]}

{chr(10).join(graph_lines)}

## 5. 标签分布 Top 值

```text
{json.dumps(report["label_top_counts"], ensure_ascii=False, indent=2)}
```

攻击类型分布：

```text
{json.dumps(report["attack_label_counts"], ensure_ascii=False, indent=2)}
```

## 6. 建议

{chr(10).join(recommendation_lines)}
"""

    markdown_path.write_text(text, encoding="utf-8")
    return json_path, markdown_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="检查网络入侵检测数据集是否适合训练")
    parser.add_argument("--data-dir", default="data", help="CSV 数据所在目录")
    parser.add_argument("--results-dir", default="results", help="体检报告输出目录")
    parser.add_argument("--max-rows", type=int, default=20000, help="最多读取多少行数据进行体检")
    parser.add_argument("--demo-data", action="store_true", help="没有真实 CSV 时使用演示数据")
    parser.add_argument("--demo-samples", type=int, default=600, help="演示数据样本数")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--no-graph-features", action="store_true", help="关闭 IP/端口轻量图结构特征增强检查")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = parse_args()
    csv_files = find_csv_files(args.data_dir)
    data_source = "csv" if csv_files else "demo"
    raw_data = load_or_build_dataframe(
        data_dir=args.data_dir,
        max_rows=args.max_rows,
        demo_data=args.demo_data,
        demo_samples=args.demo_samples,
        random_state=args.random_state,
    )
    report = inspect_dataframe(
        raw_data=raw_data,
        csv_files=csv_files,
        data_source=data_source,
        use_graph_features=not args.no_graph_features,
    )
    json_path, markdown_path = write_inspection_report(report, args.results_dir)
    print(f"数据体检完成：{json_path.resolve()}")
    print(f"Markdown 报告：{markdown_path.resolve()}")


if __name__ == "__main__":
    main()
