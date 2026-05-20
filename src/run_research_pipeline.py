"""一键运行研究型课程项目完整流水线。

这个脚本把前面分散的步骤串起来：
1. 数据体检；
2. 主实验；
3. Mask Ratio 消融实验；
4. 生成提交清单。

对课程汇报来说，它的意义是“可复现”：老师或同学可以按同一条命令重新得到结果。
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from src.artifacts import (
    build_dataset_summary,
    save_dataset_summary,
    save_run_config,
    write_experiment_report,
)
from src.evaluate import (
    plot_label_distribution,
    plot_mask_ratio_ablation,
    plot_metric_comparison,
    save_metrics,
)
from src.inspect_data import inspect_dataframe, write_inspection_report
from src.preprocess import (
    find_csv_files,
    label_summary,
    load_or_build_dataframe,
    load_prepared_data,
)
from src.run_mask_ablation import parse_mask_ratios
from src.train_ae import run_autoencoder, run_vanilla_autoencoder
from src.train_baseline import run_baselines


def _relative_status(path: Path) -> str:
    """返回文件是否已经生成，方便写提交清单。"""

    return "已生成" if path.exists() else "未生成"


def write_submission_checklist(
    results_dir: str | Path,
    include_mask_ablation: bool,
) -> Path:
    """生成最终提交清单，帮助整理 PPT 和项目报告。"""

    output_dir = Path(results_dir)
    output_path = output_dir / "submission_checklist.md"

    required_files = [
        "data_inspection.md",
        "experiment_report.md",
        "metrics.csv",
        "dataset_summary.json",
        "label_distribution.png",
        "model_metric_comparison.png",
        "feature_importance_randomforest.png",
        "confusion_matrix_transformer_autoencoder.png",
        "confusion_matrix_masked_transformer_autoencoder.png",
        "score_histogram_transformer_autoencoder.png",
        "score_histogram_masked_transformer_autoencoder.png",
        "threshold_analysis_transformer_autoencoder.png",
        "threshold_analysis_masked_transformer_autoencoder.png",
    ]
    if include_mask_ablation:
        required_files.extend(["mask_ratio_ablation.csv", "mask_ratio_ablation.png"])

    rows = [
        f"| `{file_name}` | {_relative_status(output_dir / file_name)} |"
        for file_name in required_files
    ]

    text = f"""# 项目提交清单

## 1. 必备结果文件

| 文件 | 状态 |
| --- | --- |
{chr(10).join(rows)}

## 2. PPT 推荐图片

1. `label_distribution.png`：说明正常/攻击样本比例。
2. `model_metric_comparison.png`：说明模型消融对比。
3. `confusion_matrix_transformer_autoencoder.png`：说明误报和漏报。
4. `score_histogram_transformer_autoencoder.png`：说明重构误差为什么能检测异常。
5. `threshold_analysis_transformer_autoencoder.png`：说明阈值选择对 Recall 和 F1 的影响。
6. `mask_ratio_ablation.png`：说明掩码比例不是随便选择的。

## 3. 汇报时建议强调

- 本项目不是只做普通分类，而是做自监督异常检测。
- 通过 `Vanilla Autoencoder`、`Transformer Autoencoder` 和 `Masked Transformer Autoencoder` 做消融实验。
- 通过阈值策略实验解释网络安全中为什么重视 Recall。
- 通过 mask ratio 消融实验解释掩码重构任务的关键超参数。
- 如果真实数据包含 IP/端口字段，项目会生成轻量图结构特征，对应 GNN/Graph Transformer 论文思想。

## 4. 最终检查

- 确认 `data_inspection.md` 中数据来源不是演示数据。
- 确认 `metrics.csv` 中至少包含五个模型：RandomForest、MLP、Vanilla Autoencoder、Transformer Autoencoder、Masked Transformer Autoencoder。
- 确认 PPT 中明确写出“参考论文方法”和“本项目简化/改进点”。
"""

    output_path.write_text(text, encoding="utf-8")
    return output_path


def write_pipeline_manifest(
    args: Namespace,
    inspection_report: dict[str, Any],
    metrics: list[dict[str, Any]],
    mask_metrics: list[dict[str, Any]],
    output_dir: str | Path,
) -> Path:
    """保存流水线总览 JSON，方便之后追踪本次实验配置。"""

    output_path = Path(output_dir) / "pipeline_manifest.json"
    manifest = {
        "pipeline_args": vars(args),
        "data_source": inspection_report["data_source"],
        "raw_rows": inspection_report["raw_rows"],
        "feature_count_after_preprocess": inspection_report["feature_count_after_preprocess"],
        "graph_feature_count": inspection_report["graph_feature_count"],
        "main_models": [item.get("model") for item in metrics],
        "mask_ablation_count": len(mask_metrics),
    }
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="一键运行网络流量异常检测研究流水线")
    parser.add_argument("--data-dir", default="data", help="CSV 数据所在目录")
    parser.add_argument("--results-dir", default="results", help="实验结果输出目录")
    parser.add_argument("--max-rows", type=int, default=20000, help="最多读取多少行真实数据")
    parser.add_argument("--demo-data", action="store_true", help="没有真实 CSV 时使用演示数据")
    parser.add_argument("--demo-samples", type=int, default=2400, help="演示数据样本数")
    parser.add_argument("--test-size", type=float, default=0.25, help="测试集比例")
    parser.add_argument("--epochs", type=int, default=10, help="主实验自编码器训练轮数")
    parser.add_argument("--ablation-epochs", type=int, default=6, help="Mask Ratio 消融实验每组训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--mask-ratio", type=float, default=0.15, help="主实验使用的掩码比例")
    parser.add_argument("--mask-ratios", default="0,0.15,0.3,0.45", help="消融实验的掩码比例列表")
    parser.add_argument("--target-recall", type=float, default=0.9, help="阈值选择时的目标召回率")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--no-graph-features", action="store_true", help="关闭 IP/端口轻量图结构特征增强")
    parser.add_argument("--skip-mask-ablation", action="store_true", help="跳过 Mask Ratio 消融实验")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    use_graph_features = not args.no_graph_features

    print("步骤 1/4：数据体检...")
    csv_files = find_csv_files(args.data_dir)
    data_source = "csv" if csv_files else "demo"
    raw_data = load_or_build_dataframe(
        data_dir=args.data_dir,
        max_rows=args.max_rows,
        demo_data=args.demo_data,
        demo_samples=args.demo_samples,
        random_state=args.random_state,
    )
    inspection_report = inspect_dataframe(
        raw_data=raw_data,
        csv_files=csv_files,
        data_source=data_source,
        use_graph_features=use_graph_features,
    )
    write_inspection_report(inspection_report, results_dir)

    print("步骤 2/4：主实验训练与评价...")
    data = load_prepared_data(
        data_dir=args.data_dir,
        max_rows=args.max_rows,
        demo_data=args.demo_data,
        demo_samples=args.demo_samples,
        test_size=args.test_size,
        random_state=args.random_state,
        use_graph_features=use_graph_features,
    )
    dataset_summary = build_dataset_summary(data)
    save_run_config(args, results_dir)
    save_dataset_summary(dataset_summary, results_dir)
    plot_label_distribution(data.y_train, data.y_test, results_dir)

    print(f"  训练集：{data.x_train.shape[0]} 行，测试集：{data.x_test.shape[0]} 行")
    print(f"  特征数：{data.x_train.shape[1]}")
    print(f"  训练集标签分布：{label_summary(data.y_train)}")
    print(f"  测试集标签分布：{label_summary(data.y_test)}")

    metrics = run_baselines(
        data,
        results_dir=results_dir,
        random_state=args.random_state,
        save_plots=True,
    )
    metrics.append(
        run_vanilla_autoencoder(
            data,
            results_dir=results_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            target_recall=args.target_recall,
            random_state=args.random_state,
            save_plots=True,
        )
    )
    metrics.append(
        run_autoencoder(
            data,
            results_dir=results_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            mask_ratio=0.0,
            target_recall=args.target_recall,
            random_state=args.random_state,
            save_plots=True,
        )
    )
    metrics.append(
        run_autoencoder(
            data,
            results_dir=results_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            mask_ratio=args.mask_ratio,
            target_recall=args.target_recall,
            random_state=args.random_state,
            save_plots=True,
        )
    )
    save_metrics(metrics, results_dir / "metrics.csv")
    plot_metric_comparison(metrics, results_dir)
    write_experiment_report(metrics, dataset_summary, results_dir)

    print("步骤 3/4：Mask Ratio 消融实验...")
    mask_metrics: list[dict[str, Any]] = []
    if args.skip_mask_ablation:
        print("  已跳过 Mask Ratio 消融实验。")
    else:
        for mask_ratio in parse_mask_ratios(args.mask_ratios):
            print(f"  训练 mask_ratio={mask_ratio:.2f}...")
            result = run_autoencoder(
                data,
                results_dir=results_dir,
                epochs=args.ablation_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                mask_ratio=mask_ratio,
                target_recall=args.target_recall,
                random_state=args.random_state,
                save_plots=False,
            )
            result["experiment"] = "mask_ratio_ablation"
            mask_metrics.append(result)
        save_metrics(mask_metrics, results_dir / "mask_ratio_ablation.csv")
        plot_mask_ratio_ablation(mask_metrics, results_dir)

    print("步骤 4/4：生成提交清单...")
    write_submission_checklist(results_dir, include_mask_ablation=not args.skip_mask_ablation)
    write_pipeline_manifest(args, inspection_report, metrics, mask_metrics, results_dir)
    print(f"流水线完成，所有结果已保存到：{results_dir.resolve()}")


if __name__ == "__main__":
    main()
