"""一键运行全部实验。

这个脚本会依次训练 RandomForest、MLP 和 Transformer Autoencoder，
并把统一指标表与图片保存到 results 目录。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.artifacts import (
    build_dataset_summary,
    save_dataset_summary,
    save_run_config,
    write_experiment_report,
)
from src.evaluate import plot_label_distribution, plot_metric_comparison, save_metrics
from src.preprocess import label_summary, load_prepared_data
from src.train_ae import run_autoencoder, run_vanilla_autoencoder
from src.train_baseline import run_baselines


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="运行网络流量异常检测完整实验")
    parser.add_argument("--data-dir", default="data", help="CSV 数据所在目录")
    parser.add_argument("--results-dir", default="results", help="实验结果输出目录")
    parser.add_argument("--max-rows", type=int, default=None, help="最多读取多少行真实数据")
    parser.add_argument("--demo-data", action="store_true", help="没有真实 CSV 时使用演示数据")
    parser.add_argument("--demo-samples", type=int, default=2400, help="演示数据样本数")
    parser.add_argument("--test-size", type=float, default=0.25, help="测试集比例")
    parser.add_argument("--epochs", type=int, default=10, help="自编码器训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--mask-ratio", type=float, default=0.15, help="Transformer 自监督训练时随机遮住的特征比例")
    parser.add_argument("--target-recall", type=float, default=0.9, help="阈值选择时的目标召回率")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--no-graph-features", action="store_true", help="关闭 IP/端口轻量图结构特征增强")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_prepared_data(
        data_dir=args.data_dir,
        max_rows=args.max_rows,
        demo_data=args.demo_data,
        demo_samples=args.demo_samples,
        test_size=args.test_size,
        random_state=args.random_state,
        use_graph_features=not args.no_graph_features,
    )
    dataset_summary = build_dataset_summary(data)
    save_run_config(args, results_dir)
    save_dataset_summary(dataset_summary, results_dir)
    plot_label_distribution(data.y_train, data.y_test, results_dir)

    print("数据预处理完成：")
    print(f"  训练集：{data.x_train.shape[0]} 行，测试集：{data.x_test.shape[0]} 行")
    print(f"  特征数：{data.x_train.shape[1]}")
    print(f"  训练集标签分布：{label_summary(data.y_train)}")
    print(f"  测试集标签分布：{label_summary(data.y_test)}")

    print("\n开始训练 RandomForest 和 MLP...")
    metrics = run_baselines(
        data,
        results_dir=results_dir,
        random_state=args.random_state,
        save_plots=True,
    )

    print("\n开始训练 Vanilla Autoencoder 消融对照组...")
    vanilla_metrics = run_vanilla_autoencoder(
        data,
        results_dir=results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        target_recall=args.target_recall,
        random_state=args.random_state,
        save_plots=True,
    )
    metrics.append(vanilla_metrics)

    print("\n开始训练 Masked Transformer Autoencoder...")
    autoencoder_metrics = run_autoencoder(
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
    metrics.append(autoencoder_metrics)

    save_metrics(metrics, results_dir / "metrics.csv")
    plot_metric_comparison(metrics, results_dir)
    write_experiment_report(metrics, dataset_summary, results_dir)
    print(f"\n全部实验完成，指标表和图片已保存到：{results_dir.resolve()}")


if __name__ == "__main__":
    main()
