"""运行 Mask Ratio 消融实验。

Mask Ratio（掩码比例）表示训练时随机遮住多少特征。
这个脚本用来观察不同掩码比例对自监督 Transformer 异常检测效果的影响。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluate import plot_mask_ratio_ablation, save_metrics
from src.preprocess import label_summary, load_prepared_data
from src.train_ae import run_autoencoder


def parse_mask_ratios(text: str) -> list[float]:
    """把命令行中的逗号分隔字符串转换成浮点数列表。"""

    ratios = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not ratios:
        raise ValueError("至少需要提供一个 mask ratio。")
    for ratio in ratios:
        if ratio < 0 or ratio >= 1:
            raise ValueError("mask ratio 必须在 [0, 1) 范围内。")
    return ratios


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="运行 Mask Ratio 消融实验")
    parser.add_argument("--data-dir", default="data", help="CSV 数据所在目录")
    parser.add_argument("--results-dir", default="results", help="实验结果输出目录")
    parser.add_argument("--max-rows", type=int, default=None, help="最多读取多少行真实数据")
    parser.add_argument("--demo-data", action="store_true", help="没有真实 CSV 时使用演示数据")
    parser.add_argument("--demo-samples", type=int, default=2400, help="演示数据样本数")
    parser.add_argument("--test-size", type=float, default=0.25, help="测试集比例")
    parser.add_argument("--epochs", type=int, default=8, help="每个掩码比例训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--mask-ratios", default="0,0.15,0.3,0.45", help="逗号分隔的掩码比例列表")
    parser.add_argument("--target-recall", type=float, default=0.9, help="阈值选择时的目标召回率")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--no-graph-features", action="store_true", help="关闭 IP/端口轻量图结构特征增强")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    mask_ratios = parse_mask_ratios(args.mask_ratios)

    data = load_prepared_data(
        data_dir=args.data_dir,
        max_rows=args.max_rows,
        demo_data=args.demo_data,
        demo_samples=args.demo_samples,
        test_size=args.test_size,
        random_state=args.random_state,
        use_graph_features=not args.no_graph_features,
    )

    print("Mask Ratio 消融实验数据准备完成：")
    print(f"  训练集：{data.x_train.shape[0]} 行，测试集：{data.x_test.shape[0]} 行")
    print(f"  特征数：{data.x_train.shape[1]}")
    print(f"  训练集标签分布：{label_summary(data.y_train)}")
    print(f"  测试集标签分布：{label_summary(data.y_test)}")

    metrics = []
    for mask_ratio in mask_ratios:
        print(f"\n开始训练 mask_ratio={mask_ratio:.2f} 的 Transformer Autoencoder...")
        result = run_autoencoder(
            data,
            results_dir=results_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            mask_ratio=mask_ratio,
            target_recall=args.target_recall,
            random_state=args.random_state,
            save_plots=False,
        )
        result["experiment"] = "mask_ratio_ablation"
        metrics.append(result)

    save_metrics(metrics, results_dir / "mask_ratio_ablation.csv")
    plot_mask_ratio_ablation(metrics, results_dir)
    print(f"\nMask Ratio 消融实验完成，结果已保存到：{results_dir.resolve()}")


if __name__ == "__main__":
    main()
