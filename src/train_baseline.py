"""训练传统基线模型：RandomForest 和 MLP。

这两个模型用于和 Transformer Autoencoder 对比：
RandomForest 代表传统机器学习，MLP 代表普通深度学习。
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from src.evaluate import (
    classification_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    save_metrics,
    save_feature_importance,
)
from src.preprocess import PreparedData, load_prepared_data


def train_random_forest(data: PreparedData, random_state: int = 42) -> tuple[object, dict]:
    """训练 RandomForest，并返回模型和评价结果。"""

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=None,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(data.x_train, data.y_train)

    scores = model.predict_proba(data.x_test)[:, 1]
    predictions = (scores >= 0.5).astype(int)
    metrics = classification_metrics("RandomForest", data.y_test, predictions, scores=scores)
    return model, {"metrics": metrics, "predictions": predictions, "scores": scores}


def train_mlp(data: PreparedData, random_state: int = 42) -> tuple[object, dict]:
    """训练 MLP 全连接神经网络，并返回模型和评价结果。"""

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=80,
        early_stopping=True,
        random_state=random_state,
    )

    # MLP 有时会在少量 epoch 后提示“还没完全收敛”，课程实验里这通常不影响流程演示。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data.x_train, data.y_train)

    scores = model.predict_proba(data.x_test)[:, 1]
    predictions = (scores >= 0.5).astype(int)
    metrics = classification_metrics("MLP", data.y_test, predictions, scores=scores)
    return model, {"metrics": metrics, "predictions": predictions, "scores": scores}


def run_baselines(
    data: PreparedData,
    results_dir: str | Path = "results",
    random_state: int = 42,
    save_plots: bool = True,
) -> list[dict]:
    """依次训练两个基线模型，并可选保存图表。"""

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random_forest_model, random_forest_result = train_random_forest(data, random_state=random_state)
    _, mlp_result = train_mlp(data, random_state=random_state)
    results = [random_forest_result, mlp_result]

    if save_plots:
        save_feature_importance(
            data.feature_names,
            random_forest_model.feature_importances_,
            output_dir,
        )
        plot_feature_importance(
            data.feature_names,
            random_forest_model.feature_importances_,
            output_dir,
        )
        for result in results:
            model_name = str(result["metrics"]["model"])
            plot_confusion_matrix(model_name, data.y_test, result["predictions"], output_dir)
            plot_roc_curve(model_name, data.y_test, result["scores"], output_dir)

    return [result["metrics"] for result in results]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="训练 RandomForest 和 MLP 基线模型")
    parser.add_argument("--data-dir", default="data", help="CSV 数据所在目录")
    parser.add_argument("--results-dir", default="results", help="实验结果输出目录")
    parser.add_argument("--max-rows", type=int, default=None, help="最多读取多少行真实数据")
    parser.add_argument("--demo-data", action="store_true", help="没有真实 CSV 时使用演示数据")
    parser.add_argument("--demo-samples", type=int, default=2400, help="演示数据样本数")
    parser.add_argument("--test-size", type=float, default=0.25, help="测试集比例")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--no-graph-features", action="store_true", help="关闭 IP/端口轻量图结构特征增强")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = parse_args()
    data = load_prepared_data(
        data_dir=args.data_dir,
        max_rows=args.max_rows,
        demo_data=args.demo_data,
        demo_samples=args.demo_samples,
        test_size=args.test_size,
        random_state=args.random_state,
        use_graph_features=not args.no_graph_features,
    )
    metrics = run_baselines(data, results_dir=args.results_dir, random_state=args.random_state)
    save_metrics(metrics, Path(args.results_dir) / "baseline_metrics.csv")
    print(f"基线模型训练完成，结果已保存到 {Path(args.results_dir).resolve()}")


if __name__ == "__main__":
    main()
