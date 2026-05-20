"""训练自监督 Autoencoder 系列模型。

Autoencoder（自编码器）的任务不是直接分类，而是学习“如何还原正常流量”。
如果一条流量很难被还原，说明它不像正常流量，就会得到较高的异常分数。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import (
    build_threshold_analysis,
    choose_threshold_for_recall,
    classification_metrics,
    orient_anomaly_scores,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_score_histogram,
    plot_threshold_analysis,
    save_metrics,
    save_threshold_analysis,
)
from src.preprocess import PreparedData, load_prepared_data


class TabularTransformerAutoencoder(nn.Module):
    """把一条表格流量记录当成“特征序列”来建模。"""

    def __init__(
        self,
        feature_count: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # 每个数值特征先变成一个 d_model 维向量，类似 NLP 中把词变成词向量。
        self.value_projection = nn.Linear(1, d_model)

        # 特征位置嵌入让模型知道“第几个特征代表什么含义”。
        self.feature_embedding = nn.Parameter(torch.randn(feature_count, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 解码器把每个特征 token 还原成原始数值。
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入形状是 [批大小, 特征数]，输出也是同样形状。"""

        tokens = self.value_projection(x.unsqueeze(-1))
        tokens = tokens + self.feature_embedding.unsqueeze(0)
        encoded = self.encoder(tokens)
        reconstructed = self.decoder(encoded).squeeze(-1)
        return reconstructed


class FeedForwardAutoencoder(nn.Module):
    """普通全连接 Autoencoder，用来和 Transformer 版本做消融对比。"""

    def __init__(self, feature_count: int) -> None:
        super().__init__()

        hidden_dim = max(16, min(128, feature_count * 4))
        bottleneck_dim = max(4, min(32, feature_count))

        # 编码器把原始特征压缩成更短的隐向量，逼迫模型学习正常流量的主要规律。
        self.encoder = nn.Sequential(
            nn.Linear(feature_count, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )

        # 解码器再从隐向量还原原始特征，用还原误差作为异常分数。
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入和输出都是 [批大小, 特征数]。"""

        encoded = self.encoder(x)
        return self.decoder(encoded)


def _make_loader(x: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """把 numpy 数组包装成 PyTorch 可以迭代的小批量数据。"""

    tensor = torch.tensor(x, dtype=torch.float32)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle)


def _mask_features(batch: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """随机遮住一部分特征，形成自监督的掩码重构任务。"""

    if mask_ratio <= 0:
        return batch

    # 标准化后的 0 接近“平均水平”，所以用 0 替换被遮住的特征比较自然。
    corrupted = batch.clone()
    mask = torch.rand_like(corrupted) < mask_ratio
    corrupted[mask] = 0.0
    return corrupted


def train_autoencoder_model(
    x_train_normal: np.ndarray,
    feature_count: int,
    model_type: str = "transformer",
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    mask_ratio: float = 0.0,
    d_model: int = 32,
    nhead: int = 4,
    num_layers: int = 2,
    random_state: int = 42,
    device: str | None = None,
) -> nn.Module:
    """只用正常样本训练 Autoencoder。

    model_type 为 ``vanilla`` 时训练普通全连接自编码器；
    model_type 为 ``transformer`` 时训练带注意力机制的 Transformer 自编码器。
    """

    torch.manual_seed(random_state)
    current_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if model_type == "vanilla":
        model = FeedForwardAutoencoder(feature_count=feature_count).to(current_device)
    elif model_type == "transformer":
        model = TabularTransformerAutoencoder(
            feature_count=feature_count,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        ).to(current_device)
    else:
        raise ValueError("model_type 只能是 'vanilla' 或 'transformer'。")

    loader = _make_loader(x_train_normal, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(current_device)
            optimizer.zero_grad()

            # 这里的 target 仍然是原始 batch；如果 mask_ratio > 0，
            # 模型看到的是被遮住的输入，任务是把完整正常流量还原出来。
            model_input = _mask_features(batch, mask_ratio)
            reconstructed = model(model_input)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch)

        mean_loss = epoch_loss / max(len(x_train_normal), 1)
        print(f"第 {epoch:02d}/{epochs:02d} 轮，自编码器训练损失：{mean_loss:.6f}")

    return model


@torch.no_grad()
def reconstruction_scores(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """计算每条样本的重构误差，误差越大越像异常。"""

    current_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    model.to(current_device)

    scores: list[np.ndarray] = []
    loader = _make_loader(x, batch_size=batch_size, shuffle=False)
    for (batch,) in loader:
        batch = batch.to(current_device)
        reconstructed = model(batch)
        batch_scores = torch.mean((reconstructed - batch) ** 2, dim=1)
        scores.append(batch_scores.cpu().numpy())

    return np.concatenate(scores)


def _run_autoencoder_experiment(
    data: PreparedData,
    model_name: str,
    model_type: str,
    results_dir: str | Path = "results",
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    mask_ratio: float = 0.0,
    target_recall: float = 0.9,
    random_state: int = 42,
    save_plots: bool = True,
) -> dict:
    """训练并评价一个自监督 Autoencoder 实验。"""

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stratify = data.y_train if len(np.unique(data.y_train)) == 2 else None
    x_fit, x_valid, y_fit, y_valid = train_test_split(
        data.x_train,
        data.y_train,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )

    x_fit_normal = x_fit[y_fit == 0]
    if len(x_fit_normal) == 0:
        raise ValueError("训练集中没有正常样本，无法训练自监督异常检测模型。")

    model = train_autoencoder_model(
        x_train_normal=x_fit_normal,
        feature_count=data.x_train.shape[1],
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mask_ratio=mask_ratio,
        random_state=random_state,
    )

    valid_scores = reconstruction_scores(model, x_valid, batch_size=batch_size)
    test_scores = reconstruction_scores(model, data.x_test, batch_size=batch_size)
    valid_scores, test_scores, score_direction, raw_validation_roc_auc = orient_anomaly_scores(
        y_valid,
        valid_scores,
        test_scores,
    )
    threshold = choose_threshold_for_recall(y_valid, valid_scores, target_recall=target_recall)

    predictions = (test_scores >= threshold).astype(int)
    metrics = classification_metrics(
        model_name,
        data.y_test,
        predictions,
        scores=test_scores,
        threshold=threshold,
    )
    metrics["mask_ratio"] = float(mask_ratio)
    metrics["score_direction"] = score_direction
    metrics["raw_validation_roc_auc"] = float(raw_validation_roc_auc)

    if save_plots:
        threshold_rows = build_threshold_analysis(
            model_name,
            y_valid,
            valid_scores,
            data.y_test,
            test_scores,
            target_recall=target_recall,
        )
        save_threshold_analysis(model_name, threshold_rows, output_dir)
        plot_threshold_analysis(model_name, threshold_rows, output_dir)
        plot_confusion_matrix(model_name, data.y_test, predictions, output_dir)
        plot_roc_curve(model_name, data.y_test, test_scores, output_dir)
        plot_precision_recall_curve(model_name, data.y_test, test_scores, output_dir)
        plot_score_histogram(
            model_name,
            data.y_test,
            test_scores,
            threshold,
            output_dir,
        )

    return metrics


def run_vanilla_autoencoder(
    data: PreparedData,
    results_dir: str | Path = "results",
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    target_recall: float = 0.9,
    random_state: int = 42,
    save_plots: bool = True,
) -> dict:
    """训练普通 Autoencoder，作为没有 Transformer 的消融对照组。"""

    return _run_autoencoder_experiment(
        data=data,
        model_name="Vanilla Autoencoder",
        model_type="vanilla",
        results_dir=results_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mask_ratio=0.0,
        target_recall=target_recall,
        random_state=random_state,
        save_plots=save_plots,
    )


def run_autoencoder(
    data: PreparedData,
    results_dir: str | Path = "results",
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    mask_ratio: float = 0.15,
    target_recall: float = 0.9,
    random_state: int = 42,
    save_plots: bool = True,
) -> dict:
    """训练 Masked Transformer Autoencoder。"""

    model_name = "Masked Transformer Autoencoder" if mask_ratio > 0 else "Transformer Autoencoder"
    return _run_autoencoder_experiment(
        data=data,
        model_name=model_name,
        model_type="transformer",
        results_dir=results_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mask_ratio=mask_ratio,
        target_recall=target_recall,
        random_state=random_state,
        save_plots=save_plots,
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="训练自监督 Transformer Autoencoder")
    parser.add_argument("--data-dir", default="data", help="CSV 数据所在目录")
    parser.add_argument("--results-dir", default="results", help="实验结果输出目录")
    parser.add_argument("--max-rows", type=int, default=None, help="最多读取多少行真实数据")
    parser.add_argument("--demo-data", action="store_true", help="没有真实 CSV 时使用演示数据")
    parser.add_argument("--demo-samples", type=int, default=2400, help="演示数据样本数")
    parser.add_argument("--test-size", type=float, default=0.25, help="测试集比例")
    parser.add_argument("--epochs", type=int, default=10, help="自编码器训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--mask-ratio", type=float, default=0.15, help="Transformer 训练时随机遮住的特征比例")
    parser.add_argument("--target-recall", type=float, default=0.9, help="阈值选择时的目标召回率")
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
    metrics = run_autoencoder(
        data,
        results_dir=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        mask_ratio=args.mask_ratio,
        target_recall=args.target_recall,
        random_state=args.random_state,
    )
    save_metrics([metrics], Path(args.results_dir) / "autoencoder_metrics.csv")
    print(f"自编码器训练完成，结果已保存到 {Path(args.results_dir).resolve()}")


if __name__ == "__main__":
    main()
