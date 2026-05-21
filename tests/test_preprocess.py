import numpy as np
import pandas as pd

from src.preprocess import (
    GraphFeatureBuilder,
    add_graph_features,
    build_prepared_data_from_dataframe,
    build_demo_dataframe,
    prepare_features_and_labels,
    split_and_scale,
)
from src.artifacts import build_dataset_summary, write_experiment_report


def test_demo_data_can_be_prepared() -> None:
    """演示数据应该能被清洗成模型可训练的数字矩阵。"""

    raw_data = build_demo_dataframe(n_samples=120, random_state=7)
    features, labels = prepare_features_and_labels(raw_data)
    prepared = split_and_scale(features, labels, test_size=0.25, random_state=7)

    assert prepared.x_train.shape[1] == len(prepared.feature_names)
    assert prepared.x_test.shape[1] == len(prepared.feature_names)
    assert prepared.y_train_raw.shape[0] == prepared.y_train.shape[0]
    assert prepared.y_test_raw.shape[0] == prepared.y_test.shape[0]
    assert set(np.unique(prepared.y_train)).issubset({0, 1})
    assert not np.isnan(prepared.x_train).any()
    assert not np.isnan(prepared.x_test).any()


def test_dataframe_preparation_uses_train_only_statistics() -> None:
    """正式训练入口应该先划分数据，再拟合标准化和图统计。"""

    raw_data = build_demo_dataframe(n_samples=120, random_state=17)
    prepared = build_prepared_data_from_dataframe(raw_data, test_size=0.25, random_state=17)

    assert prepared.x_train.shape[0] == 90
    assert prepared.x_test.shape[0] == 30
    assert prepared.x_train.shape[1] == len(prepared.feature_names)
    assert not np.isnan(prepared.x_train).any()
    assert not np.isnan(prepared.x_test).any()


def test_graph_feature_builder_does_not_count_test_only_ip() -> None:
    """测试集中新出现的 IP 不应该反过来增加训练集图统计。"""

    train_data = pd.DataFrame(
        {
            "Source IP": ["10.0.0.1", "10.0.0.1"],
            "Destination IP": ["192.168.1.1", "192.168.1.2"],
            "Source Port": [5000, 5001],
            "Destination Port": [80, 443],
            "Label": ["BENIGN", "BENIGN"],
        }
    )
    test_data = pd.DataFrame(
        {
            "Source IP": ["10.0.0.9"],
            "Destination IP": ["192.168.1.9"],
            "Source Port": [5999],
            "Destination Port": [9999],
            "Label": ["Attack"],
        }
    )

    builder = GraphFeatureBuilder.fit(train_data)
    test_graph_features = builder.transform(test_data)

    assert test_graph_features.loc[0, "graph_src_out_count"] == 0.0
    assert test_graph_features.loc[0, "graph_dst_in_count"] == 0.0
    assert test_graph_features.loc[0, "graph_dst_port_count"] == 0.0


def test_graph_features_are_added_when_ip_columns_exist() -> None:
    """如果数据里有 IP 和端口字段，预处理应该生成轻量图结构特征。"""

    raw_data = build_demo_dataframe(n_samples=8, random_state=13)
    raw_data["Source IP"] = [
        "10.0.0.1",
        "10.0.0.1",
        "10.0.0.2",
        "10.0.0.3",
        "10.0.0.3",
        "10.0.0.3",
        "10.0.0.4",
        "10.0.0.5",
    ]
    raw_data["Destination IP"] = [
        "192.168.1.10",
        "192.168.1.10",
        "192.168.1.20",
        "192.168.1.20",
        "192.168.1.30",
        "192.168.1.30",
        "192.168.1.30",
        "192.168.1.40",
    ]
    raw_data["Source Port"] = [5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007]
    raw_data["Destination Port"] = [80, 80, 443, 443, 22, 22, 22, 8080]

    enhanced_data, graph_feature_names = add_graph_features(raw_data)
    features, _ = prepare_features_and_labels(enhanced_data, use_graph_features=False)

    assert graph_feature_names
    assert any(name.startswith("graph_") for name in features.columns)


def test_report_artifact_can_be_written(tmp_path) -> None:
    """实验报告应该能根据指标和数据概况自动生成。"""

    raw_data = build_demo_dataframe(n_samples=120, random_state=11)
    features, labels = prepare_features_and_labels(raw_data)
    prepared = split_and_scale(features, labels, test_size=0.25, random_state=11)
    summary = build_dataset_summary(prepared)
    metrics = [
        {
            "model": "RandomForest",
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.97,
            "f1": 0.95,
            "roc_auc": 0.98,
        }
    ]

    report_path = write_experiment_report(metrics, summary, tmp_path)

    assert report_path.exists()
    assert "RandomForest" in report_path.read_text(encoding="utf-8")
