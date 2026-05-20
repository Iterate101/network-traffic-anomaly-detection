from pathlib import Path

from src.inspect_data import inspect_dataframe, write_inspection_report
from src.preprocess import build_demo_dataframe


def test_inspect_dataframe_reports_graph_features(tmp_path) -> None:
    """数据体检应该能识别标签、样本数量和轻量图结构特征。"""

    raw_data = build_demo_dataframe(n_samples=20, random_state=21)
    raw_data["Source IP"] = ["10.0.0.1"] * 10 + ["10.0.0.2"] * 10
    raw_data["Destination IP"] = ["192.168.1.10"] * 5 + ["192.168.1.20"] * 15
    raw_data["Source Port"] = list(range(5000, 5020))
    raw_data["Destination Port"] = [80, 443] * 10

    report = inspect_dataframe(
        raw_data=raw_data,
        csv_files=[Path("demo.csv")],
        data_source="csv",
        use_graph_features=True,
    )
    json_path, markdown_path = write_inspection_report(report, tmp_path)

    assert report["label_column"] == "Label"
    assert report["raw_rows"] == 20
    assert report["graph_feature_count"] > 0
    assert json_path.exists()
    assert markdown_path.exists()

