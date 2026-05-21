"""网络流量数据预处理工具。

这个文件负责把原始 CSV 数据变成模型可以直接训练的数字矩阵。
对初学者来说，可以把它理解成“把杂乱表格整理成机器学习输入”的步骤。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


LABEL_CANDIDATES = (
    "Label",
    "label",
    "Class",
    "class",
    "Attack",
    "attack",
    "target",
    "Target",
)

# 这些字段更像“身份信息”或“时间戳”，容易让模型死记硬背，不利于泛化。
NON_FEATURE_COLUMNS = {
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Src IP",
    "Dst IP",
    "Timestamp",
    "Time",
}

SOURCE_IP_COLUMNS = ("Source IP", "Src IP", "src_ip", "source_ip", "SourceIP")
DESTINATION_IP_COLUMNS = ("Destination IP", "Dst IP", "dst_ip", "destination_ip", "DestinationIP")
SOURCE_PORT_COLUMNS = ("Source Port", "Src Port", "src_port", "source_port", "SrcPort")
DESTINATION_PORT_COLUMNS = ("Destination Port", "Dst Port", "dst_port", "destination_port", "DstPort")


@dataclass
class PreparedData:
    """保存一次训练所需要的全部数据。"""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_train_raw: np.ndarray
    y_test_raw: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler


@dataclass
class GraphFeatureBuilder:
    """只用训练集拟合出来的轻量图结构特征构造器。

    这里的“拟合”不是训练神经网络，而是统计训练集中 IP、端口、边的出现次数。
    测试集只允许查询这些训练集统计量，不能反过来影响统计结果。
    """

    source_ip_column: str | None
    destination_ip_column: str | None
    source_port_column: str | None
    destination_port_column: str | None
    source_out_count: dict[str, int]
    destination_in_count: dict[str, int]
    edge_count: dict[str, int]
    source_unique_destination: dict[str, int]
    destination_unique_source: dict[str, int]
    destination_port_count: dict[str, int]
    source_port_pair_count: dict[str, int]
    source_port_count: dict[str, int]

    @classmethod
    def fit(cls, raw_data: pd.DataFrame) -> "GraphFeatureBuilder":
        """只根据训练集统计图结构信息。"""

        data = raw_data.copy()
        data.columns = [str(column).strip() for column in data.columns]
        source_ip_column = _first_existing_column(data.columns, SOURCE_IP_COLUMNS)
        destination_ip_column = _first_existing_column(data.columns, DESTINATION_IP_COLUMNS)
        source_port_column = _first_existing_column(data.columns, SOURCE_PORT_COLUMNS)
        destination_port_column = _first_existing_column(data.columns, DESTINATION_PORT_COLUMNS)

        source_out_count: dict[str, int] = {}
        destination_in_count: dict[str, int] = {}
        edge_count: dict[str, int] = {}
        source_unique_destination: dict[str, int] = {}
        destination_unique_source: dict[str, int] = {}
        destination_port_count: dict[str, int] = {}
        source_port_pair_count: dict[str, int] = {}
        source_port_count: dict[str, int] = {}

        if source_ip_column and destination_ip_column:
            source_ip = data[source_ip_column].astype(str).fillna("unknown")
            destination_ip = data[destination_ip_column].astype(str).fillna("unknown")
            edge = source_ip + "->" + destination_ip
            graph_frame = pd.DataFrame(
                {
                    "source_ip": source_ip,
                    "destination_ip": destination_ip,
                    "edge": edge,
                }
            )
            source_out_count = graph_frame["source_ip"].value_counts().to_dict()
            destination_in_count = graph_frame["destination_ip"].value_counts().to_dict()
            edge_count = graph_frame["edge"].value_counts().to_dict()
            source_unique_destination = graph_frame.groupby("source_ip")["destination_ip"].nunique().to_dict()
            destination_unique_source = graph_frame.groupby("destination_ip")["source_ip"].nunique().to_dict()

        if destination_port_column:
            destination_port = data[destination_port_column].astype(str).fillna("unknown")
            destination_port_count = destination_port.value_counts().to_dict()

        if source_ip_column and destination_port_column:
            source_ip = data[source_ip_column].astype(str).fillna("unknown")
            destination_port = data[destination_port_column].astype(str).fillna("unknown")
            source_port_pair = source_ip + "->" + destination_port
            source_port_pair_count = source_port_pair.value_counts().to_dict()

        if source_port_column:
            source_port = data[source_port_column].astype(str).fillna("unknown")
            source_port_count = source_port.value_counts().to_dict()

        return cls(
            source_ip_column=source_ip_column,
            destination_ip_column=destination_ip_column,
            source_port_column=source_port_column,
            destination_port_column=destination_port_column,
            source_out_count=source_out_count,
            destination_in_count=destination_in_count,
            edge_count=edge_count,
            source_unique_destination=source_unique_destination,
            destination_unique_source=destination_unique_source,
            destination_port_count=destination_port_count,
            source_port_pair_count=source_port_pair_count,
            source_port_count=source_port_count,
        )

    @property
    def feature_names(self) -> list[str]:
        """返回当前构造器能够生成的图特征名称。"""

        names: list[str] = []
        if self.source_ip_column and self.destination_ip_column:
            names.extend(
                [
                    "graph_src_out_count",
                    "graph_dst_in_count",
                    "graph_edge_count",
                    "graph_src_unique_dst",
                    "graph_dst_unique_src",
                ]
            )
        if self.destination_port_column:
            names.append("graph_dst_port_count")
        if self.source_ip_column and self.destination_port_column:
            names.append("graph_src_dst_port_count")
        if self.source_port_column:
            names.append("graph_src_port_count")
        return names

    def transform(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """把任意数据集转换成训练集统计口径下的图特征。"""

        data = raw_data.copy()
        data.columns = [str(column).strip() for column in data.columns]
        graph_features = pd.DataFrame(index=data.index)

        if self.source_ip_column and self.destination_ip_column:
            source_ip = data[self.source_ip_column].astype(str).fillna("unknown")
            destination_ip = data[self.destination_ip_column].astype(str).fillna("unknown")
            edge = source_ip + "->" + destination_ip

            # 这里用训练集统计表做映射；测试集中没见过的 IP/边记为 0，避免泄漏测试集信息。
            graph_features["graph_src_out_count"] = source_ip.map(self.source_out_count).fillna(0)
            graph_features["graph_dst_in_count"] = destination_ip.map(self.destination_in_count).fillna(0)
            graph_features["graph_edge_count"] = edge.map(self.edge_count).fillna(0)
            graph_features["graph_src_unique_dst"] = source_ip.map(self.source_unique_destination).fillna(0)
            graph_features["graph_dst_unique_src"] = destination_ip.map(self.destination_unique_source).fillna(0)

        if self.destination_port_column:
            destination_port = data[self.destination_port_column].astype(str).fillna("unknown")
            graph_features["graph_dst_port_count"] = destination_port.map(self.destination_port_count).fillna(0)

        if self.source_ip_column and self.destination_port_column:
            source_ip = data[self.source_ip_column].astype(str).fillna("unknown")
            destination_port = data[self.destination_port_column].astype(str).fillna("unknown")
            source_port_pair = source_ip + "->" + destination_port
            graph_features["graph_src_dst_port_count"] = source_port_pair.map(self.source_port_pair_count).fillna(0)

        if self.source_port_column:
            source_port = data[self.source_port_column].astype(str).fillna("unknown")
            graph_features["graph_src_port_count"] = source_port.map(self.source_port_count).fillna(0)

        if graph_features.empty:
            return graph_features

        graph_features = graph_features.astype(np.float32)
        graph_features = np.log1p(graph_features)
        return graph_features


def find_csv_files(data_dir: str | Path) -> list[Path]:
    """在数据目录中递归查找 CSV 文件。"""

    root = Path(data_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def _read_single_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """读取单个 CSV，并兼容少数非 UTF-8 编码的数据文件。"""

    try:
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, nrows=nrows, low_memory=False, encoding="latin1")


def load_traffic_csvs(
    data_dir: str | Path,
    max_rows: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """读取 data 目录下的所有 CSV，并合并成一个 DataFrame（表格数据）。"""

    csv_files = find_csv_files(data_dir)
    if not csv_files:
        raise FileNotFoundError(
            f"没有在 {Path(data_dir).resolve()} 找到 CSV 文件；可以先使用 --demo-data 跑演示数据。"
        )

    frames: list[pd.DataFrame] = []
    rows_per_file = None
    if max_rows is not None:
        rows_per_file = max(1, math.ceil(max_rows / len(csv_files)))

    for path in csv_files:
        frame = _read_single_csv(path, nrows=rows_per_file)
        frame["__source_csv"] = path.name
        frames.append(frame)

    if not frames:
        raise ValueError("CSV 文件存在，但没有读取到任何数据行。")

    data = pd.concat(frames, ignore_index=True)
    if max_rows is not None and len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=random_state)
    return data.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def build_demo_dataframe(n_samples: int = 2400, random_state: int = 42) -> pd.DataFrame:
    """生成一份小型演示数据，让项目在没有真实数据时也能跑通。

    这里不是伪装成真实论文结果，而是构造一个教学用数据集：
    正常流量更平稳，攻击流量在包速率、SYN 标志、字节速率等特征上更异常。
    """

    rng = np.random.default_rng(random_state)
    normal_count = int(n_samples * 0.65)
    attack_count = n_samples - normal_count

    normal = pd.DataFrame(
        {
            "Flow Duration": rng.normal(85000, 12000, normal_count).clip(1000),
            "Total Fwd Packets": rng.poisson(8, normal_count) + 1,
            "Total Backward Packets": rng.poisson(7, normal_count) + 1,
            "Fwd Packet Length Mean": rng.normal(420, 70, normal_count).clip(40),
            "Bwd Packet Length Mean": rng.normal(390, 65, normal_count).clip(40),
            "Flow Bytes/s": rng.normal(18000, 3000, normal_count).clip(1000),
            "Flow Packets/s": rng.normal(28, 6, normal_count).clip(1),
            "SYN Flag Count": rng.binomial(1, 0.08, normal_count),
            "ACK Flag Count": rng.binomial(1, 0.92, normal_count),
            "Down/Up Ratio": rng.normal(1.1, 0.2, normal_count).clip(0.1),
            "Label": "BENIGN",
        }
    )

    attack = pd.DataFrame(
        {
            "Flow Duration": rng.normal(26000, 9000, attack_count).clip(500),
            "Total Fwd Packets": rng.poisson(28, attack_count) + 1,
            "Total Backward Packets": rng.poisson(2, attack_count) + 1,
            "Fwd Packet Length Mean": rng.normal(960, 170, attack_count).clip(80),
            "Bwd Packet Length Mean": rng.normal(120, 50, attack_count).clip(20),
            "Flow Bytes/s": rng.normal(62000, 12000, attack_count).clip(3000),
            "Flow Packets/s": rng.normal(110, 24, attack_count).clip(5),
            "SYN Flag Count": rng.binomial(1, 0.76, attack_count),
            "ACK Flag Count": rng.binomial(1, 0.35, attack_count),
            "Down/Up Ratio": rng.normal(0.35, 0.12, attack_count).clip(0.02),
            "Label": "Attack",
        }
    )

    data = pd.concat([normal, attack], ignore_index=True)
    return data.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def detect_label_column(columns: Iterable[str]) -> str:
    """自动寻找标签列，兼容 CIC-IDS2017 和一些 Kaggle 清洗版字段名。"""

    normalized = {column.strip(): column for column in columns}
    for candidate in LABEL_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(f"没有找到标签列，候选名称包括：{', '.join(LABEL_CANDIDATES)}")


def _first_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """从多个候选字段名中找到数据里实际存在的字段。"""

    normalized = {str(column).strip(): str(column).strip() for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def add_graph_features(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """基于 IP 和端口生成轻量图结构特征。

    这里的“图”可以这样理解：每个 IP 是节点，每条流量是节点之间的一条边。
    我们不直接训练复杂 GNN，而是先把节点度数、边出现次数等图统计量作为新特征。
    """

    data = raw_data.copy()
    source_ip_column = _first_existing_column(data.columns, SOURCE_IP_COLUMNS)
    destination_ip_column = _first_existing_column(data.columns, DESTINATION_IP_COLUMNS)
    source_port_column = _first_existing_column(data.columns, SOURCE_PORT_COLUMNS)
    destination_port_column = _first_existing_column(data.columns, DESTINATION_PORT_COLUMNS)

    graph_features = pd.DataFrame(index=data.index)

    if source_ip_column and destination_ip_column:
        source_ip = data[source_ip_column].astype(str).fillna("unknown")
        destination_ip = data[destination_ip_column].astype(str).fillna("unknown")
        edge = source_ip + "->" + destination_ip
        graph_frame = pd.DataFrame(
            {
                "source_ip": source_ip,
                "destination_ip": destination_ip,
                "edge": edge,
            }
        )

        # 出度近似表示某个源主机发起了多少连接，攻击主机常常会异常活跃。
        source_out_count = graph_frame["source_ip"].value_counts()
        graph_features["graph_src_out_count"] = source_ip.map(source_out_count)

        # 入度近似表示某个目的主机被多少连接访问，受攻击目标可能会有异常入流量。
        destination_in_count = graph_frame["destination_ip"].value_counts()
        graph_features["graph_dst_in_count"] = destination_ip.map(destination_in_count)

        # 边出现次数表示同一对主机之间通信是否过于频繁。
        edge_count = graph_frame["edge"].value_counts()
        graph_features["graph_edge_count"] = edge.map(edge_count)

        source_unique_destination = graph_frame.groupby("source_ip")["destination_ip"].nunique()
        graph_features["graph_src_unique_dst"] = source_ip.map(source_unique_destination)

        destination_unique_source = graph_frame.groupby("destination_ip")["source_ip"].nunique()
        graph_features["graph_dst_unique_src"] = destination_ip.map(destination_unique_source)

    if destination_port_column:
        destination_port = data[destination_port_column].astype(str).fillna("unknown")
        destination_port_count = destination_port.value_counts()
        graph_features["graph_dst_port_count"] = destination_port.map(destination_port_count)

    if source_ip_column and destination_port_column:
        source_ip = data[source_ip_column].astype(str).fillna("unknown")
        destination_port = data[destination_port_column].astype(str).fillna("unknown")
        source_port_pair = source_ip + "->" + destination_port
        source_port_pair_count = source_port_pair.value_counts()
        graph_features["graph_src_dst_port_count"] = source_port_pair.map(source_port_pair_count)

    if source_port_column:
        source_port = data[source_port_column].astype(str).fillna("unknown")
        source_port_count = source_port.value_counts()
        graph_features["graph_src_port_count"] = source_port.map(source_port_count)

    if graph_features.empty:
        return data, []

    graph_features = graph_features.fillna(0).astype(np.float32)
    graph_features = np.log1p(graph_features)
    feature_names = list(graph_features.columns)
    data = pd.concat([data, graph_features], axis=1)
    return data, feature_names


def prepare_features_and_labels(
    raw_data: pd.DataFrame,
    use_graph_features: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """清洗原始表格，并返回特征矩阵 X 和二分类标签 y。"""

    data = raw_data.copy()
    data.columns = [str(column).strip() for column in data.columns]

    label_column = detect_label_column(data.columns)

    # 先删除重复行，避免同一条记录在训练和测试中同时出现，导致评价过于乐观。
    data = data.drop_duplicates().reset_index(drop=True)

    labels = data[label_column].astype(str).str.strip()
    y = labels.map(lambda value: 0 if value.upper() in {"BENIGN", "NORMAL"} else 1).to_numpy()

    if use_graph_features:
        data, _ = add_graph_features(data)

    removable_columns = [column for column in NON_FEATURE_COLUMNS if column in data.columns]
    features = data.drop(columns=[label_column, *removable_columns], errors="ignore")

    # 机器学习模型只能处理数字，所以这里把非数字内容转成 NaN，再统一清理。
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=1, how="all")

    medians = features.median(numeric_only=True).fillna(0)
    features = features.fillna(medians)

    if features.empty:
        raise ValueError("清洗后没有可用的数值特征，请检查 CSV 字段。")

    return features.astype(np.float32), y.astype(np.int64)


def split_and_scale(
    features: pd.DataFrame,
    labels: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42,
) -> PreparedData:
    """划分训练集和测试集，并对特征做标准化。"""

    stratify = labels if len(np.unique(labels)) == 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features.to_numpy(dtype=np.float32),
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    return PreparedData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        y_train_raw=np.where(y_train == 0, "BENIGN", "Attack"),
        y_test_raw=np.where(y_test == 0, "BENIGN", "Attack"),
        feature_names=list(features.columns),
        scaler=scaler,
    )


def _numeric_feature_frame(data: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """把原始表格中的普通数值特征整理出来。"""

    removable_columns = [column for column in NON_FEATURE_COLUMNS if column in data.columns]
    features = data.drop(columns=[label_column, *removable_columns], errors="ignore")
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def build_prepared_data_from_dataframe(
    raw_data: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    use_graph_features: bool = True,
) -> PreparedData:
    """从原始 DataFrame 构造训练数据，并避免测试集信息泄漏。

    和旧的“先生成全部特征再划分”不同，这个函数先划分训练/测试，
    再只用训练集拟合缺失值填充值、图结构统计量和标准化器。
    """

    data = raw_data.copy()
    data.columns = [str(column).strip() for column in data.columns]
    label_column = detect_label_column(data.columns)

    data = data.drop_duplicates().reset_index(drop=True)
    labels = data[label_column].astype(str).str.strip()
    y = labels.map(lambda value: 0 if value.upper() in {"BENIGN", "NORMAL"} else 1).to_numpy(dtype=np.int64)
    raw_labels = labels.to_numpy(dtype=str)

    stratify = y if len(np.unique(y)) == 2 else None
    indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    train_raw = data.iloc[train_indices].reset_index(drop=True)
    test_raw = data.iloc[test_indices].reset_index(drop=True)
    y_train = y[train_indices]
    y_test = y[test_indices]
    y_train_raw = raw_labels[train_indices]
    y_test_raw = raw_labels[test_indices]

    train_features = _numeric_feature_frame(train_raw, label_column)
    test_features = _numeric_feature_frame(test_raw, label_column)

    # 只根据训练集决定保留哪些普通数值列，避免测试集字段状态影响训练流程。
    usable_columns = [column for column in train_features.columns if not train_features[column].isna().all()]
    train_features = train_features[usable_columns]
    test_features = test_features.reindex(columns=usable_columns)

    medians = train_features.median(numeric_only=True).fillna(0)
    train_features = train_features.fillna(medians).fillna(0)
    test_features = test_features.fillna(medians).fillna(0)

    if use_graph_features:
        graph_builder = GraphFeatureBuilder.fit(train_raw)
        train_graph_features = graph_builder.transform(train_raw)
        test_graph_features = graph_builder.transform(test_raw)
        if not train_graph_features.empty:
            train_features = pd.concat([train_features.reset_index(drop=True), train_graph_features.reset_index(drop=True)], axis=1)
            test_features = pd.concat([test_features.reset_index(drop=True), test_graph_features.reset_index(drop=True)], axis=1)

    if train_features.empty:
        raise ValueError("清洗后没有可用的数值特征，请检查 CSV 字段。")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_features.to_numpy(dtype=np.float32)).astype(np.float32)
    x_test = scaler.transform(test_features.to_numpy(dtype=np.float32)).astype(np.float32)

    return PreparedData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        y_train_raw=y_train_raw,
        y_test_raw=y_test_raw,
        feature_names=list(train_features.columns),
        scaler=scaler,
    )


def load_or_build_dataframe(
    data_dir: str | Path,
    max_rows: int | None = None,
    demo_data: bool = False,
    demo_samples: int = 2400,
    random_state: int = 42,
) -> pd.DataFrame:
    """优先读取真实 CSV；如果没有数据且允许演示，就生成教学数据。"""

    if find_csv_files(data_dir):
        return load_traffic_csvs(data_dir, max_rows=max_rows, random_state=random_state)
    if demo_data:
        return build_demo_dataframe(n_samples=demo_samples, random_state=random_state)
    raise FileNotFoundError(
        f"没有在 {Path(data_dir).resolve()} 找到 CSV 文件；如果想先跑通流程，请加上 --demo-data。"
    )


def load_prepared_data(
    data_dir: str | Path = "data",
    max_rows: int | None = None,
    demo_data: bool = False,
    demo_samples: int = 2400,
    test_size: float = 0.25,
    random_state: int = 42,
    use_graph_features: bool = True,
) -> PreparedData:
    """完成“读取、清洗、划分、标准化”的完整预处理流程。"""

    raw_data = load_or_build_dataframe(
        data_dir=data_dir,
        max_rows=max_rows,
        demo_data=demo_data,
        demo_samples=demo_samples,
        random_state=random_state,
    )
    return build_prepared_data_from_dataframe(
        raw_data,
        test_size=test_size,
        random_state=random_state,
        use_graph_features=use_graph_features,
    )


def label_summary(labels: np.ndarray) -> dict[str, int]:
    """统计正常和攻击样本数量，方便在汇报中展示数据分布。"""

    return {
        "BENIGN": int(np.sum(labels == 0)),
        "Attack": int(np.sum(labels == 1)),
    }
