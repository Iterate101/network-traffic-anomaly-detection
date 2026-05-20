import numpy as np
import pytest

from src.evaluate import build_threshold_analysis, choose_threshold_for_f1, orient_anomaly_scores
from src.run_mask_ablation import parse_mask_ratios


def test_threshold_analysis_builds_multiple_strategies() -> None:
    """阈值分析应该生成多种可比较的策略。"""

    y_valid = np.array([0, 0, 0, 1, 1, 1])
    valid_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y_test = np.array([0, 0, 1, 1])
    test_scores = np.array([0.15, 0.25, 0.75, 0.85])

    rows = build_threshold_analysis(
        "Demo Autoencoder",
        y_valid,
        valid_scores,
        y_test,
        test_scores,
        target_recall=0.9,
    )

    strategies = {row["strategy"] for row in rows}
    assert "best_f1" in strategies
    assert "normal_95_percentile" in strategies
    assert len(rows) >= 4


def test_best_f1_threshold_returns_float() -> None:
    """最佳 F1 阈值应该返回一个数字。"""

    threshold = choose_threshold_for_f1(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
    )

    assert isinstance(threshold, float)


def test_orient_anomaly_scores_inverts_when_validation_auc_is_bad() -> None:
    """当低分更像攻击时，异常分数方向应该被校准。"""

    y_valid = np.array([0, 0, 1, 1])
    valid_scores = np.array([0.9, 0.8, 0.2, 0.1])
    test_scores = np.array([0.7, 0.3])

    oriented_valid, oriented_test, direction, raw_auc = orient_anomaly_scores(
        y_valid,
        valid_scores,
        test_scores,
    )

    assert direction == "inverted_reconstruction_error"
    assert raw_auc < 0.5
    assert np.allclose(oriented_valid, -valid_scores)
    assert np.allclose(oriented_test, -test_scores)


def test_parse_mask_ratios() -> None:
    """mask ratio 字符串应该被解析成浮点数列表。"""

    assert parse_mask_ratios("0, 0.15,0.3") == [0.0, 0.15, 0.3]

    with pytest.raises(ValueError):
        parse_mask_ratios("1.2")
