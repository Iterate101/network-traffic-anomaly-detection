from argparse import Namespace

from src.run_research_pipeline import write_pipeline_manifest, write_submission_checklist


def test_pipeline_artifacts_are_written(tmp_path) -> None:
    """流水线应该能生成提交清单和总览 JSON。"""

    (tmp_path / "metrics.csv").write_text("model,accuracy\nDemo,1.0\n", encoding="utf-8")
    checklist_path = write_submission_checklist(tmp_path, include_mask_ablation=True)
    manifest_path = write_pipeline_manifest(
        Namespace(demo_data=True, max_rows=100),
        {
            "data_source": "demo",
            "raw_rows": 100,
            "feature_count_after_preprocess": 10,
            "graph_feature_count": 0,
        },
        [{"model": "Demo"}],
        [{"model": "Masked Transformer Autoencoder"}],
        tmp_path,
    )

    assert checklist_path.exists()
    assert manifest_path.exists()
    assert "项目提交清单" in checklist_path.read_text(encoding="utf-8")

