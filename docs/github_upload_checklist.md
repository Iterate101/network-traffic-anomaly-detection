# GitHub 上传清单

这份清单用于上传仓库前最后核对，目标是：**上传代码和文档，不上传数据集、实验结果和 PPT 文件**。

## 1. 应该上传的内容

- `src/`：项目源代码，包括预处理、训练、评价和研究流水线。
- `tests/`：自动化测试，证明关键预处理和实验函数可运行。
- `docs/`：论文方法对应、实验指南、汇报提纲和讲稿。
- `README.md`：项目主页说明。
- `requirements.txt`：Python 依赖列表。
- `.gitignore`：忽略大文件和本地产物。
- `data/.gitkeep`：保留数据目录占位。
- `results/.gitkeep`：保留结果目录占位。

## 2. 不应该上传的内容

- `data/*.csv`：真实 CIC-IDS2017 数据文件较大，不能直接上传。
- `data/raw_parquet/`：下载缓存，不属于源码。
- `results/` 中的 `.csv`、`.json`、`.png`、`.md`：实验输出可以由代码重新生成。
- `presentation_workspace/`：PPT 工作区和导出的 `output.pptx` 不上传。
- `.pytest_cache/`、`__pycache__/`：本地缓存，不上传。

## 3. 上传前建议运行

```powershell
python -m pytest tests
python -m compileall src tests
git status --short
```

检查 `git status --short` 时，理想情况下只应该看到源码、测试、文档和配置文件。不要出现真实 CSV、PPTX、结果图片或缓存目录。

## 4. 复现实验命令

别人克隆仓库后，需要自己把 CIC-IDS2017 或 Kaggle 清洗版 CSV 放入 `data/`，然后运行：

```powershell
python -m src.run_research_pipeline --data-dir data --results-dir results --max-rows 20000 --epochs 10 --ablation-epochs 6 --mask-ratios 0,0.15,0.3,0.45
```

如果只是想快速确认环境，可以先运行演示数据：

```powershell
python -m src.run_research_pipeline --demo-data --demo-samples 600 --epochs 2 --ablation-epochs 1
```

## 5. 当前课程汇报结论

- 本项目参考 TOP 论文中的自监督学习、Transformer Autoencoder 和图结构建模思想。
- 项目实现了普通 AE、Transformer AE、Masked Transformer AE 的消融对比。
- 当前真实数据结果显示：Transformer AE 在自监督模型中综合表现最好。
- Masked Transformer AE 的 Recall 更高，但 Precision 和 F1 下降，因此掩码重构需要通过实验验证。
- 验证集用于异常分数方向校准和阈值选择，汇报时应表述为“自监督表示学习 + 少量标签阈值校准”。
