"""生成教学演示数据。

真实课程汇报建议使用 CIC-IDS2017；这个脚本只是为了让代码流程可以马上跑通。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocess import build_demo_dataframe


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="生成网络流量异常检测演示数据")
    parser.add_argument("--output", default="data/demo_traffic.csv", help="输出 CSV 路径")
    parser.add_argument("--samples", type=int, default=2400, help="样本数量")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = build_demo_dataframe(n_samples=args.samples, random_state=args.random_state)
    data.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"演示数据已生成：{output_path.resolve()}，共 {len(data)} 行。")


if __name__ == "__main__":
    main()

