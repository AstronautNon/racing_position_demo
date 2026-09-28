#!/usr/bin/env python3
"""抽盲标帧（独立测试集）。

用法（在仓库根目录执行）：

    /opt/anaconda3/bin/python3 tools/build_blind.py            # 默认抽 30 帧
    /opt/anaconda3/bin/python3 tools/build_blind.py --n 20

抽出的帧写在 `outputs/annotations/blind/queue/<素材>.csv`，与普通队列同格式，
但**提示列与运动方向列都是空的** —— 盲标的意义就在于此。

抽完去标：

    /opt/anaconda3/bin/python3 -m src.annotate --serve --blind

标完评分：

    /opt/anaconda3/bin/python3 tools/score_blind.py

纪律：抽定后**不许换**。换采样等于偷看答案（见 src/blind.py 的模块说明）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import blind as B  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="抽取盲标帧（独立测试集）")
    ap.add_argument("--n", type=int, default=30, help="总抽帧数（默认 30，按素材帧数分配）")
    ap.add_argument("--seed", type=int, default=20260928, help="随机种子（当前定序采样，只作留痕）")
    args = ap.parse_args(argv)
    print(f"抽盲标帧：目标 {args.n} 帧（从各素材**标注队列之外**取）")
    B.build_all(n_total=args.n, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
