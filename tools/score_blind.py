#!/usr/bin/env python3
"""盲标精度评分：系统交付的车身轴 vs 独立盲标。

用法（在仓库根目录执行）：

    /opt/anaconda3/bin/python3 tools/score_blind.py

读 `outputs/annotations/blind/labels_<素材>.csv` 与 `outputs/tracks/<素材>.csv`，
按 `axis_dist`（折 90° 的无向轴距离）算 MAE / 中位 / p90 / 最大，
并写出 `outputs/reports/D8_盲标精度.md`。

判据本体在 `src/blind.py`（与将来接进报告的实现共用一份），本脚本只是壳。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import blind as B  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    s = B.score(verbose=True)
    if s["per"]:
        print(f"\n报告： {s['report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
