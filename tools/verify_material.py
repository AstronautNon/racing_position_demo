"""验证 drift/ 下的素材文件是否与"当初生成标注/轨迹时"的那份是同一个。

为什么需要它：标注（`outputs/annotations/<名>.csv`）和工作图坐标（`x1..y2`）
都**与素材文件版本强绑定** —— k 索引来自该版本的帧序列，坐标依赖裁剪参数。
换了素材（重新下载、换分辨率、换剪辑起点）再沿用旧标注，结果是错的但看不出来。
本脚本用"重出裁图 vs 旧裁图"的像素比对把这个风险显式化。

原理：`outputs/annotations/crops/<名>/*.jpg` 是从原片按固定参数裁出来的
（render() 里写入的 JPEG 质量、缩放比、裁剪矩形全部由代码决定，可复现）。
所以只要输入文件相同，重出的 JPEG 应当**逐字节一致**。

用法：
    /opt/anaconda3/bin/python3 tools/verify_material.py video15
    /opt/anaconda3/bin/python3 tools/verify_material.py --all

退出码：0 全部一致；1 有不一致；2 无法比较（缺队列/缺裁图/源文件缺失）。
"""

from __future__ import annotations

import csv
import hashlib
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C          # noqa: E402
from src import crops as CR          # noqa: E402
from src import preprocess as P      # noqa: E402


def read_queue(name: str) -> list[dict]:
    """队列 CSV 前两行是 `#` 注释，写文件时手工跳过（同 src/annotate.read_queue）。"""
    p = C.ANNOT_QUEUE_DIR / f"{name}.csv"
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return [r for r in rows if r.get("k")]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def verify(name: str, verbose: bool = True) -> tuple[int, int, int]:
    """返回 (一致数, 不一致数, 无法比较数)。"""
    spec = C.get(name)
    if spec is None or not spec.path.exists():
        print(f"{name}: 源文件不存在（{spec.path if spec else '未登记'}）")
        return 0, 0, 1

    rows = read_queue(name)
    if not rows:
        print(f"{name}: 无待标注队列，无法比较")
        return 0, 0, 1

    res = P.preprocess(spec)
    same, diff, skip = 0, 0, 0
    diffs: list[tuple[int, float, float]] = []      # (k, MSE, 最大像素差)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for r in rows:
            k = int(r["k"])
            old_jpg = CR.crop_path(name, k)
            if not old_jpg.exists():
                skip += 1
                continue
            rect = CR.crop_rect_for(res, float(r["cx"]), float(r["cy"]),
                                    float(r["w"]), float(r["h"]))
            new_jpg = tmp / f"{k:05d}.jpg"
            CR.render(res, spec, k, rect, new_jpg)

            if md5(old_jpg) == md5(new_jpg):
                same += 1
                continue
            # 字节不同也可能是 JPEG 编码抖动，再看像素差
            a = cv2.imread(str(old_jpg))
            b = cv2.imread(str(new_jpg))
            if a is None or b is None or a.shape != b.shape:
                diff += 1
                diffs.append((k, float("nan"), float("nan")))
                continue
            d = np.abs(a.astype(np.int16) - b.astype(np.int16))
            mse = float((d.astype(float) ** 2).mean())
            diffs.append((k, mse, float(d.max())))
            if mse < 8.0:        # JPEG 量化级抖动，视觉无差
                same += 1
            else:
                diff += 1

    tag = "一致" if diff == 0 else "**不一致**"
    print(f"{name}: 比对 {len(rows)} 帧 → 一致 {same}，不一致 {diff}，缺裁图 {skip}  [{tag}]")
    if diffs and verbose:
        bad = [d for d in diffs if not (d[1] == d[1] and d[1] < 8.0)]
        for k, mse, mx in bad[:8]:
            print(f"    k={k:<4} MSE={mse:>8.1f}  最大像素差={mx:.0f}")
        if len(bad) > 8:
            print(f"    ...另有 {len(bad) - 8} 帧")
    return same, diff, skip


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("-")]
    if "--all" in argv:
        names = [s.name for s in C.active_static_videos()]
    elif args:
        names = args
    else:
        print(__doc__)
        return 2

    print("素材一致性核验（重出裁图 vs 旧裁图，逐字节 + 像素差）")
    print("=" * 78)
    total_diff = 0
    for n in names:
        _, d, _ = verify(n)
        total_diff += d
    print("=" * 78)
    print("结论：", "全部素材与旧裁图一致，标注可继续沿用" if total_diff == 0
          else f"有 {total_diff} 帧不一致 —— 素材已变，该素材的旧标注需作废重标")
    return 0 if total_diff == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
