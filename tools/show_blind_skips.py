#!/usr/bin/env python3
"""把「盲标抽到了、但没能计入评分」的帧画出来，供排查。

为什么要这个
------------
`tools/score_blind.py` 报的分母（例如 30 抽定 → 19 计入）会让人问一句
「另外 11 帧呢」。光看 CSV 答不上来 —— 得看图：

  · 是**没标**（人工侧判不了），还是**标了但系统不交付轴**（系统侧弃权）？
  · 没标的那帧，画面里到底有什么？目标是不是被画面边界切掉了？
  · 系统弃权的那帧，系统当时看到的位置在哪？

它同时也写给**下一批盲标**用：抽帧后先跑一遍，就能提前看出哪些帧注定计不了分
（标了也白标），从而在冻结名单前决定要不要把它们换掉。

用法（仓库根目录）：

    /opt/anaconda3/bin/python3 tools/show_blind_skips.py            # 全部素材
    /opt/anaconda3/bin/python3 tools/show_blind_skips.py --name video02

产出：`outputs/reports/figures/blind_skips.jpg`

判据说明
--------
脚本只报**几何上可判定**的事实（有没有检出、框是否越出画面、系统是否弃权），
不臆断「为什么人工标不了」—— 那一条由人写进
`outputs/annotations/blind/MANIFEST.md` 的「标注执行记录」。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from src import annotate as A  # noqa: E402
from src import blind as B  # noqa: E402
from src import config as C  # noqa: E402
from src import detect as D  # noqa: E402
from src import preprocess as P  # noqa: E402

TILE_W = 440          # 每格目标宽度（像素）
COLS = 4              # 每行格数
PAD = 8


def _detection_box(name: str, k: int, res, byk: dict) -> tuple[tuple[int, int, int, int] | None, bool]:
    """返回 (检测框, 是否被画面边界切掉)。无检出时框为 None。"""
    d = byk.get(k)
    if d is None:
        return None, False
    h, w = res.work_size[1], res.work_size[0]
    x1, y1 = d.cx - d.w / 2, d.cy - d.h / 2
    x2, y2 = d.cx + d.w / 2, d.cy + d.h / 2
    clipped = x1 < 0 or y1 < 0 or x2 > w or y2 > h
    return (int(x1), int(y1), int(x2), int(y2)), clipped


def _label_for(k: int, axis_deg: str | None, box, clipped: bool, why: str) -> str:
    """该帧状态的简短描述（只写几何事实）。"""
    if axis_deg:                                   # 有轴 ⇒ 本来就不该出现在这里
        return "有轴"
    bits = []
    if box is None:
        bits.append("无检出")
    else:
        bits.append("框被画面边界切掉" if clipped else "框在画面内")
    bits.append(why)
    return "；".join(bits)


def render(name: str, out_dir: Path, strip_buf: dict) -> list[tuple[int, str]]:
    """把该素材「抽到但没计入」的帧画成拼图，返回 [(k, 描述)]。"""
    spec = C.get(name)
    res = P.preprocess(spec)
    sampled = B._sampled_ks(name)
    if not sampled:
        return []
    lb = A.read_blind_labels(name)
    aks = sorted(A.read_labels(name))

    tr: dict[int, str] = {}
    tp = C.TRACK_DIR / f"{name}.csv"
    if tp.exists():
        with tp.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    tr[int(r["k"])] = (r.get("axis_deg") or "").strip()
                except (ValueError, TypeError):
                    continue

    dets, _ = D.run_detection(spec, res)
    byk = {d.k: d for d in dets}

    rows: list[tuple[int, str]] = []
    tiles = []
    for k in sampled:
        ax = tr.get(k, "")
        if ax:
            continue                               # 能计分，不属本图
        box, clipped = _detection_box(name, k, res, byk)
        if k not in lb:
            why = "人工未标"
        else:
            why = B._skip_reason(k, aks)
        rows.append((k, _label_for(k, ax or None, box, clipped, why)))

        fr = P.read_work_frame(res, spec.path, res.kept_frames[k]).copy()
        if box is not None:
            cv2.rectangle(fr, (box[0], box[1]), (box[2], box[3]),
                          (0, 0, 255) if not clipped else (255, 0, 255), 3)
            cv2.circle(fr, (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)),
                       8, (0, 255, 255), -1)
        h, w = fr.shape[:2]
        s = TILE_W / w
        fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        cv2.rectangle(fr, (0, 0), (fr.shape[1] - 1, 26),
                      (20, 20, 20), cv2.FILLED)
        cv2.putText(fr, f"{name} k={k}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (0, 255, 0), 2)
        cv2.putText(fr, ("切边" if clipped else "") + ("无检出" if box is None else ""),
                    (fr.shape[1] - 110, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 255), 2)
        tiles.append(fr)

    if not tiles:
        return rows

    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, PAD, cv2.BORDER_CONSTANT,
                                value=(25, 25, 25)) for t in tiles]
    nrow = (len(tiles) + COLS - 1) // COLS
    while len(tiles) < nrow * COLS:                # 补齐成矩形，便于 hstack
        tiles.append(np.full_like(tiles[0], 25) if tiles else None)
    strip = np.vstack([np.hstack(tiles[i * COLS:(i + 1) * COLS]) for i in range(nrow)])
    strip_buf[name] = strip

    # 逐素材单存一份，便于单独看
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"blind_skips_{name}.jpg"), strip)
    return rows


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out_dir = C.REPORT_DIR / "figures"
    names = []
    if "--name" in argv:
        names = [argv[argv.index("--name") + 1]]
    else:
        names = [s.name for s in C.annotation_videos() if B.has_blind(s.name)]

    total = 0
    strips: dict[str, np.ndarray] = {}
    for n in names:
        try:
            rows = render(n, out_dir, strips)
        except Exception as e:                      # noqa: BLE001
            print(f"  {n}: 渲染失败 {type(e).__name__}: {e}")
            continue
        if not rows:
            print(f"  {n}: 抽到的帧全部计入，无需排查")
            continue
        total += len(rows)
        print(f"  {n}: {len(rows)} 帧抽到但未计入")
        for k, why in rows:
            print(f"      k={k:>4}  {why}")

    if strips:
        w = max(s.shape[1] for s in strips.values())
        padded = [cv2.copyMakeBorder(s, 0, 0, 0, w - s.shape[1], cv2.BORDER_CONSTANT,
                                     value=(25, 25, 25)) for s in strips.values()]
        cv2.imwrite(str(out_dir / "blind_skips.jpg"), np.vstack(padded))
        print(f"\n合计 {total} 帧；拼图 {out_dir / 'blind_skips.jpg'}"
              f"（另有逐素材单独一份）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
