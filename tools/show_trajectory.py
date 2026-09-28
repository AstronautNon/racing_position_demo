"""把某素材的检测轨迹画在一张工作图上，一眼看清形态与被剔除的帧。

为什么需要它
------------
位置问题有两类，肉眼看曲线很难区分：
- **孤立假跳点**（掩膜瞬时并进阴影/杂物）→ 轨迹上是一个尖刺；
- **平滑的系统性偏移**（车后水雾尾迹并进掩膜、质心被拖走）→ 轨迹本身光滑，
  但整段偏离车身，**任何"位置跳变"式判据都发现不了**（README 约定 13）。

静态相机把轨迹叠在一张底图上，这两类的形状差异一目了然：
尖刺 vs 顺滑的大弧。配合逐帧位移表，能判断某段是"真实快动作"还是"跟丢了"。

用法：
    python3 tools/show_trajectory.py video15            # k=0..25
    python3 tools/show_trajectory.py video12 80         # k=0..79
输出：outputs/reports/figures/<素材名>_trajectory.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src import detect as D  # noqa: E402
from src import kinematics as K  # noqa: E402
from src import preprocess as P  # noqa: E402


def draw(name: str, kmax: int, cols_out: Path | None = None) -> Path:
    spec = C.get(name)
    res = P.preprocess(spec)
    dets, _ = D.run_detection(spec, res)
    if not dets:
        raise SystemExit(f"{name}: 无检出")
    kin = K.compute(dets, res)
    byk = {d.k: d for d in dets}
    rej = kin["rejected"]

    base = P.read_work_frame(res, spec.path, dets[0].frame).copy()
    base = cv2.addWeighted(base, 0.45, np.full_like(base, 255), 0.55, 0)

    ks = [k for k in range(min(kmax, res.n)) if k in byk]
    if len(ks) < 2:
        raise SystemExit(f"{name}: 该区间检出不足")
    # 轨迹连线：蓝 → 红表示时间顺序，跳过的缺口（无检出）不连
    for a, b in zip(ks[:-1], ks[1:]):
        if b - a > 3:
            continue
        f = (a - ks[0]) / max(1, ks[-1] - ks[0])
        col = (int(255 * (1 - f)), 60, int(255 * f))
        cv2.line(base, (int(byk[a].cx), int(byk[a].cy)),
                 (int(byk[b].cx), int(byk[b].cy)), col, 2, cv2.LINE_AA)

    for k in ks:
        d = byk[k]
        c = (int(d.cx), int(d.cy))
        if rej[k]:
            cv2.drawMarker(base, c, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 3)
            col = (0, 0, 255)
        else:
            cv2.circle(base, c, 6, (0, 190, 0), -1, cv2.LINE_AA)
            col = (0, 190, 0)
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(base, (int(x1), int(y1)), (int(x2), int(y2)), col, 1, cv2.LINE_AA)
        for thick, tc in ((3, (0, 0, 0)), (1, (255, 255, 255))):
            cv2.putText(base, str(k), (c[0] + 9, c[1] - 9), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, tc, thick, cv2.LINE_AA)

    hdr = (f"{name}  k={ks[0]}..{ks[-1]}   红叉=被判位置离群   绿点=通过   "
           f"连线按时间蓝→红；缺口>3帧不连")
    cv2.putText(base, hdr, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)

    out = cols_out or (C.ROOT / "outputs/reports/figures" / f"{name}_trajectory.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), base, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("写出", out)

    print()
    print(f"{'k':>4}{'step到前帧px':>13}{'面积':>8}{'w':>6}{'h':>6}{'离群':>6}")
    prev = None
    for k in ks:
        d = byk[k]
        step = float(np.hypot(d.cx - prev.cx, d.cy - prev.cy)) if prev else float("nan")
        print(f"{k:>4}{step:>13.1f}{d.area:>8.0f}{d.w:>6.0f}{d.h:>6.0f}"
              f"{'是' if rej[k] else '':>6}")
        prev = d
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    name = argv[0] if argv else "video15"
    kmax = int(argv[1]) if len(argv) > 1 else 26
    draw(name, kmax)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
