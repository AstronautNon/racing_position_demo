"""诊断：把掩膜、PCA 轴、人工轴画在一起，肉眼看偏差从哪来。

用法：
    /opt/anaconda3/bin/python3 tools/probe_shadow_view.py video09 1 6 15 28
输出：outputs/cache/shadow_view_<名>.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src import kinematics as K  # noqa: E402
from src import preprocess as P  # noqa: E402
from probe_shadow import collect, pca_axis  # noqa: E402

MOD = C.BODY_AXIS_MOD


def line(img, ax, ay, ang, length, color, thick=1):
    r = np.radians(ang)
    dx, dy = np.cos(r) * length / 2, np.sin(r) * length / 2
    p1 = (int(round(ax - dx)), int(round(ay - dy)))
    p2 = (int(round(ax + dx)), int(round(ay + dy)))
    cv2.line(img, p1, p2, color, thick, cv2.LINE_AA)
    return p1, p2


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "video09"
    ks = [int(v) for v in sys.argv[2:]] or [1, 6, 15, 28]
    spec = C.get(name)
    res = P.preprocess(spec)
    rows = {k: (gx, gy, dv, bbox) for k, gx, gy, dv, bbox in collect(spec, res)}
    manual = K.load_annotation_detail(name, res.n)

    cap = cv2.VideoCapture(str(spec.path))
    frames = {}
    for k, fr in enumerate(res.kept_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if ok:
            frames[k] = img
    cap.release()

    tiles = []
    for k in ks:
        if k not in rows or k not in frames:
            continue
        gx, gy, dv, (bx, by, bw, bh) = rows[k]
        img = frames[k]
        # 工作图 → 原图坐标（必须走 preprocess.work_to_orig：它带 ROI 裁剪偏移，
        # 自己按比例换算会漏掉偏移，取到错误区域）
        pad = 40
        x0 = max(0.0, bx - pad)
        y0 = max(0.0, by - pad)
        x1 = min(float(res.work_size[0]), bx + bw + pad)
        y1 = min(float(res.work_size[1]), by + bh + pad)
        scale = 3.0
        o0 = P.work_to_orig(res, x0, y0)
        o1 = P.work_to_orig(res, x1, y1)
        src = [int(round(o0[0])), int(round(o0[1])),
               int(round(o1[0])), int(round(o1[1]))]
        crop = img[src[1]:src[3], src[0]:src[2]].copy()
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST)

        vis = crop.copy()
        px = (gx - x0) * scale
        py = (gy - y0) * scale
        for i, (a, b) in enumerate(zip(px, py)):
            cv2.circle(vis, (int(round(a)), int(round(b))), 1,
                       (0, 255, 255), -1)

        cx, cy = px.mean(), py.mean()
        ax_auto = pca_axis(gx, gy)
        ax_man = manual["raw"][k] if manual else np.nan
        length = 1.15 * float(np.hypot(px.max() - px.min(), py.max() - py.min()))
        line(vis, cx, cy, ax_auto, length, (0, 0, 255), 3)
        if np.isfinite(ax_man):
            line(vis, cx, cy, ax_man, length * 0.8, (255, 0, 0), 3)
        # 连通域包围盒
        cv2.rectangle(vis, (int((bx - x0) * scale), int((by - y0) * scale)),
                      (int((bx + bw - x0) * scale), int((by + bh - y0) * scale)),
                      (0, 200, 0), 2)
        d = "" if not np.isfinite(ax_man) else f"  dA={((ax_auto - ax_man + 90) % MOD - 90):+.1f}"
        cv2.putText(vis, f"k={k}{d}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, f"k={k}{d}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(vis)

    if not tiles:
        print("没有可画的帧")
        return 1
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1],
                               cv2.BORDER_CONSTANT, value=(40, 40, 40))
             for t in tiles]
    cols = 2
    rowsn = int(np.ceil(len(tiles) / cols))
    while len(tiles) < cols * rowsn:
        tiles.append(np.full_like(tiles[0], 40))
    grid = np.vstack([np.hstack(tiles[i * cols:(i + 1) * cols]) for i in range(rowsn)])
    out = ROOT / "outputs/cache" / f"shadow_view_{name}.png"
    cv2.imwrite(str(out), grid)
    print(f"红=PCA 自动轴　蓝=人工轴　黄=掩膜像素　绿=检测框　→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
