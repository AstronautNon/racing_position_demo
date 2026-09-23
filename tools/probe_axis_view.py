"""把「检测框 / 车身分割掩膜 / 三种轴」画在一张图上，用于肉眼找根因。

三种轴：
- 黄 = 月牙掩膜 PCA（现状，来自 BgSubDetector）
- 蓝 = 人工标注真值（若有）
- 红 = 通用轮廓轴（detect.contour_mask + PCA）

用法：
    /opt/anaconda3/bin/python3 tools/probe_axis_view.py video02 10 84 118
    /opt/anaconda3/bin/python3 tools/probe_axis_view.py video09 1 6 15

输出：outputs/cache/axis_view_<素材>.png（按约定不入库）
每帧一行：左=原图+检测框，右=工作图放大（绿=分割掩膜，黄/蓝/红=三种轴）
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import annotate as A          # noqa: E402
from src import config as C            # noqa: E402
from src import detect as D            # noqa: E402
from src import preprocess as P        # noqa: E402

MOD = C.BODY_AXIS_MOD
PANEL_H = 300


def sdiff(a: float, b: float) -> float:
    """带符号差，折到 ±90。"""
    return (a - b + 90.0) % MOD - 90.0


def draw_axis(img: np.ndarray, ang: float, cx: float, cy: float,
              length: float, color, thick: int = 2) -> None:
    r = np.radians(ang)
    dx, dy = np.cos(r), np.sin(r)
    p1 = (int(round(cx - dx * length)), int(round(cy - dy * length)))
    p2 = (int(round(cx + dx * length)), int(round(cy + dy * length)))
    cv2.line(img, p1, p2, color, thick, cv2.LINE_AA)


def put(img: np.ndarray, txt: str, y: int, color=(255, 255, 255)) -> None:
    cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
    cv2.putText(img, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def main(argv: list[str]) -> int:
    name = argv[1] if len(argv) > 1 else "video09"
    ks_arg = [int(a) for a in argv[2:]]

    spec = C.get(name)
    if spec is None or not spec.path.exists():
        print(f"素材 {name} 不可用")
        return 1
    res = P.preprocess(spec)
    dets = D.BgSubDetector(res, spec.path).run()
    by_k = {d.k: d for d in dets}
    truth = {k: v["axis_deg"] for k, v in A.read_labels(name).items()
             if "axis_deg" in v}

    if ks_arg:
        ks = ks_arg
    elif truth:
        # 优先挑偏差最大的几帧，最能看到病灶
        ks = sorted(truth)[:3]
    else:
        keys = sorted(by_k)
        ks = [keys[0], keys[len(keys) // 2], keys[-1]]

    want = set(ks)
    works: dict[int, np.ndarray] = {}
    for k, _, _, wf in D.iter_work_frames(res, spec.path):
        if k in want:
            works[k] = wf
            if len(works) == len(want):
                break

    sx, sy = P.work_scale(res)
    l, t = res.crop[0], res.crop[1]

    rows = []
    for k in ks:
        d = by_k.get(k)
        wf = works.get(k)
        if d is None or wf is None:
            print(f"k={k} 缺检测或工作图，跳过")
            continue

        # ---- 左：原图 + 检测框 ----
        cap = cv2.VideoCapture(str(spec.path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, res.kept_frames[k])
        ok, fr = cap.read()
        cap.release()
        if not ok:
            continue
        frame = fr.copy()
        bx0 = int(round((d.cx - d.w / 2) * sx)) + l
        by0 = int(round((d.cy - d.h / 2) * sy)) + t
        bx1 = int(round((d.cx + d.w / 2) * sx)) + l
        by1 = int(round((d.cy + d.h / 2) * sy)) + t
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), (0, 200, 255), 3)
        put(frame, f"k={k}  box {int(d.w)}x{int(d.h)}", 42)
        sc = PANEL_H / frame.shape[0]
        left = cv2.resize(frame, (int(frame.shape[1] * sc), PANEL_H))

        # ---- 右：工作图放大 + 掩膜 + 三种轴 ----
        pad = 0.55
        wx0 = int(max(0, d.cx - d.w * (0.5 + pad)))
        wy0 = int(max(0, d.cy - d.h * (0.5 + pad)))
        wx1 = int(min(res.work_size[0], d.cx + d.w * (0.5 + pad)))
        wy1 = int(min(res.work_size[1], d.cy + d.h * (0.5 + pad)))
        if wx1 - wx0 < 8 or wy1 - wy0 < 8:
            continue
        patch = wf[wy0:wy1, wx0:wx1].copy()

        got = D.contour_mask(wf, d.cx, d.cy, d.w, d.h)
        axis_cv, q_fore = (None, None)
        if got is not None:
            mask, q_fore = got
            m = mask[wy0:wy1, wx0:wx1]
            if m.size and m.any():
                ov = patch.copy()
                ov[m > 0] = (90, 235, 90)
                patch = cv2.addWeighted(patch, 0.55, ov, 0.45, 0)
            # 轮廓轴
            ys, xs = np.nonzero(mask)
            if len(xs) >= 30:
                pts = np.stack([xs, ys], 1).astype(np.float32)
                if len(pts) > 2000:
                    pts = pts[:: len(pts) // 2000 + 1]
                _, ev = cv2.PCACompute(pts, mean=None)
                axis_cv = float(np.degrees(np.arctan2(ev[0][1], ev[0][0])) % MOD)

        # 放大到面板高
        sc2 = PANEL_H / patch.shape[0]
        patch = cv2.resize(patch, (max(1, int(patch.shape[1] * sc2)), PANEL_H),
                           interpolation=cv2.INTER_NEAREST)
        fx = patch.shape[1] / max(1e-9, (wx1 - wx0))
        fy = patch.shape[0] / max(1e-9, (wy1 - wy0))
        pcx, pcy = (d.cx - wx0) * fx, (d.cy - wy0) * fy
        d2 = max(d.w * fx, d.h * fy) * 1.2

        draw_axis(patch, d.axis_deg, pcx, pcy, d2, (0, 215, 255), 2)   # 黄：月牙
        if k in truth:
            draw_axis(patch, truth[k], pcx, pcy, d2, (255, 110, 40), 3)  # 蓝：真值
        if axis_cv is not None:
            draw_axis(patch, axis_cv, pcx, pcy, d2, (60, 30, 220), 2)    # 红：轮廓

        y = 20
        put(patch, f"k={k}", y, (255, 255, 255)); y += 22
        put(patch, f"moon(黄) {d.axis_deg:.1f}", y, (0, 215, 255)); y += 22
        if k in truth:
            dd = sdiff(d.axis_deg, truth[k])
            put(patch, f"truth(蓝) {truth[k]:.1f}  d_moon={dd:+.1f}", y, (255, 130, 60)); y += 22
        else:
            put(patch, "truth(蓝) -", y, (200, 200, 200)); y += 22
        if axis_cv is not None:
            dc = sdiff(axis_cv, truth[k]) if k in truth else None
            txt = f"contour(红) {axis_cv:.1f}"
            if dc is not None:
                txt += f"  d={dc:+.1f}"
            put(patch, txt, y, (90, 60, 240)); y += 22
            put(patch, f"fore_frac {q_fore:.3f}", y, (90, 235, 90))
        else:
            put(patch, "contour 分割失败", y, (90, 60, 240))

        canvas = np.full((PANEL_H, left.shape[1] + patch.shape[1] + 10, 3), 245, np.uint8)
        canvas[:, :left.shape[1]] = left
        canvas[:, left.shape[1] + 10:] = patch
        rows.append(canvas)

    if not rows:
        print("没有可画的帧")
        return 1
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=245)
            for r in rows]
    sheet = np.vstack(rows)
    bar = np.full((30, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, f"{name} | left: orig+box | right: work patch, green=mask, "
                     f"yellow=moon PCA, blue=manual truth, red=contour",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    sheet = np.vstack([bar, sheet])
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = C.CACHE_DIR / f"axis_view_{name}.png"
    cv2.imwrite(str(out), sheet)
    print(f"已生成 {out.relative_to(ROOT)}  {sheet.shape[1]}x{sheet.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
