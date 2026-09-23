"""验证：若掩膜是"运动月牙"导致主轴偏 7°，那么换成真正的车身轮廓应当贴合人工标注。

做法：video09 的车是深红色、地面是灰色柏油，用 HSV 饱和度切出车身（同时自动排除
阴影——阴影是低饱和的暗灰），取最大连通域做 PCA，与人工标注比。

用法：
    /opt/anaconda3/bin/python3 tools/probe_color_axis.py video09
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
from probe_shadow import collect, pca_axis  # noqa: E402

MOD = C.BODY_AXIS_MOD


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "video09"
    spec = C.get(name)
    res = P.preprocess(spec)
    rows = {k: (gx, gy) for k, gx, gy, _, _ in collect(spec, res)}

    frames = {}
    for k, _, _, wf in D.iter_work_frames(res, spec.path):
        if k in rows:
            frames[k] = wf

    manual = K.load_annotation_detail(name, res.n)
    hsv_lo = (0, 90, 40)
    print(f"[{name}] 颜色阈值 S>={hsv_lo[1]} V>={hsv_lo[2]}")
    print()
    hdr = f"{'k':>4}{'人工轴':>9}{'月牙PCA':>10}{'颜色PCA':>10}{'Δ月牙':>9}{'Δ颜色':>9}{'面积':>8}"
    print(hdr)
    print("-" * len(hdr))
    dc, dm, dh, keep = [], [], [], 0
    for k in sorted(rows):
        if not np.isfinite(manual["raw"][k]):
            continue
        gx, gy = rows[k]
        wf = frames.get(k)
        if wf is None:
            continue
        cxm, cym = gx.mean(), gy.mean()
        hsv = cv2.cvtColor(wf, cv2.COLOR_BGR2HSV)
        # 红色在 HSV 里跨 0/180 两端，两段并起来
        m = ((hsv[..., 0] <= 12) | (hsv[..., 0] >= 168)) \
            & (hsv[..., 1] >= hsv_lo[1]) & (hsv[..., 2] >= hsv_lo[2])
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
        if n <= 1:
            print(f"{k:>4}  颜色分割失败")
            continue
        # 取质心离检出质心最近的连通域（画面里可能有多块红色）
        best, bd = -1, 1e18
        for j in range(1, n):
            if stats[j, 4] < 300:
                continue
            d = float(np.hypot(cents[j][0] - cxm, cents[j][1] - cym))
            if d < bd:
                bd, best = d, j
        if best < 0:
            continue
        ys, xs = np.nonzero(lab == best)
        if len(xs) < 50:
            continue
        a_col = pca_axis(xs.astype(float), ys.astype(float))
        a_moon = pca_axis(gx.astype(float), gy.astype(float))
        # 车窗是暗色低饱和，会被颜色阈值漏掉 → 用凸包/填洞把车身补成整体再 PCA
        cmask = (lab == best).astype(np.uint8)
        cnts, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = np.zeros_like(cmask)
        if cnts:
            cv2.fillConvexPoly(hull, cv2.convexHull(max(cnts, key=cv2.contourArea)), 1)
        hy, hx = np.nonzero(hull)
        a_hull = pca_axis(hx.astype(float), hy.astype(float)) if len(hx) >= 50 else np.nan
        ref = manual["raw"][k]
        d1 = (a_moon - ref + 90) % MOD - 90
        d2 = (a_col - ref + 90) % MOD - 90
        d3 = (a_hull - ref + 90) % MOD - 90 if np.isfinite(a_hull) else np.nan
        dm.append(abs(d1))
        dc.append(abs(d2))
        if np.isfinite(d3):
            dh.append(abs(d3))
        keep += 1
        print(f"{k:>4}{ref:>9.1f}{a_moon:>10.1f}{a_col:>10.1f}"
              f"{a_hull:>10.1f}{d1:>+8.1f}{d2:>+8.1f}{d3:>+8.1f}{stats[best, 4]:>8d}")
    print()
    dm, dc = np.array(dm), np.array(dc)
    dh = np.array(dh) if dh else np.array([np.nan])
    print(f"{'方案':<18}{'|Δ| 中位':>10}{'|Δ| p90':>9}{'|Δ| 最大':>9}")
    print("-" * 48)
    print(f"{'月牙掩膜 PCA':<18}{np.median(dm):>9.2f}°{np.percentile(dm, 90):>8.2f}°{dm.max():>8.2f}°")
    print(f"{'颜色轮廓 PCA':<18}{np.median(dc):>9.2f}°{np.percentile(dc, 90):>8.2f}°{dc.max():>8.2f}°")
    print(f"{'颜色+凸包 PCA':<18}{np.median(dh):>9.2f}°{np.percentile(dh, 90):>8.2f}°{dh.max():>8.2f}°")
    print()
    print(f"比较 {keep} 帧。若颜色轮廓明显更贴合人工轴，说明偏差主要来自"
          "「掩膜是运动月牙」而不是阈值/权重怎么调。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
