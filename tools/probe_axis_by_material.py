"""无人监督的可用性评估：各素材的"轮廓轴"与"月牙轴"差多少。

目的：video09 已证实月牙 PCA 有约 7° 单向偏差、而轮廓+凸包只有 1°。
其它素材没有人工标注，但可以用两个**独立**方法的差来筛：
若轮廓轴与月牙轴差得大且方向一致，说明该素材的月牙基线大概率同样带偏；
若两者接近，说明这段素材的月牙基线尚可，标注优先级可以降。

同时报告轮廓分割是否稳定（找到的车身面积变异系数），
颜色线索不成立的素材（灰车/彩地面）会出现面积不稳或找不到大块。

用法：
    /opt/anaconda3/bin/python3 tools/probe_axis_by_material.py [每段采样帧数]
"""

from __future__ import annotations

import csv
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
from probe_shadow import pca_axis  # noqa: E402

MOD = C.BODY_AXIS_MOD
N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def crescent_axes(name: str) -> dict[int, float]:
    p = C.TRACK_DIR / f"{name}.csv"
    if not p.exists():
        return {}
    out = {}
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("axis_deg"):
                try:
                    out[int(float(row["k"]))] = float(row["axis_deg"])
                except ValueError:
                    pass
    return out


def silhouette_axis(wf, ref_c, ref_r):
    """HSV 饱和度过阈值 → 最大近邻连通域 → 凸包 → PCA。返回 (轴, 面积) 或 None。"""
    hsv = cv2.cvtColor(wf, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] <= 12) | (hsv[..., 0] >= 168)) \
        & (hsv[..., 1] >= 90) & (hsv[..., 2] >= 40)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    best, bd = -1, 1e18
    for j in range(1, n):
        if stats[j, 4] < 300:
            continue
        d = float(np.hypot(cents[j][0] - ref_c, cents[j][1] - ref_r))
        if d < bd:
            bd, best = d, j
    if best < 0:
        return None
    cm = (lab == best).astype(np.uint8)
    cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    hull = np.zeros_like(cm)
    cv2.fillConvexPoly(hull, cv2.convexHull(max(cnts, key=cv2.contourArea)), 1)
    hy, hx = np.nonzero(hull)
    if len(hx) < 200:
        return None
    return pca_axis(hx.astype(float), hy.astype(float)), int(hull.sum())


def main() -> int:
    print(f"每段采样 {N_SAMPLE} 帧\n")
    hdr = (f"{'素材':<9}{'帧数':>5}{'有效':>5}{'轴差 中位':>10}{'轴差 p90':>9}"
           f"{'同号':>6}{'轮廓面积CV':>11}  判读")
    print(hdr)
    print("-" * 90)
    for spec in C.active_static_videos():
        cres = crescent_axes(spec.name)
        if not cres:
            print(f"{spec.name:<9}  无轨道文件，跳过")
            continue
        try:
            res = P.preprocess(spec)
            src_path = spec.path          # 源文件缺失是在这里才抛的，必须一起包进 try
        except FileNotFoundError as e:
            print(f"{spec.name:<9}  源文件缺失，跳过（{e}）")
            continue
        # 已标注的素材，tracks.<axis_deg> 已经是人工/插值轴而不是月牙 PCA ——
        # 那就把它当基准用（更准），但必须换个标签说明，别混为一谈。
        ann = K.load_annotation_detail(spec.name, res.n)
        base_lbl = "人工标注" if ann is not None else "月牙PCA"
        ks = sorted(cres)
        picks = [ks[i] for i in np.linspace(0, len(ks) - 1, min(N_SAMPLE, len(ks))).astype(int)]
        picks = sorted(set(picks))
        frames = {}
        for k, _, _, wf in D.iter_work_frames(res, src_path):
            if k in picks:
                frames[k] = wf
                if len(frames) == len(picks):
                    break
        diffs, areas = [], []
        for k in picks:
            wf = frames.get(k)
            if wf is None:
                continue
            # 以月牙掩膜的质心作参考位置（从轨道坐标反推不方便，这里用亮区重心近似）
            got = silhouette_axis(wf, wf.shape[1] * 0.5, wf.shape[0] * 0.5)
            if got is None:
                continue
            a, area = got
            d = (a - cres[k] + 90) % MOD - 90
            diffs.append(d)
            areas.append(area)
        if not diffs:
            print(f"{spec.name:<9}{len(picks):>5}{0:>5}{'—':>10}{'—':>9}{'—':>6}{'—':>11}"
                  f"  颜色线索不成立（找不到车身块）")
            continue
        d = np.array(diffs)
        ad = np.abs(d)
        same = max((d > 0).mean(), (d < 0).mean()) * 100
        cv = float(np.std(areas) / max(1e-9, np.mean(areas)))
        if same >= 80 and np.median(ad) >= 5:
            verdict = "⚠ 月牙基线疑偏，建议人工标注"
        elif np.median(ad) <= 3 and cv < 0.35:
            verdict = "✓ 两法一致且轮廓稳 → 可少标/抽检"
        else:
            verdict = "? 需人工看几帧再定"
        print(f"{spec.name + '[' + base_lbl + ']':<9}{len(picks):>5}{len(diffs):>5}{np.median(ad):>9.2f}°"
              f"{np.percentile(ad, 90):>8.2f}°{same:>5.0f}%{cv:>11.2f}  {verdict}")
    print()
    print("「轴差」= 轮廓轴 − 基准轴（mod 180 折到 ±90）。基准=月牙 PCA（未标注素材）")
    print("        或人工标注（已标注素材）。两法独立，差得多说明基准那侧有问题。")
    print("「轮廓面积CV」= 车身分割面积的变异系数，大说明颜色线索不稳（灰车/彩地面/漏切）。")
    print("注：本表是**无人监督**的筛查，不能替代人工基准；只用来排标注优先级。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
