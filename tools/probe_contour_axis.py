"""验证「通用轮廓轴」（`detect.contour_axis`）与人工标注的贴合度。

与 `probe_color_axis.py` 的区别：那个脚本把 HSV 阈值写死成红色（只对 video09 的
深红车成立），这里是**与车色无关**的通用实现 —— 用"框外像素估地面色 + 框内 Otsu"
分割车身，所以蓝车/黄车/白车都能试。

用法：
    /opt/anaconda3/bin/python3 tools/probe_contour_axis.py video09
    /opt/anaconda3/bin/python3 tools/probe_contour_axis.py video09 video02 video03

有真值（`outputs/annotations/<名>.csv`）时报与真值的偏差；
无真值时退化为"与月牙 PCA 轴之差"，只能看两法分歧有多大，不能当精度。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import annotate as A          # noqa: E402
from src import config as C            # noqa: E402
from src import detect as D            # noqa: E402
from src import preprocess as P        # noqa: E402

MOD = C.BODY_AXIS_MOD


def diff_mod(a: float, b: float) -> float:
    """两角之差折到 ±90（无向轴）。"""
    return abs((a - b + 90.0) % MOD - 90.0)


def check(name: str) -> dict:
    spec = C.get(name)
    if spec is None or not spec.path.exists():
        print(f"\n[{name}] 源文件不存在，跳过")
        return {}
    res = P.preprocess(spec)
    dets = D.BgSubDetector(res, spec.path).run()
    by_k = {d.k: d for d in dets}
    frames: dict[int, np.ndarray] = {}
    for k, _, _, wf in D.iter_work_frames(res, spec.path):
        if k in by_k:
            frames[k] = wf

    labels = A.read_labels(name)
    truth = {k: v["axis_deg"] for k, v in labels.items() if "axis_deg" in v}
    mode = "对比人工真值" if truth else "无真值，仅看与月牙轴的分歧"

    print(f"\n[{name}] 检出 {len(by_k)} 帧，人工真值 {len(truth)} 帧 — {mode}")
    hdr = f"{'k':>5}{'月牙轴':>9}{'轮廓轴':>9}{'真值':>9}{'Δ轮廓-真':>10}{'Δ轮廓-月牙':>12}{'前景占比':>10}"
    print(hdr)
    print("-" * len(hdr))

    d_truth, d_moon, quals, miss = [], [], [], 0
    for k in sorted(by_k):
        d = by_k[k]
        wf = frames.get(k)
        if wf is None:
            continue
        got = D.contour_axis(wf, d.cx, d.cy, d.w, d.h)
        if got is None:
            miss += 1
            print(f"{k:>5}{d.axis_deg:>9.1f}{'失败':>9}")
            continue
        axis, q = got
        quals.append(q)
        dm = diff_mod(axis, d.axis_deg)
        d_moon.append(dm)
        if k in truth:
            dt = diff_mod(axis, truth[k])
            d_truth.append(dt)
            print(f"{k:>5}{d.axis_deg:>9.1f}{axis:>9.1f}{truth[k]:>9.1f}"
                  f"{dt:>+10.1f}{dm:>+12.1f}{q:>10.3f}")
        else:
            print(f"{k:>5}{d.axis_deg:>9.1f}{axis:>9.1f}{'—':>9}{'—':>10}{dm:>+12.1f}{q:>10.3f}")

    print()
    print(f"分割成功率 {len(d_moon)}/{len(d_moon) + miss}")
    if d_truth:
        a = np.array(d_truth)
        print(f"轮廓轴 vs 人工真值：|Δ| 中位 {np.median(a):.2f}°  "
              f"p90 {np.percentile(a, 90):.2f}°  最大 {a.max():.2f}°  (n={len(a)})")
    if d_moon:
        b = np.array(d_moon)
        print(f"轮廓轴 vs 月牙 PCA：|Δ| 中位 {np.median(b):.2f}°  "
              f"p90 {np.percentile(b, 90):.2f}°  最大 {b.max():.2f}°")
    if quals:
        print(f"前景占框面积比：中位 {np.median(quals):.3f}（过小说明分割没抓到车身）")
    return {"name": name, "n_truth": len(d_truth),
            "med_truth": float(np.median(d_truth)) if d_truth else None,
            "max_truth": float(np.max(d_truth)) if d_truth else None,
            "med_moon": float(np.median(d_moon)) if d_moon else None,
            "n_ok": len(d_moon), "miss": miss}


def main(argv: list[str]) -> int:
    names = [a for a in argv[1:] if not a.startswith("-")] or ["video09"]
    print("通用轮廓轴验证（框外估地面色 + 框内 Otsu 分割 + 凸包 + PCA）")
    print("=" * 92)
    out = [check(n) for n in names]
    print("\n" + "=" * 92)
    print(f"{'素材':<10}{'有真值帧':>9}{'|Δ|中位':>9}{'|Δ|最大':>9}{'与月牙轴中位':>13}{'分割成功':>9}")
    print("-" * 62)
    for o in out:
        if not o:
            continue
        f = lambda v, w=8: (f"{v:.2f}°" if v is not None else "—").rjust(w)   # noqa: E731
        print(f"{o['name']:<10}{o['n_truth']:>9}{f(o['med_truth'], 9)}"
              f"{f(o['max_truth'], 9)}{f(o['med_moon'], 13)}{o['n_ok']:>9}")
    print("\n说明：有真值的行才是精度；只有「与月牙轴之差」的行只能看分歧，不能当精度。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
