"""探测：在掩膜里压掉车身阴影，能否把 PCA 主轴拉回人工标注水平。

背景：video09 上 PCA 基线与人工标注的 β 差中位 7.15°、且方向一致 —— 是系统性偏差，
怀疑来自"阴影并进同一连通域，把主轴往一侧拉"。现在有 20 帧人工标注可作基准，
所以可以客观比较几种变体。

临时脚本，放 tools/（入库，因为 14.7 那组数字由它产出，要能复现）。

用法：
    /opt/anaconda3/bin/python3 tools/probe_shadow.py [素材名]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src import detect as D  # noqa: E402
from src import kinematics as K  # noqa: E402
from src import preprocess as P  # noqa: E402

MOD = C.BODY_AXIS_MOD


def gci_std(vals: np.ndarray) -> float:
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return float("nan")
    z = np.radians(2.0 * a)
    r = np.hypot(np.cos(z).mean(), np.sin(z).mean())
    return float(np.degrees(np.sqrt(-2.0 * np.log(max(r, 1e-12)))) / 2.0)


def pca_axis(xs: np.ndarray, ys: np.ndarray, w: np.ndarray | None = None) -> float:
    """主轴角度（mod 180）。w 给定时按强度加权。"""
    if w is None:
        w = np.ones(len(xs), dtype=np.float64)
    w = w.astype(np.float64)
    tw = w.sum()
    mx, my = (xs * w).sum() / tw, (ys * w).sum() / tw
    dx, dy = xs - mx, ys - my
    cov = np.array([[float((dx * dx * w).sum()), float((dx * dy * w).sum())],
                    [float((dx * dy * w).sum()), float((dy * dy * w).sum())]]) / tw
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, int(np.argmax(evals))]
    return float(np.degrees(np.arctan2(v[1], v[0])) % MOD)


def collect(spec: C.VideoSpec, res: P.PreprocessResult):
    """复刻 BgSubDetector 的滚动背景循环，逐帧留下 (k, diff, 选中连通域掩膜)。"""
    det = D.BgSubDetector(res, spec.path)
    buf: dict[int, np.ndarray] = {}
    next_block = 0
    upcoming = D.iter_work_frames(res, spec.path)
    out = []

    def fill_until(limit: int) -> bool:
        while (not buf) or (max(buf) < limit):
            try:
                k, _, _, frame = next(upcoming)
            except StopIteration:
                return False
            buf[k] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return True

    while next_block < res.n:
        b0 = next_block
        b1 = min(res.n, b0 + det.block)
        need = min(res.n - 1, b1 + det.window - 1)
        fill_until(need)
        keys = sorted(buf)
        if not keys:
            break
        lo = max(0, b0 - det.window)
        sample = [k for k in keys if lo <= k <= keys[-1]][:: C.BG_SAMPLE_STEP]
        if len(sample) < 5:
            sample = keys[: min(len(keys), 8)]
        bg = np.median(np.stack([buf[k] for k in sample]), axis=0).astype(np.uint8)

        for k in range(b0, b1):
            g = buf.get(k)
            if g is None:
                continue
            diff = cv2.absdiff(g, bg)
            mask = (diff > det.threshold).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    np.ones((C.BG_CLOSE_K, C.BG_CLOSE_K), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones((C.BG_OPEN_K, C.BG_OPEN_K), np.uint8))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            if n <= 1:
                continue
            cand = []
            for j in range(1, n):
                x, y, w, h, area = stats[j]
                if area < det.min_area or area > det.max_area:
                    continue
                if max(w, h) / max(1, min(w, h)) > C.MAX_ASPECT:
                    continue
                cand.append(j)
            if not cand:
                continue
            bk = max(cand, key=lambda j: float(diff[lab == j].sum()))
            x, y, w, h, area = stats[bk]
            sub = lab[y:y + h, x:x + w] == bk
            ys, xs = np.nonzero(sub)
            if len(xs) < 20:
                continue
            gx, gy = xs + x, ys + y
            dvals = diff[y:y + h, x:x + w][sub].astype(np.float64)
            out.append((k, gx, gy, dvals, (float(x), float(y), float(w), float(h))))
        for x in [t for t in buf if t < b0 - det.window]:
            del buf[x]
        next_block = b1
    return out


VARIANTS = ["V0 基线(全掩膜等权)", "V1 全掩膜+强度加权", "V2 核心(>0.5*p90)等权",
            "V3 核心(>中位)等权", "V4 核心(>0.5*p90)+强度加权"]


def variants_axis(gx, gy, dvals):
    p90 = float(np.percentile(dvals, 90))
    med = float(np.median(dvals))
    core_half = dvals > 0.5 * p90
    core_med = dvals > med
    res = {}
    res[VARIANTS[0]] = pca_axis(gx, gy)
    res[VARIANTS[1]] = pca_axis(gx, gy, dvals)
    res[VARIANTS[2]] = pca_axis(gx[core_half], gy[core_half]) if core_half.sum() >= 20 else np.nan
    res[VARIANTS[3]] = pca_axis(gx[core_med], gy[core_med]) if core_med.sum() >= 20 else np.nan
    res[VARIANTS[4]] = (pca_axis(gx[core_half], gy[core_half], dvals[core_half])
                        if core_half.sum() >= 20 else np.nan)
    res["_keep_frac"] = float(core_half.sum()) / len(dvals)
    return res


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "video09"
    spec = C.get(name)
    res = P.preprocess(spec)
    rows = collect(spec, res)
    print(f"[{name}] 复刻检出 {len(rows)}/{res.n} 帧")

    manual = K.load_annotation_detail(name, res.n)
    if manual is None:
        print("该素材没有人工标注，无法作基准。")
        return 1
    raw = manual["raw"]

    data = {}
    for k, gx, gy, dvals, _ in rows:
        data[k] = variants_axis(gx, gy, dvals)

    print()
    hdr = f"{'k':>4}{'人工轴':>9}" + "".join(f"{v.split()[0]:>9}" for v in VARIANTS)
    print(hdr)
    print("-" * len(hdr))
    per = {v: [] for v in VARIANTS}
    signed = {v: [] for v in VARIANTS}
    for k in sorted(data):
        if not np.isfinite(raw[k]):
            continue
        line = f"{k:>4}{raw[k]:>9.1f}"
        for v in VARIANTS:
            a = data[k][v]
            if np.isfinite(a):
                d = (a - raw[k] + 90.0) % MOD - 90.0
                per[v].append(abs(d))
                signed[v].append(d)
                line += f"{d:>+9.1f}"
            else:
                line += f"{'—':>9}"
        print(line)

    print()
    print(f"{'变体':<26}{'|Δ| 中位':>10}{'|Δ| p90':>9}{'符号一致':>10}{'有符号均值':>12}")
    print("-" * 70)
    for v in VARIANTS:
        if not per[v]:
            continue
        a = np.array(per[v])
        s = np.array(signed[v])
        same = max((s > 0).mean(), (s < 0).mean()) * 100
        print(f"{v:<26}{np.median(a):>9.2f}°{np.percentile(a, 90):>8.2f}°"
              f"{same:>9.0f}%{s.mean():>+11.2f}°")
    print()
    print("「符号一致」= 偏差同号的比例。越接近 100% 说明是系统性偏差（可校准），"
          "接近 50% 说明是随机噪声。")
    kf = np.mean([d["_keep_frac"] for d in data.values()])
    print(f"V2/V4 核心保留的掩膜像素占比：中位 {np.median([d['_keep_frac'] for d in data.values()]):.2f}"
          f"（均值 {kf:.2f}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
