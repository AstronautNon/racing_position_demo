"""运动学：轨迹平滑、速度矢量、航向 psi_vel，以及滑移角 beta 的接口。

角度约定（与《项目规划.md》§2.5 一致）：
    图像 x 轴向右为 0°，顺时针为正，范围 [0, 360)。
    图像 y 轴向下，故 atan2(dy, dx) 天然就是"顺时针为正"，不需要翻转符号。
    滑移角 beta = psi_body - psi_vel，顺时针为正。

关键点：**psi_body 与 psi_vel 的性质完全不同**
    psi_vel  由轨迹的时间导数得到，平滑、可靠，是 M1 的交付物。
    psi_body 是车身轴向（无向量，mod 180°），必须在每帧上独立估计，
             是 L4 的难点，M1 阶段只提供两个基线：
               (a) 掩膜 PCA 主轴 —— 零标注，但因车辆阴影并入掩膜而噪声很大；
               (b) 人工标注 —— 从 outputs/annotations/<name>.csv 读取。
    取"与运动方向夹角 <= 90°"的那一端即为车头，从而自动消解 180° 歧义。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import config as C
from .detect import Detection
from .preprocess import PreprocessResult

try:  # Savitzky-Golay 保形平滑，优于简单滑动平均
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# 角度工具
# ---------------------------------------------------------------------------
def wrap360(deg: np.ndarray | float) -> np.ndarray | float:
    return np.mod(deg, C.ANGLE_MOD)


def angle_diff(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """a - b，结果落在 (-180, 180]。"""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def undirected_resolve(axis_deg: np.ndarray | float,
                       ref_deg: np.ndarray | float) -> np.ndarray | float:
    """把无向的车身轴（mod 180°）解成有向角：取与参考方向夹角 <= 90° 的那一端。

    漂移时恒有 |beta| < 90°，故参考方向取运动方向 psi_vel 即可自动定出车头。
    """
    a = np.mod(np.asarray(axis_deg, dtype=float), 180.0)
    r = np.mod(np.asarray(ref_deg, dtype=float), 360.0)
    cand = np.stack([a, a + 180.0], axis=-1)
    d = np.abs(angle_diff(cand, np.asarray(r)[..., None]))
    return np.take_along_axis(cand, np.argmin(d, axis=-1)[..., None], axis=-1)[..., 0]


# ---------------------------------------------------------------------------
# 轨迹准备
# ---------------------------------------------------------------------------
def build_series(dets: list[Detection], res: PreprocessResult) -> dict:
    """把稀疏的检测列表摊成等间隔序列，并对内部缺口做线性插值。"""
    n = res.n
    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    axis = np.full(n, np.nan)
    conf = np.full(n, np.nan)
    area = np.full(n, np.nan)
    for d in dets:
        if 0 <= d.k < n:
            cx[d.k] = d.cx
            cy[d.k] = d.cy
            axis[d.k] = d.axis_deg
            conf[d.k] = d.conf
            area[d.k] = d.area

    has = ~np.isnan(cx)
    idx = np.nonzero(has)[0]
    gaps = np.diff(idx) - 1 if len(idx) > 1 else np.array([0])
    max_gap = int(gaps.max()) if len(gaps) else 0

    # 只在检测范围内插值，两端不出外推
    if len(idx) >= 2:
        lo, hi = idx[0], idx[-1]
        span = np.arange(lo, hi + 1)
        cx[lo:hi + 1] = np.interp(span, idx, cx[idx])
        cy[lo:hi + 1] = np.interp(span, idx, cy[idx])
        a = axis[idx]
        if np.all(np.isnan(a)):
            axis[lo:hi + 1] = np.nan
        else:
            # 主轴是 mod 180 的量，用倍角法插值避免 0/180 跳变
            rad = np.radians(a * 2.0)
            re = np.interp(span, idx, np.cos(rad))
            im = np.interp(span, idx, np.sin(rad))
            axis[lo:hi + 1] = np.degrees(np.arctan2(im, re)) / 2.0 % 180.0

    interp_frac = 0.0  # 占位，见返回字典中的 covered/n
    covered = int(np.sum(has))
    return dict(cx=cx, cy=cy, axis=axis, conf=conf, area=area,
                covered=covered, n=n, max_gap=max_gap,
                valid_from=int(idx[0]) if len(idx) else -1,
                valid_to=int(idx[-1]) if len(idx) else -1)


def smooth(values: np.ndarray, window: int = C.SMOOTH_WINDOW) -> np.ndarray:
    """对含 NaN 的序列做平滑，返回同长度数组（NaN 位置保持 NaN）。"""
    out = values.copy()
    m = ~np.isnan(values)
    if m.sum() < 5:
        return out
    v = values[m]
    w = min(window if window % 2 == 1 else window + 1, (len(v) // 2) * 2 - 1)
    if w < 3:
        out[m] = v
        return out
    if _HAS_SCIPY:
        out[m] = savgol_filter(v, w, 2, mode="interp")
    else:
        k = np.ones(w) / w
        out[m] = np.convolve(np.pad(v, w // 2, mode="edge"), k, mode="valid")
    return out


# ---------------------------------------------------------------------------
# 主计算
# ---------------------------------------------------------------------------
def compute(dets: list[Detection], res: PreprocessResult,
            body_axis: np.ndarray | None = None) -> dict:
    """算出速度矢量、航向 psi_vel，以及滑移角 beta。

    body_axis 若为 None，则用掩膜 PCA 主轴作为基线（噪声大，仅供参考）。
    """
    s = build_series(dets, res)
    cx, cy = s["cx"], s["cy"]
    dt = 1.0 / res.eff_fps

    cx_s = smooth(cx)
    cy_s = smooth(cy)

    valid = ~np.isnan(cx_s)
    vx = np.full(res.n, np.nan)
    vy = np.full(res.n, np.nan)
    if valid.sum() >= 3:
        idx = np.nonzero(valid)[0]
        xs, ys = cx_s[idx], cy_s[idx]
        vx[idx] = np.gradient(xs, dt)
        vy[idx] = np.gradient(ys, dt)

    speed = np.hypot(vx, vy)
    psi_vel = wrap360(np.degrees(np.arctan2(vy, vx)))

    # 速度过低时航向无定义（静止帧的 atan2 是噪声）
    psi_vel = np.where(speed >= C.SPEED_MIN_PXS, psi_vel, np.nan)

    axis_use = s["axis"] if body_axis is None else np.asarray(body_axis, dtype=float)
    psi_body = undirected_resolve(axis_use, np.nan_to_num(psi_vel, nan=0.0))
    psi_body = np.where(np.isnan(axis_use), np.nan, psi_body)

    beta = angle_diff(psi_body, psi_vel)
    # β 已由定头规则约束在 (-90, 90]，这里再夹一次防止 NaN 传播导致的越界
    beta = np.where(np.abs(beta) > 90.0, beta - np.sign(beta) * 180.0, beta)
    # 航向无定义的帧（速度过低）滑移角同样不成立
    beta = np.where(np.isnan(psi_vel), np.nan, beta)

    size_px = np.nanmedian([max(d.w, d.h) for d in dets]) if dets else float("nan")
    return dict(
        s=s, cx=cx_s, cy=cy_s, vx=vx, vy=vy, speed=speed,
        psi_vel=psi_vel, axis=axis_use, psi_body=psi_body, beta=beta,
        dt=dt, size_px=size_px,
        speed_blps=speed / size_px if size_px and np.isfinite(size_px) else speed * np.nan,
    )


# ---------------------------------------------------------------------------
# 人工标注接口
# ---------------------------------------------------------------------------
def load_annotation(name: str, n: int) -> np.ndarray | None:
    """读取人工标注的车身轴。

    文件 outputs/annotations/<name>.csv，两列：k(去重序号), axis_deg(度, mod 180)。
    行可以稀疏，未标注的帧返回 NaN。
    """
    p = C.ANNOT_DIR / f"{name}.csv"
    if not p.exists():
        return None
    arr = np.full(n, np.nan)
    with p.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                k = int(float(row[0]))
                a = float(row[1])
            except (ValueError, IndexError):
                continue
            if 0 <= k < n:
                arr[k] = a % C.BODY_AXIS_MOD
    return arr if np.any(~np.isnan(arr)) else None


def write_series_csv(path: Path, res: PreprocessResult, kin: dict) -> None:
    """导出逐帧运动学序列，供外部分析与后续标注使用。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["k", "frame", "t", "cx", "cy", "vx", "vy", "speed_px_s",
            "speed_bl_per_s", "psi_vel_deg", "axis_deg", "psi_body_deg", "beta_deg"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in range(res.n):
            w.writerow([
                k, res.kept_frames[k], f"{res.t[k]:.4f}",
                *[("" if not np.isfinite(v) else f"{v:.4f}") for v in (
                    kin["cx"][k], kin["cy"][k], kin["vx"][k], kin["vy"][k],
                    kin["speed"][k], kin["speed_blps"][k],
                    kin["psi_vel"][k], kin["axis"][k],
                    kin["psi_body"][k], kin["beta"][k])],
            ])
