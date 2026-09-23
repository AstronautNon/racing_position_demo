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
    cx_raw, cy_raw = cx.copy(), cy.copy()

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

    covered = int(np.sum(has))
    return dict(cx=cx, cy=cy, cx_raw=cx_raw, cy_raw=cy_raw,
                axis=axis, conf=conf, area=area,
                covered=covered, n=n, max_gap=max_gap,
                valid_from=int(idx[0]) if len(idx) else -1,
                valid_to=int(idx[-1]) if len(idx) else -1)


def _interp_inplace(values: np.ndarray) -> np.ndarray:
    """对内部缺口做线性插值，两端不外推。"""
    out = values.copy()
    idx = np.nonzero(~np.isnan(out))[0]
    if len(idx) < 2:
        return out
    lo, hi = idx[0], idx[-1]
    span = np.arange(lo, hi + 1)
    out[lo:hi + 1] = np.interp(span, idx, out[idx])
    return out


AXIS_RESID_WIN_S = 0.5


def axis_residual_std(axis: np.ndarray, dt: float,
                      win_s: float = AXIS_RESID_WIN_S) -> float | None:
    """车身轴"抖不抖"：去掉平滑趋势后的残差散布（度）。

    为什么不用普通 std，也不用倍角圆 std：车身轴在整段素材里会**真实转过几十度**
    （video09 约 39°），这两种度量都被真实转动主导，测不出噪声。
    实测 video09 的 PCA 基线与人工标注的圆 std 是 12.1° vs 11.8° —— 几乎一样，
    但两者的 β 中位相差 7.15°、且**单向**（基线系统性高估 |β|）。
    所以"抖不抖"必须先把趋势减掉再量。

    做法：在倍角域（2θ 的 cos/sin）做滑动平均得到趋势（这样 0/180 接缝不会
    把均值拉成 90°），再看逐帧偏离趋势多少。

    实测（窗口 0.5 s，video09）：PCA 基线 3.8°，人工标注 1.6°。
    """
    a = np.asarray(axis, dtype=float)
    m = np.isfinite(a)
    if m.sum() < 5:
        return None
    win = max(5, int(round(win_s / dt)) | 1)
    if win > m.sum():
        win = m.sum() | 1
    k = np.ones(win) / win
    w = np.convolve(m.astype(float), k, "same")
    z = np.radians(2.0 * a)
    u = np.convolve(np.where(m, np.cos(z), 0.0), k, "same") / np.maximum(w, 1e-9)
    v = np.convolve(np.where(m, np.sin(z), 0.0), k, "same") / np.maximum(w, 1e-9)
    trend = np.degrees(np.arctan2(v, u)) / 2.0
    r = (a - trend) % C.BODY_AXIS_MOD
    r = np.where(r > 90.0, r - 180.0, r)
    return float(np.std(r[m]))


def reject_pose_outliers(x: np.ndarray, y: np.ndarray,
                         radius: int = 5, sigma: float = 4.0,
                         floor_frac: float = 0.03,
                         work_width: float = 960.0) -> tuple[np.ndarray, np.ndarray, int]:
    """剔除轨迹上的离群点（车被画面边缘截断、掩膜临时并进阴影/杂物等）。

    做法：用滑动窗口中值得到一条鲁棒轨迹，残留超过
    `max(floor_frac * 画面宽, sigma * 残留中位数)` 的点判为离群，
    置为 NaN，交给后续的插值补齐。

    先做这一步再做 Savitzky-Golay 平滑，否则单个离群点会把整个平滑窗口带偏，
    在速度曲线上制造出上万个 px/s 的假尖峰。
    """
    n = len(x)
    if n < 7:
        return x, y, 0
    xs, ys = x.copy(), y.copy()
    valid = ~np.isnan(xs)
    rx, ry = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        rx[i] = np.nanmedian(xs[lo:hi])
        ry[i] = np.nanmedian(ys[lo:hi])
    resid = np.hypot(xs - rx, ys - ry)
    thr = max(floor_frac * work_width, sigma * np.nanmedian(resid))
    bad = valid & (resid > thr)
    xs[bad] = np.nan
    ys[bad] = np.nan
    return xs, ys, int(bad.sum())


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
            body_axis: np.ndarray | None = None,
            body_mask: np.ndarray | None = None) -> dict:
    """算出速度矢量、航向 psi_vel，以及滑移角 beta。

    body_axis 为 None 时用掩膜 PCA 主轴作基线（噪声大，仅供参考）；
    否则用给定的人工标注轴。body_mask 标记"哪些帧是真正人工标过的"，
    其余非空帧视为插值补出来的 —— 二者在图上要分开画，不能混为一谈。

    返回的 axis_kind 逐帧记录轴的来源：
        0 = 无（NaN）  1 = 人工标注  2 = 人工标注插值  3 = 掩膜 PCA 基线
    """
    s = build_series(dets, res)
    dt = 1.0 / res.eff_fps

    # 1) 先剔离群点，再插值补齐，最后平滑。顺序不能换：
    #    离群点先平滑会被窗口摊开成一整段假轨迹，在速度曲线上形成巨大尖峰。
    cx_r, cy_r, n_out = reject_pose_outliers(
        s["cx_raw"], s["cy_raw"], work_width=float(res.work_size[0]))
    cx = _interp_inplace(cx_r)
    cy = _interp_inplace(cy_r)
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

    # 位置被判为离群的帧
    rejected = np.isnan(cx_r) & ~np.isnan(s["cx_raw"])

    if body_axis is None:
        axis_use = s["axis"]
        # PCA 主轴是从同一个掩膜算出来的，掩膜出错时它一定跟着错，故一并置空
        axis_use = np.where(rejected, np.nan, axis_use)
        axis_kind = np.where(np.isnan(axis_use), 0, 3)
    else:
        axis_use = np.asarray(body_axis, dtype=float)
        axis_kind = np.where(np.isnan(axis_use), 0, 2)
        if body_mask is not None:
            axis_kind = np.where(np.asarray(body_mask, dtype=bool), 1, axis_kind)
        # 人工标注是逐帧的目视测量，与位置轨迹的可靠性无关，
        # 所以**不**按 rejected 置空（车被边缘截断时轴往往仍看得清）。

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
        axis_kind=axis_kind, rejected=rejected,
        dt=dt, size_px=size_px, n_outliers=n_out,
        speed_blps=speed / size_px if size_px and np.isfinite(size_px) else speed * np.nan,
    )


# ---------------------------------------------------------------------------
# 人工标注接口
# ---------------------------------------------------------------------------
def load_annotation_detail(name: str, n: int) -> dict | None:
    """读取人工标注的车身轴，并把稀疏标注插值成可用的逐帧序列。

    文件 outputs/annotations/<name>.csv。第 1 列 `k`（去重序号），
    第 2 列 `axis_deg`（度，mod 180）；以 `#` 开头的行忽略，
    其余多余列（标注台另外记下的点选坐标等）不影响读取。

    为什么要插值：车身轴的角速度在稳态漂移下近似恒定，人工标 50 帧就能
    撑起整条曲线。但插值毕竟不是观测，所以：

      - 只在**两个已标帧之间**插值，两端不外推；
      - 缺口超过 ANNOT_MAX_GAP 帧就不插（"线性转动"的假设撑不住），留 NaN；
      - 返回的 annotated 掩膜标出哪些帧是真正人工标的，供绘图区分。

    返回 dict(axis=插值后序列, annotated=布尔掩膜, raw=稀疏原值, count=标注帧数)。
    """
    p = C.ANNOT_DIR / f"{name}.csv"
    raw = np.full(n, np.nan)
    if not p.exists():
        return None
    with p.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                k = int(float(row[0]))
                a = float(row[1])
            except (ValueError, IndexError):
                continue
            if 0 <= k < n and np.isfinite(a):
                raw[k] = a % C.BODY_AXIS_MOD

    annotated = ~np.isnan(raw)
    count = int(annotated.sum())
    if count == 0:
        return None

    axis = raw.copy()
    idx = np.nonzero(annotated)[0]
    if len(idx) >= 2:
        for a, b in zip(idx[:-1], idx[1:]):
            gap = b - a - 1
            if gap <= 0 or gap > C.ANNOT_MAX_GAP:
                continue
            span = np.arange(a + 1, b)
            # 主轴是 mod 180 的量，用倍角法插值，避免 0/180 附近的跳变被拉成大半圈
            ra = np.radians(np.array([raw[a], raw[b]]) * 2.0)
            t = (span - a) / (b - a)
            re = np.interp(t, [0, 1], np.cos(ra))
            im = np.interp(t, [0, 1], np.sin(ra))
            axis[span] = np.degrees(np.arctan2(im, re)) / 2.0 % C.BODY_AXIS_MOD

    return dict(axis=axis, annotated=annotated, raw=raw, count=count)


def load_annotation(name: str, n: int) -> np.ndarray | None:
    """兼容旧接口：只取插值后的轴序列。"""
    d = load_annotation_detail(name, n)
    return d["axis"] if d is not None else None


def annotate_stats(kin: dict) -> dict:
    """统计一帧段的轴来源构成，用于报告与绘图。"""
    kind = kin.get("axis_kind")
    if kind is None:
        return dict(annotated=0, interp=0, pca=0, none=0)
    return dict(annotated=int((kind == 1).sum()), interp=int((kind == 2).sum()),
                pca=int((kind == 3).sum()), none=int((kind == 0).sum()))


AXIS_SRC_NAMES = {0: "none", 1: "annotated", 2: "interp", 3: "pca"}


def write_series_csv(path: Path, res: PreprocessResult, kin: dict) -> None:
    """导出逐帧运动学序列，供外部分析与后续标注使用。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["k", "frame", "t", "cx", "cy", "vx", "vy", "speed_px_s",
            "speed_bl_per_s", "psi_vel_deg", "axis_deg", "axis_src",
            "psi_body_deg", "beta_deg"]
    kind = kin.get("axis_kind", np.zeros(res.n, dtype=int))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in range(res.n):
            w.writerow([
                k, res.kept_frames[k], f"{res.t[k]:.4f}",
                *[("" if not np.isfinite(v) else f"{v:.4f}") for v in (
                    kin["cx"][k], kin["cy"][k], kin["vx"][k], kin["vy"][k],
                    kin["speed"][k], kin["speed_blps"][k],
                    kin["psi_vel"][k], kin["axis"][k])],
                AXIS_SRC_NAMES.get(int(kind[k]), "?"),
                *[("" if not np.isfinite(v) else f"{v:.4f}") for v in (
                    kin["psi_body"][k], kin["beta"][k])],
            ])
