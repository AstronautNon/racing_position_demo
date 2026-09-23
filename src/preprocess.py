"""预处理：黑边检测与裁剪、重复帧剔除、时间轴重建、背景模型可用性判定。

为什么这几件事必须捆绑做（详见《项目规划.md》§12.1 / §13.1）：

1. **重复帧**：素材多为二次录制的屏幕内容，同一原始画面被重复写入多帧。
   若不去重，相邻帧完全相同会让帧间差分速度归零，在滑移角曲线上制造假毛刺。
2. **时间轴重建**：去重后帧序不再对应原始 fps，必须用有效帧率重算时间戳，
   否则速度会整体缩水（video01 的有效时长只剩原始帧数的 1/3 左右）。
3. **黑边**：黑边会拖累背景建模与像素统计，也会让"车是否在画面内"误判。

关于相机运动判定的一个方法论决定
--------------------------------
早期用相位相关法测"相对参考帧的位移"，在低对比度/重复纹理的地面（如 video12 的碎石地）
上会出现多峰失锁，估出的位移在不同参考帧之间能差一个数量级，**不可靠**。
ORB + RANSAC 单应虽更稳（video02/07/09/13/15 内点率 83%~98%），但在同一类画面上同样会崩
（video12 内点率仅 24%~35%）。

因此这里改用**直接检验问题本身**的指标：背景模型残留差异占比
——建中值背景，逐帧求差分掩膜，扣掉最大连通域（车辆）后剩余的差异像素占比。
它直接回答"一个中值背景能不能代表整段素材"，且对相机运动、光照变化都敏感。
实测结果与人工目视判断完全一致（见 §13 与 outputs/reports 下的报告）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config as C

SCHEMA = 2  # 预处理缓存结构版本，不匹配则重算

_HANNING_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _hanning(shape: tuple[int, int]) -> np.ndarray:
    if shape not in _HANNING_CACHE:
        h, w = shape
        _HANNING_CACHE[shape] = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    return _HANNING_CACHE[shape]


@dataclass
class PreprocessResult:
    """一段素材的预处理结果。"""

    schema: int
    name: str
    src_fps: float
    src_size: tuple[int, int]           # (宽, 高)
    total_frames: int
    crop: tuple[int, int, int, int]     # 裁掉的黑边 (左, 上, 右, 下)，像素
    roi_size: tuple[int, int]           # 裁剪后尺寸
    work_size: tuple[int, int]          # 工作图尺寸
    trim: tuple[int, int]               # 实际使用的原始帧区间 [start, end)
    kept_frames: list[int]              # 去重后保留的原始帧号
    t: list[float]                      # 重建时间轴（秒）
    eff_fps: float                      # 有效帧率
    dup_ratio: float                    # 重复帧比例
    bg_residual_frac: float             # 背景模型残留差异占比（扣掉车辆后）
    camera_static: bool                 # 背景模型是否可用
    warnings: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.kept_frames)

    @property
    def duration(self) -> float:
        return self.t[-1] if self.t else 0.0

    @property
    def work_area(self) -> int:
        return self.work_size[0] * self.work_size[1]

    def to_json(self) -> str:
        d = dict(self.__dict__)
        for k in ("src_size", "crop", "roi_size", "work_size", "trim"):
            d[k] = list(d[k])
        return json.dumps(d, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "PreprocessResult":
        d = json.loads(text)
        for k in ("src_size", "crop", "roi_size", "work_size", "trim"):
            d[k] = tuple(d[k])
        return cls(**d)

    def summary_line(self) -> str:
        w, h = self.src_size
        return (
            f"{self.name:<8} {w}x{h} @{self.src_fps:.0f}fps "
            f"| 用帧 {self.trim[0]}~{self.trim[1]}（{self.trim[1] - self.trim[0]} 帧）"
            f"→ 唯一 {self.n}（重复 {self.dup_ratio * 100:.1f}%）"
            f"| 有效 {self.duration:.1f}s @{self.eff_fps:.1f}fps "
            f"| 背景残留 {self.bg_residual_frac * 100:.2f}% "
            f"{'静止' if self.camera_static else '相机在动'}"
        )


# ---------------------------------------------------------------------------
# 黑边检测
# ---------------------------------------------------------------------------
def detect_black_bars(path: Path, n_scan: int = 40, dark: int = 24,
                      max_frac: float = 0.25) -> tuple[int, int, int, int]:
    """检测上下左右的黑边宽度。

    做法：抽样若干帧，逐列/逐行取最大值（黑边列的最大值也接近 0），
    从四周向内扫描到第一个非暗的行/列。跨样本取最大值，宁可少裁不可多裁；
    最后按 max_frac 限幅，防止画面边缘本身很暗时误裁。
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频：{path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0 or W <= 0 or H <= 0:
        cap.release()
        return (0, 0, 0, 0)

    best = [0, 0, 0, 0]
    for i in np.linspace(0, total - 1, min(n_scan, total)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        h, w = g.shape
        l = int(np.argmax(g.max(axis=0) > dark))
        r = w - int(np.argmax(g.max(axis=0)[::-1] > dark))
        t = int(np.argmax(g.max(axis=1) > dark))
        b = h - int(np.argmax(g.max(axis=1)[::-1] > dark))
        best = [max(best[0], l), max(best[1], t), max(best[2], w - r), max(best[3], h - b)]
    cap.release()

    best[0] = min(best[0], int(W * max_frac))
    best[2] = min(best[2], int(W * max_frac))
    best[1] = min(best[1], int(H * max_frac))
    best[3] = min(best[3], int(H * max_frac))
    return (best[0], best[1], best[2], best[3])


# ---------------------------------------------------------------------------
# 背景模型可用性：扣掉车辆后剩余多少差异像素
# ---------------------------------------------------------------------------
def _bg_residual_frac(smalls: list[np.ndarray], thr: int = 30) -> float:
    """中值背景 vs 各帧，扣掉最大连通域(车辆)后的差异像素占比，取中位数。"""
    bg = np.median(np.stack(smalls), axis=0)
    fracs = []
    for g in smalls:
        d = np.abs(g - bg)
        m = (d > thr).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 1:
            k = 1 + int(np.argmax(stats[1:, 4]))
            m[lab == k] = 0
        fracs.append(float(m.mean()))
    return float(np.median(fracs))


# ---------------------------------------------------------------------------
# 主扫描：去重 + 时间轴 + 背景可用性，单次顺序读盘完成
# ---------------------------------------------------------------------------
def _scan(path: Path, crop: tuple[int, int, int, int],
          trim_head: int, trim_tail: int) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频：{path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    l, t, r, b = crop
    x0, y0, x1, y1 = l, t, W - r, H - b
    start = max(0, trim_head)
    end = min(total, total - trim_tail)
    if end - start < 2:
        cap.release()
        raise ValueError(f"{path.name}：裁掉首尾后只剩 {end - start} 帧，无法处理")

    dw, dh = C.DEDUP_SIZE
    sample_step = max(1, (end - start) // 200)
    prev: np.ndarray | None = None
    kept: list[int] = []
    smalls: list[np.ndarray] = []

    for i in range(end):
        ok, fr = cap.read()
        if not ok:
            break
        if i < start:
            continue
        roi = fr[y0:y1, x0:x1]
        g = cv2.cvtColor(cv2.resize(roi, (dw, dh), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.int16)
        if prev is None:
            kept.append(i)
        else:
            changed = float((np.abs(g - prev) > C.DEDUP_PIXEL_THR).mean())
            if changed >= C.DEDUP_FRAC:
                kept.append(i)
        prev = g
        if (i - start) % sample_step == 0 and len(smalls) < 200:
            smalls.append(g.astype(np.float32))
    cap.release()

    n_used = end - start
    dup_ratio = 1.0 - len(kept) / n_used
    eff_fps = fps * len(kept) / n_used if n_used else fps
    residual = _bg_residual_frac(smalls) if len(smalls) >= 8 else float("nan")
    ww = C.work_width_for(x1 - x0)

    return dict(
        schema=SCHEMA,
        src_fps=fps,
        src_size=(W, H),
        total_frames=total,
        crop=(l, t, r, b),
        roi_size=(x1 - x0, y1 - y0),
        work_size=(ww, int(round(ww * (y1 - y0) / (x1 - x0)))),
        trim=(start, end),
        kept_frames=kept,
        t=[k / eff_fps for k in range(len(kept))],
        eff_fps=eff_fps,
        dup_ratio=dup_ratio,
        bg_residual_frac=residual,
        camera_static=bool(residual < C.BG_RESIDUAL_MAX) if np.isfinite(residual) else False,
    )


def preprocess(spec: C.VideoSpec, force: bool = False) -> PreprocessResult:
    """预处理一段素材，结果缓存到 outputs/preprocess/<name>.json。"""
    cache = C.PREPROC_DIR / f"{spec.name}.json"
    if cache.exists() and not force:
        try:
            got = PreprocessResult.from_json(cache.read_text(encoding="utf-8"))
            if got.schema == SCHEMA:
                return got
        except (TypeError, KeyError, json.JSONDecodeError):
            pass  # 旧结构，重算

    crop = detect_black_bars(spec.path)
    raw = _scan(spec.path, crop, spec.trim_head, spec.trim_tail)
    warns: list[str] = []

    if not raw["camera_static"]:
        msg = (f"背景模型残留差异 {raw['bg_residual_frac'] * 100:.2f}%，"
               f"超过 {C.BG_RESIDUAL_MAX * 100:.0f}% 阈值")
        msg += "，与登记的静止机位不符，请复核" if spec.camera == "static" else "，与登记相符，走 geo-trax 分支"
        warns.append(msg)
    if raw["dup_ratio"] < 0.02:
        warns.append("几乎无重复帧，时间轴重建无实质影响")
    elif raw["dup_ratio"] > 0.30:
        warns.append(f"重复帧比例 {raw['dup_ratio'] * 100:.1f}%，"
                     f"有效时长被压缩到 {raw['t'][-1]:.1f}s，样本量受限")
    if sum(crop) > 0:
        warns.append(f"裁掉黑边 {crop}")

    res = PreprocessResult(name=spec.name, warnings=warns, **raw)
    cache.write_text(res.to_json(), encoding="utf-8")
    return res


def load(name: str) -> PreprocessResult:
    cache = C.PREPROC_DIR / f"{name}.json"
    if not cache.exists():
        raise FileNotFoundError(f"{name} 尚未预处理，先运行 preprocess 阶段")
    return PreprocessResult.from_json(cache.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 取帧工具
# ---------------------------------------------------------------------------
def iter_work_frames(res: PreprocessResult, path: Path, step: int = 1):
    """顺序产出 (去重序号, 原始帧号, 时间戳, 工作图BGR)。"""
    l, t, r, b = res.crop
    ww, wh = res.work_size
    keep_map = {f: k for k, f in enumerate(res.kept_frames)}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频：{path}")
    for i in range(res.trim[1]):
        ok, fr = cap.read()
        if not ok:
            break
        k = keep_map.get(i)
        if k is None or k % step:
            continue
        roi = fr[t:fr.shape[0] - b, l:fr.shape[1] - r]
        yield k, i, res.t[k], cv2.resize(roi, (ww, wh), interpolation=cv2.INTER_AREA)
    cap.release()


def read_work_frame(res: PreprocessResult, path: Path, orig_frame: int) -> np.ndarray:
    """随机读取某一原始帧（工作图）。窗口类操作会用，故单独提供。"""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, orig_frame)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"读取 {path.name} 第 {orig_frame} 帧失败")
    l, t, r, b = res.crop
    roi = fr[t:fr.shape[0] - b, l:fr.shape[1] - r]
    return cv2.resize(roi, res.work_size, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# 坐标换算：工作图 <-> 原图
# ---------------------------------------------------------------------------
def work_scale(res: PreprocessResult) -> tuple[float, float]:
    """工作图 1 像素对应原图多少像素。"""
    return (res.roi_size[0] / res.work_size[0], res.roi_size[1] / res.work_size[1])


def work_to_orig(res: PreprocessResult, x: float, y: float) -> tuple[float, float]:
    sx, sy = work_scale(res)
    return x * sx + res.crop[0], y * sy + res.crop[1]


def orig_to_work(res: PreprocessResult, x: float, y: float) -> tuple[float, float]:
    sx, sy = work_scale(res)
    return (x - res.crop[0]) / sx, (y - res.crop[1]) / sy
