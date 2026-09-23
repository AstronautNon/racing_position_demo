"""双分支车辆检测。

分支 A —— 静止机位背景建模（零训练）
    中值背景 + 帧差 + 形态学 + 连通域，取面积最大的连通域作为车辆。
    用**滚动窗口**重建背景（每 BG_BLOCK 帧重建一次，用前后各 BG_WINDOW 帧求中值），
    这样即使相机有缓慢漂移（video12 约 23 px）也能保住背景清晰度，
    而不用整段共用一个背景。

分支 B —— 相机运动走 geo-trax
    直接读 drift/results/<name>_*.txt，列定义为（已核对 geo-trax README）：
        0 帧号 | 1 车辆ID | 2~5 未稳定化框(cx,cy,w,h) | 6~9 稳定化框 | 10 类别 | 11 置信度
        | 12 车长 | 13 车宽
    坐标是**原始视频帧**坐标，与本项目裁剪黑边后的工作图不同，需换算。

两个分支都统一输出 Detection（工作图坐标），便于下游共用。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .preprocess import PreprocessResult, iter_work_frames, orig_to_work


@dataclass
class Detection:
    """单车检测结果，坐标统一为工作图坐标。"""

    k: int              # 去重后序号
    frame: int          # 原始帧号
    t: float            # 重建时间轴（秒）
    cx: float           # 质心（掩膜像素质心，比框中心稳）
    cy: float
    w: float            # 外接框宽
    h: float            # 外接框高
    area: float         # 掩膜像素数
    conf: float         # 置信度代理量（掩膜内平均差异 / 255）
    axis_deg: float     # 掩膜 PCA 主轴，mod 180，仅作粗初值
    source: str         # "bgsub" | "geotrax"

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2

    @property
    def size(self) -> float:
        return float(np.hypot(self.w, self.h))


def _iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 分支 A：背景建模
# ---------------------------------------------------------------------------
class BgSubDetector:
    """静止机位的背景建模检测器（单次顺序读盘 + 滚动背景）。"""

    def __init__(self, res: PreprocessResult, path: Path,
                 block: int = C.BG_BLOCK, window: int = C.BG_WINDOW,
                 threshold: int = C.BG_THRESHOLD, weighted_centroid: bool = True):
        self.res = res
        self.path = path
        self.block = block
        self.window = window
        self.threshold = threshold
        # 车辆阴影与车身常被同一个掩膜连通域吃掉。用"按差异强度加权"的质心
        # 可以压制阴影（差异弱）对定位的拉扯，比单纯取掩膜像素质心稳得多。
        self.weighted_centroid = weighted_centroid
        self.min_area = max(C.MIN_AREA_PX, int(C.MIN_AREA_FRAC * res.work_area))
        self.max_area = int(C.MAX_AREA_FRAC * res.work_area)

    def _detect_one(self, gray: np.ndarray, bg: np.ndarray) -> tuple | None:
        diff = cv2.absdiff(gray, bg)
        mask = (diff > self.threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((C.BG_CLOSE_K, C.BG_CLOSE_K), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((C.BG_OPEN_K, C.BG_OPEN_K), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return None

        # 候选筛选：面积区间 + 长宽比上限（排除字幕带这类细长结构）
        cand: list[int] = []
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if area < self.min_area or area > self.max_area:
                continue
            if max(w, h) / max(1, min(w, h)) > C.MAX_ASPECT:
                continue
            cand.append(k)
        if not cand:
            return None

        # 打分：差异能量 = 面积 × 平均差异。
        # 单看面积会被细长的字幕带/反光带抢走（它们面积大但差异弱），
        # 单看平均差异又会偏袒零散的高对比噪点；能量同时要求"大且显著"。
        best_score, best_k = -1.0, -1
        for k in cand:
            score = float(diff[lab == k].sum())
            if score > best_score:
                best_score, best_k = score, k
        k = best_k

        x, y, w, h, area = (int(stats[k, 0]), int(stats[k, 1]),
                            int(stats[k, 2]), int(stats[k, 3]), int(stats[k, 4]))
        sub = (lab[y:y + h, x:x + w] == k)
        ys, xs = np.nonzero(sub)
        if len(xs) < 20:
            return None
        if self.weighted_centroid:
            wts = diff[y:y + h, x:x + w][sub].astype(np.float64)
            tot = wts.sum()
            cx = float((xs * wts).sum() / tot) + x
            cy = float((ys * wts).sum() / tot) + y
        else:
            cx = float(xs.mean()) + x
            cy = float(ys.mean()) + y

        pts = np.stack([xs, ys], 1).astype(np.float32)
        if len(pts) > 2000:                       # 大目标降采样，PCA 结果不受影响
            pts = pts[:: len(pts) // 2000 + 1]
        _, evec = cv2.PCACompute(pts, mean=None)
        axis = float(np.degrees(np.arctan2(evec[0][1], evec[0][0])) % C.BODY_AXIS_MOD)

        conf = float(diff[lab == k].mean()) / 255.0
        return (cx, cy, float(w), float(h), float(area), conf, axis)

    def run(self, verbose: bool = False) -> list[Detection]:
        res = self.res
        n = res.n
        dets: list[Detection] = []
        buf: dict[int, np.ndarray] = {}
        next_block = 0
        upcoming = iter_work_frames(res, self.path)

        def fill_until(limit: int) -> bool:
            """把缓冲区填到至少包含序号 limit 的帧。返回是否成功。"""
            while (not buf) or (max(buf) < limit):
                try:
                    k, _, _, frame = next(upcoming)
                except StopIteration:
                    return False
                buf[k] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return True

        while next_block < n:
            b0 = next_block
            b1 = min(n, b0 + self.block)
            need = min(n - 1, b1 + self.window - 1)
            alive = fill_until(need)
            keys = sorted(buf)
            if not keys:
                break
            avail_hi = keys[-1]
            lo = max(0, b0 - self.window)
            sample = [k for k in keys if lo <= k <= avail_hi][:: C.BG_SAMPLE_STEP]
            if len(sample) < 5:
                sample = keys[: min(len(keys), 8)]
            bg = np.median(np.stack([buf[k] for k in sample]), axis=0).astype(np.uint8)

            for k in range(b0, b1):
                g = buf.get(k)
                if g is None:
                    continue
                got = self._detect_one(g, bg)
                if got is None:
                    continue
                cx, cy, w, h, area, conf, axis = got
                dets.append(Detection(k=k, frame=res.kept_frames[k], t=res.t[k],
                                      cx=cx, cy=cy, w=w, h=h, area=area,
                                      conf=conf, axis_deg=axis, source="bgsub"))

            for k in [x for x in buf if x < b0 - self.window]:
                del buf[k]
            next_block = b1
            if verbose:
                print(f"    block {b0:>4}-{b1 - 1:<4} 背景样本 {len(sample):>3} "
                      f"累计检出 {len(dets)}", flush=True)
            if not alive and b1 >= n:
                break
        return dets


# ---------------------------------------------------------------------------
# 分支 B：geo-trax 输出
# ---------------------------------------------------------------------------
class GeoTraxDetector:
    """读取 geo-trax 的 txt 轨迹，换算到工作图坐标。"""

    def __init__(self, res: PreprocessResult, track_file: Path):
        self.res = res
        self.track_file = track_file
        self.tracks: dict[int, dict[int, Detection]] = {}
        self._load()

    def _load(self) -> None:
        keep_map = {f: k for k, f in enumerate(self.res.kept_frames)}
        with self.track_file.open(newline="") as f:
            for row in csv.reader(f):
                if len(row) < 12:
                    continue
                v = [float(x) for x in row]
                frame, tid = int(v[0]), int(v[1])
                if frame not in keep_map:
                    continue
                k = keep_map[frame]
                cx_w, cy_w = orig_to_work(self.res, v[2], v[3])
                sx, sy = self.res.roi_size[0] / self.res.work_size[0], self.res.roi_size[1] / self.res.work_size[1]
                self.tracks.setdefault(tid, {})[frame] = Detection(
                    k=k, frame=frame, t=self.res.t[k],
                    cx=cx_w, cy=cy_w, w=v[4] / sx, h=v[5] / sy,
                    area=float(v[4] * v[5]), conf=float(v[11]),
                    axis_deg=float("nan"), source="geotrax")

    def track_ids_by_length(self) -> list[tuple[int, int]]:
        return sorted(((tid, len(d)) for tid, d in self.tracks.items()),
                      key=lambda kv: -kv[1])

    def run(self, track_id: int) -> list[Detection]:
        if track_id not in self.tracks:
            raise KeyError(f"{self.track_file.name} 中没有 ID {track_id}")
        return [self.tracks[track_id][f] for f in sorted(self.tracks[track_id])]


# ---------------------------------------------------------------------------
# 后处理：时序一致性（剔除孤立误检）
# ---------------------------------------------------------------------------
def filter_temporal(dets: list[Detection], res: PreprocessResult,
                    radius: int = 4, max_resid: float | None = None) -> tuple[list[Detection], int]:
    """剔除与邻域中位轨迹偏离过大的孤立检测点。

    背景建模会偶发把字幕、水印、反光判成车辆，表现为"只在个别帧出现、
    位置突兀"。这里用滑动窗口中位数做鲁棒筛：与窗口内中位位置的距离
    超过 max_resid 的点被丢弃（首尾用边缘窗口）。
    """
    if len(dets) < 3:
        return dets, 0
    if max_resid is None:
        max_resid = 0.12 * res.work_size[0]
    pos = np.array([[d.cx, d.cy] for d in dets])
    keep = np.ones(len(dets), dtype=bool)
    for i in range(len(dets)):
        lo, hi = max(0, i - radius), min(len(dets), i + radius + 1)
        med = np.median(pos[lo:hi], axis=0)
        if np.linalg.norm(pos[i] - med) > max_resid:
            keep[i] = False
    return [d for d, m in zip(dets, keep) if m], int((~keep).sum())


# ---------------------------------------------------------------------------
# 交叉校验：背景建模 vs geo-trax
# ---------------------------------------------------------------------------
def cross_validate(ours: list[Detection],
                   geo: GeoTraxDetector) -> dict:
    """用 IoU 找出 geo-trax 中与我方检测最吻合的轨迹，评估两分支一致性。"""
    by_frame = {d.frame: d for d in ours}
    best = None
    for tid, n in geo.track_ids_by_length()[:12]:
        common = [f for f in geo.tracks[tid] if f in by_frame]
        if len(common) < 5:
            continue
        ious = [_iou(by_frame[f], geo.tracks[tid][f]) for f in common]
        rec = dict(track_id=tid, frames=len(common), iou_median=float(np.median(ious)),
                   iou_mean=float(np.mean(ious)), iou_p10=float(np.percentile(ious, 10)),
                   iou_over_05=float(np.mean(np.array(ious) > 0.5)))
        if best is None or rec["iou_median"] > best["iou_median"]:
            best = rec
    return best or {}


def run_detection(spec: C.VideoSpec, res: PreprocessResult,
                  verbose: bool = False) -> tuple[list[Detection], dict]:
    """按登记的分支跑检测，返回 (检测列表, 附加信息)。"""
    info: dict = {"branch": spec.detector}
    if spec.detector == "bgsub":
        raw = BgSubDetector(res, spec.path).run(verbose=verbose)
        dets, dropped = filter_temporal(raw, res)
        info.update(raw_count=len(raw), dropped=dropped)
    else:
        if not spec.has_geotrax:
            raise FileNotFoundError(f"{spec.name} 登记为 geo-trax 分支但缺少结果文件")
        geo = GeoTraxDetector(res, spec.geotrax_track)
        dets = geo.run(geo.track_ids_by_length()[0][0])
        info.update(note="取寿命最长的轨迹，未经人工确认")
    info["count"] = len(dets)
    return dets, info
