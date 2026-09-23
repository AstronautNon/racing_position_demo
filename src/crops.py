"""待标注裁图的渲染与坐标换算。

标注台（src/annotate.py）和队列预览（src/select_frames.py）都要做同一件事：
把某一帧里车辆周围裁出来、放大到方便点选的大小，并给出**精确的**坐标换算，
让浏览器里点的两个点能还原成工作图坐标系下的车身轴向。

坐标层级（三层，别混）
-----------------------
1. **原图坐标**：视频解码出来的原始像素。
2. **ROI 坐标**：裁掉黑边之后、缩放之前的坐标。工作图是它的等比缩放。
3. **工作图坐标**：`PreprocessResult.work_size` 下的坐标，检测、运动学、
   角度全在这一层。所有下游读取的 `axis_deg` 都必须是这一层的角度。

裁图直接从**原图**取（1:1 分辨率），而不是从缩过的工作图取，
这样点选时看到的细节最多；再用 `sx/sy` 换算回工作图算角度。

换算之所以能写成 `disp = (work - o) * s` 这种干净形式，是因为裁取矩形的
四边在 ROI 坐标里取整后，反过来把原点换成 `o = fx0 / sxf`（工作图坐标）
——这样取整误差被吸收进 `o`，映射本身保持精确。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .preprocess import PreprocessResult, work_scale


@dataclass
class CropView:
    """一张待标注裁图，以及它的显示↔工作图坐标换算。"""

    path: Path
    ox: float          # 裁图左上角在**工作图**坐标下的位置
    oy: float
    sx: float          # 水平方向：显示像素 / 工作图像素
    sy: float
    iw: int            # 显示尺寸（像素）
    ih: int
    work_rect: tuple[float, float, float, float]   # 裁取矩形（工作图坐标）

    def work_to_disp(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.ox) * self.sx, (y - self.oy) * self.sy

    def disp_to_work(self, X: float, Y: float) -> tuple[float, float]:
        return X / self.sx + self.ox, Y / self.sy + self.oy

    def to_json(self) -> str:
        d = dict(path=str(self.path), ox=self.ox, oy=self.oy, sx=self.sx, sy=self.sy,
                 iw=self.iw, ih=self.ih, work_rect=list(self.work_rect))
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "CropView":
        d = json.loads(text)
        return cls(path=Path(d["path"]), ox=d["ox"], oy=d["oy"], sx=d["sx"], sy=d["sy"],
                   iw=d["iw"], ih=d["ih"], work_rect=tuple(d["work_rect"]))


def crop_rect_for(res: PreprocessResult, cx: float, cy: float, w: float, h: float,
                  pad: float = C.ANNOT_PAD) -> tuple[float, float, float, float]:
    """以车辆为中心取一个正方形裁取矩形（工作图坐标），并夹到画面内。"""
    ww, wh = res.work_size
    side = 0.5 * pad * float(max(w, h))
    x0, y0 = cx - side, cy - side
    x1, y1 = cx + side, cy + side
    # 夹到画面内：夹完可能不再是正方形（车贴边时），映射仍按实际矩形算，故无碍
    x0, x1 = max(0.0, x0), min(float(ww), x1)
    y0, y1 = max(0.0, y0), min(float(wh), y1)
    return (x0, y0, x1, y1)


def render(res: PreprocessResult, spec: C.VideoSpec, k: int,
           rect: tuple[float, float, float, float], out_path: Path,
           disp_max: int = C.ANNOT_DISP_MAX, disp_min: int = C.ANNOT_DISP_MIN,
           quality: int = C.ANNOT_JPEG_Q) -> CropView:
    """按裁取矩形从原图取一块、缩放到便于点选的尺寸，写出 JPEG。"""
    x0, y0, x1, y1 = rect
    l, t, r, b = res.crop
    sxf, syf = work_scale(res)          # 工作图 -> ROI 的放大比

    fx0 = int(round(x0 * sxf)) + l
    fy0 = int(round(y0 * syf)) + t
    fx1 = int(round(x1 * sxf)) + l
    fy1 = int(round(y1 * syf)) + t

    cap = cv2.VideoCapture(str(spec.path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频：{spec.path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, res.kept_frames[k])
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"读取 {spec.name} 第 {res.kept_frames[k]} 帧失败")

    H, W = fr.shape[:2]
    fx0c, fx1c = max(0, fx0), min(W, fx1)
    fy0c, fy1c = max(0, fy0), min(H, fy1)
    patch = fr[fy0c:fy1c, fx0c:fx1c]
    if patch.size == 0:
        raise ValueError(f"{spec.name} k={k} 的裁取矩形落在画面外：{rect}")

    cw, ch = fx1c - fx0c, fy1c - fy0c
    target = min(disp_max, max(disp_min, max(cw, ch)))
    scale = target / max(cw, ch)
    dw, dh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    img = cv2.resize(patch, (dw, dh), interpolation=interp)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])

    # 取整误差全部吸收进原点：o = fx0c / sxf
    return CropView(path=out_path,
                    ox=fx0c / sxf, oy=fy0c / syf,
                    sx=dw / cw * sxf, sy=dh / ch * syf,
                    iw=dw, ih=dh,
                    work_rect=(fx0c / sxf, fy0c / syf, fx1c / sxf, fy1c / syf))


def crop_path(name: str, k: int) -> Path:
    return C.ANNOT_CROP_DIR / name / f"{k:05d}.jpg"


def meta_path(name: str, k: int) -> Path:
    return C.ANNOT_CROP_DIR / name / f"{k:05d}.json"


def ensure(res: PreprocessResult, spec: C.VideoSpec, k: int,
           cx: float, cy: float, w: float, h: float,
           force: bool = False) -> CropView:
    """取（必要时生成）某帧的裁图，带磁盘缓存。

    缓存同时存 JPEG 和换算参数 JSON —— 有了 JSON 就不用再碰视频文件，
    标注台重启后翻页是纯读盘，没有解码开销。
    """
    jp, mp = crop_path(spec.name, k), meta_path(spec.name, k)
    if not force and jp.exists() and mp.exists():
        try:
            v = CropView.from_json(mp.read_text(encoding="utf-8"))
            if v.path.exists():
                return v
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    rect = crop_rect_for(res, cx, cy, w, h)
    v = render(res, spec, k, rect, jp)
    mp.write_text(v.to_json(), encoding="utf-8")
    return v


# ---------------------------------------------------------------------------
# 覆盖物：把提示线算成工作图坐标下的两个端点，浏览器只做点映射
# ---------------------------------------------------------------------------
def axis_segment(cx: float, cy: float, axis_deg: float, length: float
                 ) -> list[list[float]]:
    """沿无向车身轴（mod 180）过质心的一段线段，工作图坐标。"""
    a = np.radians(float(axis_deg))
    dx, dy = np.cos(a) * length / 2, np.sin(a) * length / 2
    return [[cx - dx, cy - dy], [cx + dx, cy + dy]]


def vel_segment(cx: float, cy: float, psi_vel_deg: float, length: float,
                tail: float = 0.0) -> list[list[float]]:
    """从质心出发、指向运动方向 ψ_vel 的箭头，工作图坐标。

    尾部回退 tail 是为了让箭头不要整个压在车身上，看得更清。
    """
    a = np.radians(float(psi_vel_deg))
    x0, y0 = cx - np.cos(a) * tail, cy - np.sin(a) * tail
    return [[x0, y0], [x0 + np.cos(a) * length, y0 + np.sin(a) * length]]


def points_to_axis_deg(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """两个点定出的无向轴角，mod 180，图像右为 0°、顺时针为正。"""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return float("nan")
    return float(np.degrees(np.arctan2(dy, dx)) % C.BODY_AXIS_MOD)
