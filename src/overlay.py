"""姿态叠加视频：把 ψ_body / ψ_vel / β 画回画面上（交付物 D7 的"叠加视频"部分）。

为什么需要它
------------
项目目标写的是"输入一段固定俯视的漂移视频，输出逐帧的滑移角时序曲线
**与叠加可视化视频**"，成功标准 S2 的判定方式之一也是"目视检查叠加视频"。

原因很实在：**曲线图看不出"这一帧的车头到底朝哪边"**。只有把轴画回车身上，
人才能一眼判断对错 —— 本项目历史上两次最贵的误判（video15 的 PCA 提示偏 30°、
k=1~14 的质心被水雾拖偏）都是靠"把线画回画面"才定论的。所以它是**验收工具**，
不只是演示物料。

画什么
------
  · 车身轴 —— 颜色即可信度：绿=人工标注 / 橙=插值 / 红=PCA 基线 / 灰=无
  · 运动方向箭头（ψ_vel）
  · 左上角数值面板：β、ψ_body、ψ_vel、速度
  · 底部 β 时序带 + 当前帧游标，把单帧放回整段上下文里看

两条与别处一致的约定
--------------------
1. **β 刻度固定 ±90°**。这是结构上界（README 约定 11：车头是由运动方向推出来的，
   |β| 被夹在 ±90），固定刻度才能让不同素材横向比；也才能一眼看出
   "β 顶在 ±90 上"这种越界征兆。
2. **非实时素材（`res.realtime=False`）的速度标"nominal"**。原片加速过，
   px/s 只是名义值（约定 12）；角度量与时间基准无关，照常显示。

文字一律用 ASCII：cv2 的 Hershey 字体不支持中文，硬塞会变问号。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config as C
from . import crops as CR
from . import preprocess as P

# 轴的颜色按**来源**区分，颜色本身就在说"这一帧的轴可不可信"（BGR）
AXIS_STYLE: dict[int, tuple[tuple[int, int, int], str]] = {
    1: ((70, 205, 70), "annotated"),      # 绿：人工标注
    2: ((40, 170, 255), "interp"),        # 橙：插值补的
    3: ((60, 60, 235), "PCA base"),       # 红：PCA 基线（不可当结果）
    0: ((150, 150, 150), "none"),         # 灰：没有轴
}
VEL_COLOR = (255, 200, 70)                # 运动方向箭头
PANEL_BG = (28, 28, 28)
FG = (240, 240, 240)
DIM = (165, 165, 165)
WARN = (60, 200, 255)

BETA_LIMIT = 90.0                         # β 刻度固定 ±90（结构上界，见约定 11）
BETA_WARN = 85.0                          # 超过它 → 车头朝向等于抛硬币


def _panel(img: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.62
           ) -> None:
    """在 img 上叠一块半透明深色底，保证文字在任何画面上都读得清。"""
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    cv2.addWeighted(np.full_like(roi, PANEL_BG), alpha, roi, 1 - alpha, 0, roi)


def _put(img: np.ndarray, text: str, org: tuple[int, int], scale: float,
         color=FG, thick: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _timeline_points(kin: dict, res: P.PreprocessResult, rect: tuple[int, int, int, int]
                     ) -> np.ndarray:
    """把 β 序列预先换算成时序带里的像素点；NaN 处置 NaN，画的时候断线。"""
    x0, y0, w, h = rect
    beta = np.asarray(kin["beta"], dtype=float)
    ts = np.asarray(res.t, dtype=float)
    t0, t1 = float(ts[0]), float(ts[-1])
    span = max(1e-9, t1 - t0)
    pts = np.full((len(beta), 2), np.nan)
    for k in range(len(beta)):
        if not np.isfinite(beta[k]):
            continue
        px = x0 + (ts[k] - t0) / span * (w - 1)
        b = float(np.clip(beta[k], -BETA_LIMIT, BETA_LIMIT))
        py = y0 + h / 2 - b / BETA_LIMIT * (h / 2 - 2)
        pts[k] = (px, py)
    return pts


def _draw_timeline(img: np.ndarray, kin: dict, pts: np.ndarray,
                   rect: tuple[int, int, int, int], k: int) -> None:
    x0, y0, w, h = rect
    _panel(img, x0 - 24, y0 - 16, w + 76, h + 24, alpha=0.74)
    # ±90 与 0 三根参考线。±90 是**结构上界**（约定 11）：曲线贴上去就说明
    # "车头朝向已顶到定头规则的天花板、等于抛硬币"，所以这两根线必须显眼。
    for frac, col, lab in ((0.0, (120, 120, 120), f"+{BETA_LIMIT:.0f}"),
                           (0.5, (150, 150, 150), "0"),
                           (1.0, (120, 120, 120), f"-{BETA_LIMIT:.0f}")):
        yy = int(y0 + h * frac)
        cv2.line(img, (x0, yy), (x0 + w, yy), col, 1, cv2.LINE_AA)
        _put(img, lab, (x0 - 22, yy + 5), 0.42, DIM)
    valid = np.isfinite(pts[:, 0])
    seg = pts[valid]
    if len(seg) >= 2:
        cv2.polylines(img, [seg.astype(np.int32)], False, (250, 250, 250), 2, cv2.LINE_AA)
    px, py = pts[k]
    if np.isfinite(px):
        cv2.line(img, (int(px), y0 - 4), (int(px), y0 + h + 4), (90, 220, 255), 1,
                 cv2.LINE_AA)
        cv2.circle(img, (int(px), int(py)), 5, (60, 230, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (int(px), int(py)), 5, (20, 20, 20), 1, cv2.LINE_AA)
    _put(img, "beta timeline (deg)", (x0 + w + 6, int(y0 + 12)), 0.42, DIM)


def render(spec: C.VideoSpec, res: P.PreprocessResult, kin: dict,
           dets: list | None = None, *, out_height: int = 720,
           with_timeline: bool = True, verbose: bool = False) -> Path:
    """逐帧画好叠加信息，编码成 mp4，返回输出路径。

    按**名义帧率** `res.eff_fps` 编码：非实时素材照原样保持时间轴，
    不引入任何"猜测的加速倍率"（约定 12）。
    """
    ww, wh = res.work_size
    scale = min(1.0, out_height / float(wh))
    W = max(2, int(round(ww * scale)) // 2 * 2)
    H = max(2, int(round(wh * scale)) // 2 * 2)

    out_path = C.OVERLAY_DIR / f"{spec.name}_pose.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fps = float(res.eff_fps) if res.eff_fps and res.eff_fps > 0 else 25.0
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not vw.isOpened():
        raise IOError(f"无法创建视频：{out_path}")

    byk = {d.k: d for d in (dets or [])}
    axis = np.asarray(kin["axis"], dtype=float)
    kind = np.asarray(kin["axis_kind"], dtype=int)
    beta = np.asarray(kin["beta"], dtype=float)
    pbody = np.asarray(kin["psi_body"], dtype=float)
    pvel = np.asarray(kin["psi_vel"], dtype=float)
    speed = np.asarray(kin["speed"], dtype=float)
    size_px = float(kin.get("size_px") or 0.0)

    # 时序带：左右都留出刻度/标签的位置（左边 ±90 刻度、右边标题），别顶到画幅边
    hb = max(64, int(round(0.14 * H))) if with_timeline else 0
    tl_rect = (int(0.11 * W), H - hb + int(0.25 * hb),
               int(0.78 * W), int(0.62 * hb))
    tl_pts = _timeline_points(kin, res, tl_rect) if with_timeline else None

    n_written = 0
    try:
        for k, orig, t, frame in P.iter_work_frames(res, spec.path):
            img = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA) if scale < 1.0 \
                else frame.copy()

            d = byk.get(k)
            cx = cy = None
            if d is not None:
                cx, cy = d.cx * scale, d.cy * scale
            elif np.isfinite(kin["cx"][k]):
                cx, cy = float(kin["cx"][k]) * scale, float(kin["cy"][k]) * scale

            reach = max(34.0, (max(d.w, d.h) if d is not None else size_px) * scale)
            if cx is not None:
                if d is not None:
                    w_half, h_half = d.w * scale / 2, d.h * scale / 2
                    cv2.rectangle(img, (int(cx - w_half), int(cy - h_half)),
                                  (int(cx + w_half), int(cy + h_half)), (110, 110, 110), 1,
                                  cv2.LINE_AA)
                # 车身轴：两端等长，本来就不分车头车尾（约定 2）
                if np.isfinite(axis[k]):
                    seg = CR.axis_segment(cx, cy, axis[k], reach * 0.98)
                    cv2.line(img, tuple(np.int32(seg[0])), tuple(np.int32(seg[1])),
                             AXIS_STYLE[int(kind[k])][0], 2, cv2.LINE_AA)
                if np.isfinite(pvel[k]):
                    a = np.radians(float(pvel[k]))
                    tail = reach * 0.34
                    p0 = (int(cx - np.cos(a) * tail), int(cy - np.sin(a) * tail))
                    p1 = (int(cx + np.cos(a) * reach * 0.86),
                          int(cy + np.sin(a) * reach * 0.86))
                    cv2.arrowedLine(img, p0, p1, VEL_COLOR, 2, cv2.LINE_AA, tipLength=0.13)

            # 左上角数值面板。putText 的 org 是**文字左下角**，所以首行 y 要给足，
            # 否则第一行会被画到画面外（实测会顶掉半行）。坐标必须是 int ——
            # OpenCV 5 对 (float, int) 直接报 Bad argument。
            ln_h = 24
            ph = 166
            _panel(img, 6, 6, int(0.32 * W), ph)
            # 标题行与 beta 之间要留够：beta 的字号更大，挤太近会把标题的升部压住
            sx_, sy_ = 6 + int(6 * scale), 48
            s_txt = 0.60 * scale + 0.12
            s_sm = 0.44 * scale + 0.09
            _put(img, f"{spec.name}   k={k}/{res.n - 1}   t={t:.2f}s   "
                      f"fps {res.eff_fps:.1f}", (sx_, 20), s_sm * 0.94, DIM)
            if np.isfinite(beta[k]):
                col = WARN if abs(beta[k]) > BETA_WARN else FG
                _put(img, f"beta = {beta[k]:+.1f} deg", (sx_, sy_), s_txt, col, 2)
            else:
                _put(img, "beta = --", (sx_, sy_), s_txt, DIM, 2)
            sy_ += ln_h + 8
            if np.isfinite(pbody[k]) and np.isfinite(pvel[k]):
                _put(img, f"psi_body {pbody[k]:6.1f}   psi_vel {pvel[k]:6.1f}  (deg)",
                     (sx_, sy_), s_sm, FG)
            else:
                _put(img, "psi_body   --      psi_vel   --    (deg)", (sx_, sy_), s_sm, DIM)
            sy_ += ln_h
            sp = f"speed {speed[k]:.0f} px/s" if np.isfinite(speed[k]) else "speed  --"
            if not res.realtime:
                sp += " (nominal)"
            if size_px > 0 and np.isfinite(speed[k]):
                sp += f"  = {speed[k] / size_px:.2f} car/s"
            _put(img, sp, (sx_, sy_), s_sm, FG)
            sy_ += ln_h
            style = AXIS_STYLE[int(kind[k])]
            _put(img, f"axis src: {style[1]}", (sx_, sy_), s_sm, style[0])
            sy_ += ln_h
            _put(img, "line = body axis    arrow = motion dir", (sx_, sy_), s_sm * 0.94, DIM)
            if np.isfinite(beta[k]) and abs(beta[k]) > BETA_WARN:
                _put(img, "heading inferred from motion - unreliable",
                     (sx_, sy_ + 18), s_sm * 0.94, WARN)

            if with_timeline and tl_pts is not None:
                _draw_timeline(img, kin, tl_pts, tl_rect, k)

            vw.write(img)
            n_written += 1
            if verbose and n_written % 100 == 0:
                print(f"    {n_written}/{res.n} 帧")
    except Exception:
        # 失败时别留半个 mp4：几百字节的坏文件比没有文件更容易被误当成产物
        vw.release()
        out_path.unlink(missing_ok=True)
        raise
    vw.release()
    if n_written == 0:
        out_path.unlink(missing_ok=True)
        raise IOError(f"{spec.name}：没有可写的帧")
    return out_path
