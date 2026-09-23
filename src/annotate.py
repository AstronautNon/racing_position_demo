"""标注台：在浏览器里点两下车身轴线，产出滑移角所需的 psi_body。

为什么只要点两个点
------------------
滑移角 beta = psi_body - psi_vel，其中 psi_body 是**车身轴**。
车身轴是无向的（mod 180°）——"车头朝左"和"车头朝右"是同一条轴。
漂移时恒有 |beta| < 90°，所以取"与运动方向夹角 <= 90°"的那一端即为车头，
180° 歧义可以自动消解（见 kinematics.undirected_resolve）。

结论：标注时**只需沿车身点一条轴线，两端不分前后**。车头车尾是程序推出来的，
不需要人判断，于是每帧的标注成本降到两次点击。

三种模式
--------
  --serve    起一个本地 HTTP 服务，浏览器里边点边存（推荐）。
             保存直接落盘到 outputs/annotations/<name>.csv，无需手工搬文件。
  --export   导出一份**自包含**的 HTML（图片以 base64 内嵌），
             双击即用、不需服务；标注存 localStorage，"导出 CSV"按钮下载文件。
             适合离线或想换个环境标的时候。
  --status   只看进度与一致性体检，不启动任何界面。

坐标层级（三层，与 src/crops.py 的说明一致，别混）
--------------------------------------------------
  原图 -> ROI（裁黑边后） -> 工作图（检测/运动学/角度都在这一层）
裁图从**原图** 1:1 取，故点选精度最高；点选坐标先换算回工作图再算角度。
浏览器只做"显示坐标 -> 工作坐标"的线性映射，角度计算放在能验算的一侧。

产物
----
  outputs/annotations/<name>.csv        人工标注（k, axis_deg + 原始点选坐标供复核）
  outputs/annotations/web/<name>.html   自包含标注台（--export）

CSV 格式（kinematics.load_annotation_detail 只读前两列，其余供追溯）
    # 注释行以 # 开头，读取时忽略
    k, axis_deg, x1, y1, x2, y2, src, ts
点选坐标存的是**工作图坐标**（不是显示坐标）——这样即便裁图参数变了，
旧标注也能重新画出来，且 axis_deg 可复算。
src 记 manual / hint：它区分"人点出来的"与"人确认了 PCA 提示"，
两者都是人工确认的结果，但审计时应当能分开。
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import config as C
from . import crops as CR
from . import kinematics as K
from . import preprocess as P

WEB_DIR = C.ANNOT_WEB_DIR
LABEL_COLS = ["k", "axis_deg", "x1", "y1", "x2", "y2", "src", "ts"]


# ---------------------------------------------------------------------------
# 队列读取
# ---------------------------------------------------------------------------
@dataclass
class QueueItem:
    k: int
    frame: int
    t: float
    cx: float
    cy: float
    w: float
    h: float
    conf: float
    hint: float                 # 掩膜 PCA 主轴（mod 180），粗初值
    psi_vel: float | None
    speed_blps: float | None
    prio: int

    def bbox(self) -> tuple[float, float, float, float]:
        return (self.cx - self.w / 2, self.cy - self.h / 2,
                self.cx + self.w / 2, self.cy + self.h / 2)


def queue_path(name: str) -> Path:
    return C.ANNOT_QUEUE_DIR / f"{name}.csv"


def read_queue(name: str) -> list[QueueItem]:
    """读待标注队列。缺列时用安全默认值，队列缺失则返回空表。"""
    p = queue_path(name)
    if not p.exists():
        return []
    items: list[QueueItem] = []
    with p.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#")]
    if not rows:
        return []
    head = [c.strip() for c in rows[0]]

    def idx(col: str, default: int = -1) -> int:
        return head.index(col) if col in head else default

    def num(row: list[str], i: int, default: float = float("nan")) -> float:
        if i < 0 or i >= len(row) or row[i] == "":
            return default
        try:
            return float(row[i])
        except ValueError:
            return default

    i_k, i_fr, i_t = idx("k"), idx("frame"), idx("t")
    i_cx, i_cy, i_w, i_h = idx("cx"), idx("cy"), idx("w"), idx("h")
    i_cf, i_hint = idx("conf"), idx("hint_axis_deg")
    i_pv, i_sp, i_pr = idx("psi_vel_deg"), idx("speed_blps"), idx("prio")
    for n, row in enumerate(rows[1:]):
        if not row:
            continue
        k = int(num(row, i_k, -1))
        if k < 0:
            continue
        pv, sp = num(row, i_pv), num(row, i_sp)
        items.append(QueueItem(
            k=k, frame=int(num(row, i_fr, -1)), t=num(row, i_t),
            cx=num(row, i_cx), cy=num(row, i_cy),
            w=num(row, i_w), h=num(row, i_h), conf=num(row, i_cf),
            hint=num(row, i_hint) % C.BODY_AXIS_MOD,
            psi_vel=None if not np.isfinite(pv) else pv,
            speed_blps=None if not np.isfinite(sp) else sp,
            prio=int(num(row, i_pr, n)),
        ))
    items.sort(key=lambda d: d.prio)
    return items


# ---------------------------------------------------------------------------
# 标注读写（原子写：标注是人工劳动，不能因为中途崩溃丢掉）
# ---------------------------------------------------------------------------
def labels_path(name: str) -> Path:
    return C.ANNOT_DIR / f"{name}.csv"


def read_labels(name: str) -> dict[int, dict]:
    p = labels_path(name)
    out: dict[int, dict] = {}
    if not p.exists():
        return out
    with p.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#")]
    if not rows:
        return out
    head = [c.strip() for c in rows[0]]
    if "k" not in head:                      # 无表头的旧格式：按位置解析
        rows, head = [["k", "axis_deg", "x1", "y1", "x2", "y2", "src", "ts"]] + rows, \
                     ["k", "axis_deg", "x1", "y1", "x2", "y2", "src", "ts"]
    for row in rows[1:]:
        if len(row) < 2:
            continue
        try:
            k = int(float(row[0]))
            a = float(row[1])
        except ValueError:
            continue
        if not np.isfinite(a):
            continue
        rec: dict = {"k": k, "axis_deg": a % C.BODY_AXIS_MOD}
        for j, col in enumerate(head[2:], start=2):
            if j < len(row) and row[j] != "":
                rec[col] = row[j]
        out[k] = rec
    return out


def write_labels(name: str, labels: dict[int, dict]) -> Path:
    p = labels_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".csv.tmp")
    # 全部撤销后不留空壳文件：无标注 == 无文件，免得 ls 里出现只剩表头的 csv 让人误解
    if not labels:
        p.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
        return p
    with tmp.open("w", newline="", encoding="utf-8") as f:
        f.write(f"# {name} 人工标注的车身轴（mod 180 度），由 python -m src.annotate 生成\n")
        f.write("# axis_deg 是唯一被下游读取的字段；x1..y2 是工作图坐标下的原始点选，供复核\n")
        f.write("# src: manual=人工点选 / hint=人工确认了 PCA 提示轴\n")
        f.write(",".join(LABEL_COLS) + "\n")
        for k in sorted(labels):
            r = labels[k]
            row = [str(k), f"{float(r['axis_deg']):.3f}"]
            for col in ("x1", "y1", "x2", "y2"):
                v = r.get(col)
                row.append("" if v is None else f"{float(v):.2f}")
            row.append(str(r.get("src", "manual")))
            row.append(str(r.get("ts", "")))
            f.write(",".join(row) + "\n")
    os.replace(tmp, p)
    return p


def upsert_label(name: str, rec: dict) -> dict:
    labels = read_labels(name)
    labels[int(rec["k"])] = rec
    write_labels(name, labels)
    return labels


def drop_label(name: str, k: int) -> dict:
    labels = read_labels(name)
    labels.pop(int(k), None)
    write_labels(name, labels)
    return labels


# ---------------------------------------------------------------------------
# 载荷：把一帧要用到的东西全部算好，浏览器不再做任何几何
# ---------------------------------------------------------------------------
def _finite(seg) -> bool:
    return all(np.isfinite(v) for pt in seg for v in pt)


def fit_segment(seg, iw: float, ih: float, margin: float = 5.0):
    """沿原方向缩短线段，使落点留在画布内。

    运动方向箭头原来长 1.6 个车身，车贴边时箭头尖会被裁掉，
    屏幕上只剩一根看不出朝向的线。宁可变短，也不能丢掉箭头。
    """
    (x0, y0), (x1, y1) = seg
    x0 = min(max(float(x0), margin), iw - margin)
    y0 = min(max(float(y0), margin), ih - margin)
    dx, dy = float(x1) - float(seg[0][0]), float(y1) - float(seg[0][1])
    t = 1.0
    for d, o, hi in ((dx, x0, iw), (dy, y0, ih)):
        if abs(d) < 1e-9:
            continue
        t = min(t, ((hi - margin) - o) / d if d > 0 else (margin - o) / d)
    t = max(0.0, min(1.0, t))
    return [[x0, y0], [x0 + dx * t, y0 + dy * t]]


def build_payload(name: str, inline_images: bool = False,
                  force_crops: bool = False, verbose: bool = True) -> dict:
    spec = C.get(name)
    items = read_queue(name)
    if not items:
        raise FileNotFoundError(
            f"{name} 还没有待标注队列。先跑： python -m src.select_frames --videos {name}")

    res = P.preprocess(spec)
    labels = read_labels(name)
    ww, wh = res.work_size
    out_items = []
    for qi in items:
        v = CR.ensure(res, spec, qi.k, qi.cx, qi.cy, qi.w, qi.h, force=force_crops)
        x1, y1, x2, y2 = qi.bbox()
        # 车框一旦越出画面，说明（至少）车身的一部分在画面外，标出来的轴会偏。
        # 选帧时已按"越界幅度占框长边 10%"放宽过，这里再把剩下的如实提示给标注者。
        cut = max(0.0, -x1, -y1, x2 - ww, y2 - wh) > 1.0
        hint_seg = CR.axis_segment(qi.cx, qi.cy, qi.hint, max(qi.w, qi.h) * 0.95)
        hint_disp = [list(v.work_to_disp(*p)) for p in hint_seg]
        vel_disp = None
        if qi.psi_vel is not None and np.isfinite(qi.psi_vel):
            # 箭头起于车心后方半个车身、指向运动方向、再走 0.9 个车身：
            # 整体落在裁图内（裁图半边长是 0.95 个车身），不会压住车身也裁不掉箭头
            size = max(qi.w, qi.h)
            seg = CR.vel_segment(qi.cx, qi.cy, qi.psi_vel, size * 0.9, tail=size * 0.5)
            seg = [v.work_to_disp(*p) for p in seg]
            if _finite(seg):
                vel_disp = fit_segment(seg, v.iw, v.ih)
        bx1, by1 = v.work_to_disp(x1, y1)
        bx2, by2 = v.work_to_disp(x2, y2)

        img = f"/img?name={name}&k={qi.k}"
        if inline_images:
            raw = v.path.read_bytes()
            img = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")

        saved = labels.get(qi.k)
        out_items.append(dict(
            k=qi.k, frame=qi.frame, t=round(qi.t, 3),
            w=round(qi.w, 1), h=round(qi.h, 1), conf=round(qi.conf, 3),
            hint=round(qi.hint, 2), cut=bool(cut),
            psi_vel=None if qi.psi_vel is None else round(qi.psi_vel, 2),
            speed_blps=None if qi.speed_blps is None else round(qi.speed_blps, 3),
            img=img,
            view=dict(ox=v.ox, oy=v.oy, sx=v.sx, sy=v.sy, iw=v.iw, ih=v.ih),
            bbox=[bx1, by1, bx2, by2],
            hint_seg=hint_disp if _finite(hint_disp) else None,
            vel_seg=vel_disp if (vel_disp and _finite(vel_disp)) else None,
            saved=None if not saved else dict(
                axis=round(float(saved["axis_deg"]), 2),
                src=saved.get("src", "manual"),
                pts=([[float(saved[c]) for c in ("x1", "y1")],
                      [float(saved[c]) for c in ("x2", "y2")]]
                     if all(saved.get(c) not in (None, "") for c in ("x1", "y1", "x2", "y2"))
                     else None),
            ),
        ))

    mats = []
    for s in C.annotation_videos():
        q = len(read_queue(s.name))
        lab = len(read_labels(s.name))
        mats.append(dict(name=s.name, queue=q, done=lab,
                         role=s.role, note=s.notes[:60]))
    pre = dict(name=name, work_size=list(res.work_size), src_size=list(res.src_size),
               crop=list(res.crop), n=res.n, eff_fps=round(res.eff_fps, 2),
               warned=bool(res.warnings))
    if verbose:
        print(f"  {name}: 队列 {len(items)} 帧，其中已标 {sum(1 for i in items if i.k in labels)} 帧")
    return dict(mode="server" if not inline_images else "standalone",
                name=name, pre=pre, mats=mats, items=out_items)


def status_report(names: list[str] | None = None) -> int:
    """进度 + 一致性体检：人工轴与 PCA 提示轴的差异，衡量提示到底有多少用。"""
    specs = [C.get(n) for n in names] if names else C.annotation_videos()
    print(f"{'素材':<9}{'队列':>6}{'已标':>6}{'完成':>7}{'PCA提示 vs 人工(中位|Δ|)':>26}  说明")
    print("-" * 88)
    tq = td = 0
    for s in specs:
        q = read_queue(s.name)
        lab = read_labels(s.name)
        tq += len(q)
        td += sum(1 for i in q if i.k in lab)
        diffs = []
        for i in q:
            if i.k in lab and np.isfinite(i.hint):
                diffs.append(abs(K.angle_diff(lab[i.k]["axis_deg"], i.hint)) % 180.0)
        d = f"{np.median(diffs):.1f}°" if diffs else "—"
        note = ""
        if not q:
            note = "需先跑 select_frames"
        elif diffs and np.median(diffs) > 30:
            note = "PCA 提示不可信，必须人工点选"
        elif diffs:
            note = "PCA 提示尚可，仍以人工为准"
        pc = f"{sum(1 for i in q if i.k in lab) / len(q) * 100:.0f}%" if q else "—"
        print(f"{s.name:<9}{len(q):>6}{sum(1 for i in q if i.k in lab):>6}{pc:>7}{d:>26}  {note}")
    print(f"\n合计 {td}/{tq} 帧已标。")
    print(f"标注文件在 {C.ANNOT_DIR.relative_to(C.ROOT)}/，下游由 kinematics.load_annotation_detail 读取。")
    return 0


# ---------------------------------------------------------------------------
# 页面（服务模式与自包含模式共用同一份前端）
# ---------------------------------------------------------------------------
PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>漂移姿态标注台</title>
<style>
  :root{
    --bg:#f6f6f4; --panel:#ffffff; --line:#e3e3df; --ink:#22221f; --dim:#6b6b64;
    --accent:#185FA5; --ok:#1D9E75; --warn:#B4770F; --bad:#E24B4A;
    --hint:#F2C230; --axis:#0E9AD8; --vel:#1D9E75;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{background:var(--bg);color:var(--ink);
    font:13px/1.5 -apple-system,"PingFang SC","Helvetica Neue",Arial,sans-serif}
  #app{display:flex;flex-direction:column;height:100vh}
  header{display:flex;align-items:center;gap:14px;padding:9px 14px;background:var(--panel);
    border-bottom:1px solid var(--line);flex:0 0 auto}
  .brand{font-weight:600;letter-spacing:.02em;white-space:nowrap}
  .mats{display:flex;gap:6px;flex-wrap:wrap}
  .mats a{padding:3px 9px;border:1px solid var(--line);border-radius:99px;text-decoration:none;
    color:var(--dim);font-size:12px;white-space:nowrap}
  .mats a.on{background:var(--accent);border-color:var(--accent);color:#fff}
  .mats a .c{opacity:.75;font-variant-numeric:tabular-nums}
  .spacer{flex:1}
  .prog{display:flex;align-items:center;gap:8px;min-width:190px}
  .bar{flex:1;height:7px;background:#e9e9e5;border-radius:99px;overflow:hidden}
  .bar i{display:block;height:100%;width:0;background:var(--ok);transition:width .18s}
  .prog span{color:var(--dim);font-variant-numeric:tabular-nums;white-space:nowrap}
  .layout{display:flex;flex:1;min-height:0}
  .stage{flex:1;min-width:0;display:flex;flex-direction:column}
  .cvwrap{flex:1;overflow:auto;padding:12px;display:flex;align-items:flex-start;
    justify-content:center;position:relative}
  .stack{position:relative;flex:0 0 auto}
  canvas#cv{display:block;border:1px solid var(--line);border-radius:6px;background:#fff;
    cursor:crosshair;box-shadow:0 1px 3px rgba(0,0,0,.06)}
  canvas#loupe{position:absolute;top:20px;right:20px;width:168px;height:168px;
    border:2px solid #fff;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.28);
    background:#000;display:none;pointer-events:none}
  .nav{display:flex;gap:8px;padding:9px 12px;border-top:1px solid var(--line);
    background:var(--panel);flex:0 0 auto;flex-wrap:wrap;align-items:center}
  button{font:inherit;padding:6px 12px;border:1px solid var(--line);background:#fff;
    border-radius:6px;cursor:pointer;color:var(--ink)}
  button:hover{background:#f2f2ee}
  button:disabled{opacity:.45;cursor:not-allowed}
  button.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
  button.primary:hover{filter:brightness(1.08)}
  kbd{font:11px/1 ui-monospace,Menlo,monospace;background:#f0f0ec;border:1px solid var(--line);
    border-bottom-width:2px;border-radius:4px;padding:2px 5px;color:var(--dim)}
  aside{width:286px;flex:0 0 auto;border-left:1px solid var(--line);background:var(--panel);
    overflow:auto;padding:12px}
  .card{border:1px solid var(--line);border-radius:8px;padding:10px 11px;margin-bottom:10px}
  .card h3{margin:0 0 8px;font-size:12px;color:var(--dim);font-weight:600;
    letter-spacing:.04em;text-transform:uppercase}
  .kv{display:grid;grid-template-columns:auto 1fr;gap:3px 10px;font-variant-numeric:tabular-nums}
  .kv b{font-weight:500;color:var(--dim)}
  .kv span{text-align:right}
  .big{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}
  .axis-box{display:flex;align-items:baseline;gap:8px;margin:2px 0 6px}
  .axis-box .big{color:var(--axis)}
  .beta{font-size:15px;font-weight:600}
  .muted{color:var(--dim);font-size:12px}
  .warnrow{display:none;margin-top:8px;padding:6px 9px;border-radius:6px;
    background:#fdf3e0;color:#8a5a06;font-size:12px;line-height:1.45}
  .warnrow b{font-weight:600}
  .list{max-height:44vh;overflow:auto;margin:-4px -4px -4px}
  .row{display:flex;justify-content:space-between;gap:8px;padding:5px 7px;border-radius:5px;
    cursor:pointer;font-variant-numeric:tabular-nums}
  .row:hover{background:#f3f3ef}
  .row.here{background:#eaf2fb}
  .row .src{font-size:11px;color:var(--dim)}
  .row.hint .src{color:var(--warn)}
  .legend span{display:inline-flex;align-items:center;gap:5px;margin-right:12px;
    font-size:12px;color:var(--dim)}
  .legend i{width:14px;height:0;border-top-width:3px;border-top-style:solid;border-radius:2px}
  #toast{position:fixed;left:50%;bottom:22px;transform:translate(-50%,20px);opacity:0;
    background:#22221f;color:#fff;padding:8px 16px;border-radius:99px;font-size:12.5px;
    pointer-events:none;transition:.22s;z-index:50}
  #toast.on{opacity:1;transform:translate(-50%,0)}
  #toast.warn{background:#8a5a06}
  #toast.bad{background:#a32320}
  .export{display:none}
  .export textarea{width:100%;height:120px;font:11px/1.4 ui-monospace,Menlo,monospace;
    border:1px solid var(--line);border-radius:6px;padding:6px;background:#fbfbf9}
</style>
</head>
<body>
<div id="app">
  <header>
    <div class="brand">漂移姿态标注台</div>
    <div class="mats" id="mats"></div>
    <div class="spacer"></div>
    <div class="prog"><div class="bar"><i id="fill"></i></div><span id="ptxt">—</span></div>
  </header>

  <div class="layout">
    <div class="stage">
      <div class="cvwrap">
        <div class="stack">
          <canvas id="cv"></canvas>
          <canvas id="loupe" width="336" height="336"></canvas>
        </div>
      </div>
      <div class="nav">
        <button id="bPrev">← 上一帧 <kbd>P</kbd></button>
        <button id="bSkip">跳过 <kbd>N</kbd></button>
        <button id="bNext">下一未标 <kbd>G</kbd></button>
        <button id="bHint">用提示轴 <kbd>T</kbd></button>
        <button id="bClear">清除 <kbd>R</kbd></button>
        <button id="bUndo">撤销本帧标注 <kbd>U</kbd></button>
        <div class="spacer" style="flex:1"></div>
        <button id="bSave" class="primary">保存并下一帧 <kbd>Enter</kbd></button>
      </div>
    </div>

    <aside>
      <div class="card">
        <h3>当前帧</h3>
        <div class="axis-box">
          <span class="big" id="angNow">—</span>
          <span class="muted">车身轴 / 度</span>
        </div>
        <div class="kv">
          <b>队列序号</b><span id="oIdx">—</span>
          <b>k / 原始帧</b><span id="oK">—</span>
          <b>时间</b><span id="oTime">—</span>
          <b>检测置信度</b><span id="oConf">—</span>
          <b>车框</b><span id="oBox">—</span>
          <b>速度</b><span id="oSpeed">—</span>
          <b>ψ_vel 运动方向</b><span id="oVel">—</span>
          <b>提示轴 (PCA)</b><span id="oHint">—</span>
          <b>推出的 β</b><span class="beta" id="oBeta">—</span>
        </div>
        <div class="muted" style="margin-top:8px">
          沿车身点两点即可。红框=检测框，黄虚线=PCA 提示（需人工确认），
          绿箭头=运动方向。
        </div>
        <div class="warnrow" id="oCut">
          <b>车身被画面截断</b>（检测框越出边界）。这一帧标出来的轴会偏，
          看不清就按 <kbd>N</kbd> 跳过。
        </div>
      </div>

      <div class="card export" id="exportCard">
        <h3>导出</h3>
        <div class="muted" id="exportTip">标注已存在浏览器本地。点下面按钮下载 CSV，</div>
        <div class="nav" style="border:0;padding:8px 0 0;background:none">
          <button id="bDl" class="primary">导出 CSV</button>
          <button id="bCopy">复制</button>
        </div>
        <textarea id="csvOut" style="margin-top:8px" readonly></textarea>
        <div class="muted" id="exportPath" style="margin-top:6px"></div>
      </div>

      <div class="card">
        <h3>已标注 <span id="nDone" class="muted"></span></h3>
        <div class="list" id="doneList"><div class="muted">尚无</div></div>
      </div>

      <div class="card">
        <h3>图例</h3>
        <div class="legend">
          <span><i style="border-color:var(--axis)"></i>你的轴</span>
          <span><i style="border-color:var(--hint);border-top-style:dashed"></i>PCA 提示</span>
          <span><i style="border-color:var(--vel)"></i>运动方向</span>
          <span><i style="border-color:var(--bad)"></i>检测框</span>
        </div>
        <div class="muted" style="margin-top:6px">
          <kbd>T</kbd> 采用提示轴（记为 hint，可与人工点选区分）<br>
          <kbd>R</kbd> 清除本帧点选 · <kbd>U</kbd> 撤销已保存<br>
          鼠标拖动或两次单击都能画线；右上角是 3× 放大镜。
        </div>
      </div>
    </aside>
  </div>
  <div id="toast"></div>
</div>

<script>
const P = /*__PAYLOAD__*/;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const lz = document.getElementById('loupe'), lx = lz.getContext('2d');
let idx = 0, pts = [], saved = {}, cache = {}, dragging = false, moved = 0;
let lastSaved = null;      // 本会话最后保存的 k，供"撤销"在自动跳帧后回退

// ---------- 角度工具（与 kinematics.py 完全一致，便于互验） ----------
const mod180 = d => ((d % 180) + 180) % 180;
const wrap360 = d => ((d % 360) + 360) % 360;
// 显示坐标 <-> 工作图坐标（工作图才是角度的定义域）
function workToDisp(p){
  const v = cur().view;
  return [(p[0] - v.ox) * v.sx, (p[1] - v.oy) * v.sy];
}
function dispToWork(p){
  const v = cur().view;
  return [p[0] / v.sx + v.ox, p[1] / v.sy + v.oy];
}
function angleDiff(a, b){ let d = (a - b + 180) % 360 - 180; return d; }
function axisFromPts(a, b){
  // 先换算回工作图再算角：裁图在两个方向上的缩放比不严格相等，
  // 直接在显示坐标里算会引入一点点各向异性误差。
  const A = dispToWork(a), B = dispToWork(b);
  return mod180(Math.atan2(B[1] - A[1], B[0] - A[0]) * 180 / Math.PI);
}
function resolveHead(axis){
  const pv = cur().psi_vel;
  if (pv === null || pv === undefined) return null;
  const a = mod180(axis), cands = [a, a + 180];
  let best = cands[0], bd = 1e9;
  for (const c of cands){
    const d = Math.abs(angleDiff(c, wrap360(pv)));
    if (d < bd){ bd = d; best = c; }
  }
  return best;
}
function betaOf(axis){
  const head = resolveHead(axis), pv = cur().psi_vel;
  if (head === null || pv === null || pv === undefined) return null;
  let b = angleDiff(head, pv);
  if (Math.abs(b) > 90) b -= Math.sign(b) * 180;
  return b;
}

const cur = () => P.items[idx];
const fmt = (v, n = 1) => (v === null || v === undefined || !isFinite(v)) ? '—' : Number(v).toFixed(n);

// ---------- 绘制 ----------
function draw(){
  const it = cur(); if (!it) return;
  const img = cache[it.img];
  const v = it.view;
  cv.width = v.iw; cv.height = v.ih;
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (img){
    ctx.drawImage(img, 0, 0, v.iw, v.ih);
  } else {
    ctx.fillStyle = '#eee'; ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = '#888'; ctx.font = '16px sans-serif';
    ctx.fillText('图像加载中…', 20, 34);
  }

  // 检测框
  ctx.lineWidth = 1.6; ctx.setLineDash([]);
  ctx.strokeStyle = '#E24B4A';
  ctx.strokeRect(it.bbox[0], it.bbox[1], it.bbox[2] - it.bbox[0], it.bbox[3] - it.bbox[1]);

  // 运动方向箭头
  if (it.vel_seg){
    arrow(it.vel_seg[0], it.vel_seg[1], '#1D9E75', 1.8, 9);
  }
  // PCA 提示轴。用 line3 延长成贯穿画布的一条线，和"你的轴"画法一致 ——
  // 贴边帧的轴线端点会落在裁图外，不延长的话会被裁短、看不出完整角度。
  if (it.hint_seg){
    ctx.setLineDash([7, 5]); ctx.strokeStyle = '#F2C230'; ctx.lineWidth = 1.6;
    line3(it.hint_seg[0], it.hint_seg[1], 0.30);
    ctx.setLineDash([]);
  }
  // 已保存的轴（未编辑时）
  const sv = saved[it.k];
  if (sv && !pts.length){
    const p = sv.pts || segFromAxis(sv.axis);
    ctx.setLineDash([]); ctx.strokeStyle = 'rgba(14,154,216,.95)'; ctx.lineWidth = 2.4;
    line3(p[0], p[1], 0.30);
  }
  // 正在点的两个点
  if (pts.length){
    ctx.setLineDash([]); ctx.strokeStyle = '#0E9AD8'; ctx.lineWidth = 2.4;
    if (pts.length === 2) line3(pts[0], pts[1], 0.30);
    for (let i = 0; i < pts.length; i++){
      ctx.beginPath(); ctx.arc(pts[i][0], pts[i][1], 5, 0, 7);
      ctx.fillStyle = i ? '#0E9AD8' : '#ffffff'; ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = '#0E9AD8'; ctx.stroke();
    }
  }
  readout();
  progress();
}

function line(a, b){ ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
// 把线段延长到裁图边界，读起来像一条"轴线"而不是两根点的连线
function line3(a, b, ext){
  const v = cur().view, dx = b[0] - a[0], dy = b[1] - a[1];
  const L = Math.hypot(dx, dy) || 1;
  const e = Math.max(v.iw, v.ih) * ext;
  line([a[0] - dx / L * e, a[1] - dy / L * e], [b[0] + dx / L * e, b[1] + dy / L * e]);
}
function arrow(a, b, col, w, head){
  ctx.setLineDash([]); ctx.strokeStyle = col; ctx.lineWidth = w;
  line(a, b);
  const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
  ctx.beginPath();
  ctx.moveTo(b[0], b[1]);
  ctx.lineTo(b[0] - head * Math.cos(ang - 0.42), b[1] - head * Math.sin(ang - 0.42));
  ctx.lineTo(b[0] - head * Math.cos(ang + 0.42), b[1] - head * Math.sin(ang + 0.42));
  ctx.closePath(); ctx.fillStyle = col; ctx.fill();
}
// 由轴角推一条穿过质心的显示线段（用于重画已保存标注）
function segFromAxis(axis){
  const it = cur(), v = it.view;
  const cx = (it.bbox[0] + it.bbox[2]) / 2, cy = (it.bbox[1] + it.bbox[3]) / 2;
  const L = Math.max(it.bbox[2] - it.bbox[0], it.bbox[3] - it.bbox[1]);
  const a = axis * Math.PI / 180;
  return [[cx - Math.cos(a) * L, cy - Math.sin(a) * L],
          [cx + Math.cos(a) * L, cy + Math.sin(a) * L]];
}

// ---------- 读数 ----------
function readout(){
  const it = cur(); if (!it) return;
  const a = (pts.length === 2) ? axisFromPts(pts[0], pts[1]) : (saved[it.k] ? saved[it.k].axis : null);
  document.getElementById('angNow').textContent = a === null ? '—' : a.toFixed(1) + '°';
  const b = (a === null) ? null : betaOf(a);
  const be = document.getElementById('oBeta');
  be.textContent = b === null ? '—' : (b >= 0 ? '+' : '') + b.toFixed(1) + '°';
  be.style.color = b === null ? 'var(--dim)' : (Math.abs(b) > 45 ? 'var(--warn)' : 'var(--ink)');
  document.getElementById('oIdx').textContent = (idx + 1) + ' / ' + P.items.length;
  document.getElementById('oK').textContent = it.k + ' / ' + it.frame;
  document.getElementById('oTime').textContent = it.t.toFixed(2) + ' s';
  document.getElementById('oConf').textContent = fmt(it.conf, 2);
  document.getElementById('oBox').textContent = fmt(it.w, 0) + '×' + fmt(it.h, 0) + ' px';
  document.getElementById('oSpeed').textContent = it.speed_blps === null ? '—'
    : fmt(it.speed_blps, 2) + ' 车身长/秒';
  document.getElementById('oVel').textContent = it.psi_vel === null ? '—' : fmt(it.psi_vel, 1) + '°';
  document.getElementById('oHint').textContent = fmt(it.hint, 1) + '°';
  document.getElementById('oCut').style.display = it.cut ? 'block' : 'none';
}

function progress(){
  const done = P.items.filter(i => saved[i.k]).length;
  document.getElementById('fill').style.width = (done / P.items.length * 100) + '%';
  document.getElementById('ptxt').textContent = done + ' / ' + P.items.length;
  document.getElementById('nDone').textContent = '（' + done + '）';
  const list = document.getElementById('doneList');
  const ks = P.items.filter(i => saved[i.k]);
  if (!ks.length){ list.innerHTML = '<div class="muted">尚无</div>'; return; }
  list.innerHTML = ks.map(i => {
    const s = saved[i.k];
    return '<div class="row' + (s.src === 'hint' ? ' hint' : '') +
      (P.items[idx] && P.items[idx].k === i.k ? ' here' : '') +
      '" data-k="' + i.k + '"><span>k=' + i.k + '</span><span>' +
      s.axis.toFixed(1) + '° <span class="src">' + s.src + '</span></span></div>';
  }).join('');
  list.querySelectorAll('.row').forEach(el => el.onclick = () => {
    const j = P.items.findIndex(x => String(x.k) === el.dataset.k);
    if (j >= 0) go(j);
  });
}

// ---------- 导航 ----------
// 图片按 src 缓存；undefined=未请求，null=加载中，Image=已就绪
function ensureImg(src){
  if (cache[src] !== undefined) return;
  cache[src] = null;
  const im = new Image();
  im.onload = () => { cache[src] = im; if (cur() && cur().img === src) draw(); };
  im.src = src;
}
function go(j){
  idx = Math.max(0, Math.min(P.items.length - 1, j));
  pts = [];
  ensureImg(cur().img);
  draw();
}
function nextUnannotated(from){
  for (let s = 1; s <= P.items.length; s++){
    const j = (from + s) % P.items.length;
    if (!saved[P.items[j].k]) return j;
  }
  return -1;
}
function flash(msg, kind){
  const t = document.getElementById('toast');
  t.textContent = msg; t.className = 'on' + (kind ? ' ' + kind : '');
  clearTimeout(flash._t); flash._t = setTimeout(() => t.className = '', 1600);
}

// ---------- 保存 ----------
async function commit(rec){
  if (P.mode === 'server'){
    const r = await fetch('/api/save', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(Object.assign({name: P.name}, rec)),
    });
    if (!r.ok) throw new Error('保存失败 HTTP ' + r.status);
    return await r.json();
  }
  // 独立模式：ts 也要一起存，否则导出的 CSV 缺时间戳，无法追溯标注顺序
  saved[rec.k] = {axis: rec.axis_deg, src: rec.src, pts: rec.pts, ts: rec.ts};
  persistLocal();
  return {ok: true};
}
function saveAndNext(){
  const it = cur();
  if (pts.length !== 2){ flash('先沿车身点两个点', 'warn'); return; }
  const w1 = dispToWork(pts[0]), w2 = dispToWork(pts[1]);
  const rec = {k: it.k, axis_deg: axisFromPts(pts[0], pts[1]), src: 'manual',
    x1: w1[0], y1: w1[1], x2: w2[0], y2: w2[1],
    pts: [w1, w2], ts: new Date().toISOString()};
  commit(rec).then(res => {
    saved[it.k] = {axis: rec.axis_deg, src: 'manual', pts: rec.pts, ts: rec.ts};
    lastSaved = it.k;
    flash('已保存 k=' + it.k + '  β=' + (betaOf(rec.axis_deg) ?? 0).toFixed(1) + '°');
    const j = nextUnannotated(idx);
    if (j >= 0 && j !== idx) go(j); else { pts = []; draw(); }
    if (res && res.stats) refreshStats(res.stats);
  }).catch(e => flash(e.message, 'bad'));
}
function useHint(){
  const it = cur();
  if (it.hint_seg === null){ flash('该帧没有提示轴', 'warn'); return; }
  const w = it.hint_seg.map(q => dispToWork(q));
  const rec = {
    k: it.k, axis_deg: it.hint, src: 'hint',
    x1: w[0][0], y1: w[0][1], x2: w[1][0], y2: w[1][1],
    pts: w, ts: new Date().toISOString(),
  };
  commit(rec).then(() => {
    saved[it.k] = {axis: rec.axis_deg, src: 'hint', pts: rec.pts, ts: rec.ts};
    lastSaved = it.k;
    flash('已按提示轴记录 k=' + it.k + '（src=hint）');
    const j = nextUnannotated(idx);
    if (j >= 0 && j !== idx) go(j); else { pts = []; draw(); }
  }).catch(e => flash(e.message, 'bad'));
}
// 保存后会自动跳到下一未标帧，所以"撤销"不能只看当前帧：
// 当前帧没有标注时，回退到本会话最后保存的那一帧去撤销（并跳过去）。
function undo(){
  const here = cur();
  const tk = saved[here.k] ? here.k
    : ((lastSaved !== null && saved[lastSaved]) ? lastSaved : null);
  if (tk === null){ flash('没有可撤销的标注', 'warn'); return; }
  const finish = () => {
    delete saved[tk];
    if (lastSaved === tk) lastSaved = null;
    // 独立模式必须回写本地存储：否则撤销掉的标注在重开页面时又"复活"
    if (P.mode !== 'server') persistLocal();
    if (tk === here.k){ pts = []; draw(); }
    else { const j = P.items.findIndex(x => x.k === tk); if (j >= 0) go(j); }
    flash('已撤销 k=' + tk);
  };
  if (P.mode === 'server'){
    fetch('/api/delete', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: P.name, k: tk})})
      .then(r => r.ok ? finish() : flash('撤销失败', 'bad'))
      .catch(e => flash(e.message, 'bad'));
  } else { finish(); }
}
function refreshStats(st){
  if (!st.mats) return;
  const box = document.getElementById('mats');
  box.querySelectorAll('a').forEach(a => {
    const m = st.mats.find(x => x.name === a.dataset.n);
    if (m) a.querySelector('.c').textContent = m.done + '/' + m.queue;
  });
}

// ---------- 自包含模式的本地持久化 ----------
const LS_KEY = () => 'drift_annot_' + P.name;
function persistLocal(){
  try { localStorage.setItem(LS_KEY(), JSON.stringify(saved)); } catch (e) {}
}
function loadLocal(){
  // 先铺"磁盘上已有的标注"——导出时它们已被嵌进载荷，是权威来源；
  // 再用浏览器里的本地改动覆盖同 k 的条目（那些还没导出到磁盘，更新）。
  // 少了第一步，重新导出的页面就看不到之前标的帧，还会把它们当成"未标注"，
  // 保存时把已有标注覆盖掉。
  P.items.forEach(i => { if (i.saved) saved[i.k] = i.saved; });
  try {
    const s = JSON.parse(localStorage.getItem(LS_KEY()) || '{}');
    if (s && typeof s === 'object') Object.assign(saved, s);
  } catch (e) {}
}
function csvText(){
  const L = ['# ' + P.name + ' 人工标注的车身轴（mod 180 度）',
    '# axis_deg 是唯一被下游读取的字段；x1..y2 是工作图坐标下的原始点选，供复核',
    '# src: manual=人工点选 / hint=人工确认了 PCA 提示轴',
    'k,axis_deg,x1,y1,x2,y2,src,ts'];
  Object.keys(saved).map(Number).sort((a, b) => a - b).forEach(k => {
    const s = saved[k], p = s.pts || [[null, null], [null, null]];
    const q = z => (z === null || z === undefined) ? '' : Number(z).toFixed(2);
    L.push([k, Number(s.axis).toFixed(3), q(p[0][0]), q(p[0][1]), q(p[1][0]), q(p[1][1]),
      s.src || 'manual', s.ts || ''].join(','));
  });
  return L.join('\n') + '\n';
}
function download(){
  const blob = new Blob([csvText()], {type: 'text/csv;charset=utf-8'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = P.name + '.csv';
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  flash('已下载 ' + P.name + '.csv');
}

// ---------- 鼠标 ----------
function cpos(e){
  const r = cv.getBoundingClientRect();
  return [(e.clientX - r.left) * (cv.width / r.width),
          (e.clientY - r.top) * (cv.height / r.height)];
}
function loupeAt(p){
  const v = cur().view, img = cache[cur().img];
  lz.style.display = 'block';
  lx.fillStyle = '#111'; lx.fillRect(0, 0, lz.width, lz.height);
  if (!img) return;
  const Z = 3, src = lz.width / Z;
  lx.imageSmoothingEnabled = false;
  lx.drawImage(img, p[0] - src / 2, p[1] - src / 2, src, src, 0, 0, lz.width, lz.height);
  lx.imageSmoothingEnabled = true;
  lx.strokeStyle = 'rgba(255,60,60,.9)'; lx.lineWidth = 1;
  lx.beginPath(); lx.moveTo(lz.width / 2, 0); lx.lineTo(lz.width / 2, lz.height);
  lx.moveTo(0, lz.height / 2); lx.lineTo(lz.width, lz.height / 2); lx.stroke();
}
cv.addEventListener('pointerdown', e => {
  // 某些 webview / 合成事件下 setPointerCapture 会抛 NotFoundError，
  // 但它只是"鼠标移出画布也继续追踪"的便利功能，失败不该影响标注。
  try { cv.setPointerCapture(e.pointerId); } catch (err) {}
  const p = cpos(e);
  if (pts.length === 2) pts = [];
  dragging = true; moved = 0;
  pts.push(p);
  draw();
});
cv.addEventListener('pointermove', e => {
  const p = cpos(e);
  loupeAt(p);
  if (dragging && pts.length){
    const a = pts[pts.length - 1];
    moved = Math.hypot(p[0] - a[0], p[1] - a[1]);
    if (moved > 3){
      if (pts.length === 1) pts.push(p); else pts[1] = p;
      draw();
    }
  }
});
cv.addEventListener('pointerup', e => {
  dragging = false;
  if (pts.length === 2 && moved <= 3){ /* 两次单击模式：保留两点 */ }
  draw();
});
cv.addEventListener('pointerleave', () => { lz.style.display = 'none'; });
cv.addEventListener('dblclick', () => { lz.style.display = 'none'; });

// ---------- 键盘 ----------
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  if (k === 'enter'){ e.preventDefault(); saveAndNext(); }
  else if (k === 'n' || k === 'arrowright'){ pts = []; go(idx + 1); }
  else if (k === 'p' || k === 'arrowleft'){ pts = []; go(idx - 1); }
  else if (k === 'g'){ const j = nextUnannotated(idx); if (j >= 0) go(j); else flash('全部已标注'); }
  else if (k === 'r'){ pts = []; draw(); }
  else if (k === 't'){ useHint(); }
  else if (k === 'u'){ undo(); }
  else if (k === 'escape'){ pts = []; draw(); }
});

// ---------- 初始化 ----------
function init(){
  if (P.mode === 'standalone'){
    loadLocal();
    document.getElementById('exportCard').style.display = 'block';
    document.getElementById('exportTip').textContent =
      '标注存在浏览器本地（localStorage），关掉页面也不会丢。';
    document.getElementById('exportPath').innerHTML =
      '把下载的 CSV 放到 <code>outputs/annotations/' + P.name + '.csv</code> 即可被下游读取。';
    document.getElementById('bDl').onclick = download;
    document.getElementById('bCopy').onclick = () => {
      const t = document.getElementById('csvOut');
      t.value = csvText(); t.select();
      navigator.clipboard ? navigator.clipboard.writeText(t.value)
        .then(() => flash('已复制到剪贴板')).catch(() => flash('复制失败，请手动选中', 'warn'))
        : flash('请手动选中复制', 'warn');
    };
  }
  // 素材切换
  const box = document.getElementById('mats');
  P.mats.forEach(m => {
    const a = document.createElement('a');
    a.dataset.n = m.name;
    a.className = m.name === P.name ? 'on' : '';
    a.innerHTML = m.name + ' <span class="c">' + (m.queue ? m.done + '/' + m.queue : '未建队列') + '</span>';
    a.href = P.mode === 'server' ? ('/?name=' + m.name) : '#';
    if (P.mode === 'standalone' && m.name !== P.name){
      a.onclick = ev => { ev.preventDefault(); flash('自包含模式一次只含一段素材'); };
    }
    box.appendChild(a);
  });
  document.getElementById('bPrev').onclick = () => { pts = []; go(idx - 1); };
  document.getElementById('bSkip').onclick = () => { pts = []; go(idx + 1); };
  document.getElementById('bNext').onclick = () => { const j = nextUnannotated(idx); if (j >= 0) go(j); else flash('全部已标注'); };
  document.getElementById('bHint').onclick = useHint;
  document.getElementById('bClear').onclick = () => { pts = []; draw(); };
  document.getElementById('bUndo').onclick = undo;
  document.getElementById('bSave').onclick = saveAndNext;

  // 已有标注回填（服务模式由载荷带出；自包含模式已在 loadLocal 里）
  if (P.mode === 'server'){
    P.items.forEach(i => { if (i.saved) saved[i.k] = i.saved; });
  }
  const start = nextUnannotated(-1);
  go(start >= 0 ? start : 0);
  flash('共 ' + P.items.length + ' 帧待标注');
}
init();
</script>
</body>
</html>
"""


def render_page(payload: dict) -> str:
    js = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return PAGE.replace("/*__PAYLOAD__*/", js)


# ---------------------------------------------------------------------------
# 服务模式
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    names: list[str] = []
    default_name: str = ""

    def log_message(self, *a):        # 静音访问日志，只在服务端打印启动信息
        pass

    # -- 工具 --
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _name(self, q: dict) -> str:
        n = (q.get("name") or [""])[0]
        if n in C.VIDEOS:
            return n
        return self.default_name

    # -- 路由 --
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            name = self._name(q)
            try:
                page = render_page(build_payload(name, verbose=False))
            except FileNotFoundError as e:
                page = ("<html><meta charset='utf-8'><body style='font:14px/1.7 sans-serif;"
                        "padding:40px;max-width:640px'><h2>还没有待标注队列</h2>"
                        f"<p>{e}</p><pre style='background:#f4f4f0;padding:12px;border-radius:6px'>"
                        f"python -m src.select_frames --videos {name}</pre></body></html>")
            return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        if u.path == "/api/payload":
            try:
                return self._json(build_payload(self._name(q), verbose=False))
            except FileNotFoundError as e:
                return self._json({"error": str(e)}, 404)
        if u.path == "/img":
            name = self._name(q)
            try:
                k = int((q.get("k") or ["-1"])[0])
            except ValueError:
                return self._send(400, b"bad k", "text/plain")
            p = CR.crop_path(name, k)
            if not p.exists():
                return self._send(404, b"no crop", "text/plain")
            return self._send(200, p.read_bytes(), "image/jpeg")
        if u.path == "/api/stats":
            mats = [dict(name=s.name, queue=len(read_queue(s.name)),
                         done=len(read_labels(s.name))) for s in C.annotation_videos()]
            return self._json(dict(mode="server", name=self.default_name,
                                   pre=dict(name=self.default_name), mats=mats, items=[]))
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        try:
            d = self._body()
        except (json.JSONDecodeError, ValueError):
            return self._json({"error": "bad json"}, 400)
        name = d.get("name")
        if name not in C.VIDEOS:
            return self._json({"error": f"未登记的素材：{name}"}, 400)

        if u.path == "/api/save":
            try:
                k = int(d["k"])
                axis = float(d["axis_deg"]) % C.BODY_AXIS_MOD
            except (KeyError, TypeError, ValueError):
                return self._json({"error": "缺少 k 或 axis_deg"}, 400)
            rec = dict(k=k, axis_deg=axis, src=str(d.get("src", "manual")),
                       ts=str(d.get("ts") or time.strftime("%Y-%m-%dT%H:%M:%S")))
            for c in ("x1", "y1", "x2", "y2"):
                v = d.get(c)
                if v is not None and v != "":
                    try:
                        rec[c] = float(v)
                    except (TypeError, ValueError):
                        pass
            upsert_label(name, rec)
            return self._json(dict(ok=True, k=k, axis_deg=axis,
                                   stats=self._stats()))

        if u.path == "/api/delete":
            try:
                k = int(d["k"])
            except (KeyError, TypeError, ValueError):
                return self._json({"error": "缺少 k"}, 400)
            drop_label(name, k)
            return self._json(dict(ok=True, k=k, stats=self._stats()))

        return self._json({"error": "unknown endpoint"}, 404)

    def _stats(self) -> dict:
        return dict(mats=[dict(name=s.name, queue=len(read_queue(s.name)),
                              done=len(read_labels(s.name)))
                          for s in C.annotation_videos()])


def serve(name: str | None = None, port: int = 8765, open_browser: bool = True) -> int:
    specs = C.annotation_videos()
    if not specs:
        print("没有登记为可标注的素材。")
        return 1
    names = [s.name for s in specs]
    if name is None:
        # 默认选第一段"有队列且还没标完"的素材
        name = next((n for n in names
                     if read_queue(n) and len(read_labels(n)) < len(read_queue(n))),
                    next((n for n in names if read_queue(n)), names[0]))
    if name not in names:
        print(f"{name} 不在可标注列表 {names}")
        return 1
    if not read_queue(name):
        print(f"{name} 还没有待标注队列，先生成：")
        print(f"  python -m src.select_frames --videos {name}")
        return 1

    _Handler.names = names
    _Handler.default_name = name

    srv = None
    for p in range(port, port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), _Handler)
            port = p
            break
        except OSError:
            continue
    if srv is None:
        print(f"端口 {port}~{port + 19} 都被占用，换个 --port。")
        return 1

    url = f"http://127.0.0.1:{port}/?name={name}"
    print("=" * 62)
    print("  漂移姿态标注台已启动")
    print(f"  地址： {url}")
    print(f"  素材： {name}（{len(read_queue(name))} 帧，已标 {len(read_labels(name))} 帧）")
    print(f"  另可切换： {'、'.join(n for n in names if n != name)}")
    print(f"  标注落盘： {labels_path(name).relative_to(C.ROOT)}")
    print("  停止： Ctrl-C")
    print("=" * 62)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。标注已保存在 outputs/annotations/ 下。")
    finally:
        srv.server_close()
    return 0


# ---------------------------------------------------------------------------
# 自包含导出
# ---------------------------------------------------------------------------
def export(name: str, force_crops: bool = False) -> Path:
    payload = build_payload(name, inline_images=True, force_crops=force_crops)
    html = render_page(payload)
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    out = WEB_DIR / f"{name}.html"
    out.write_text(html, encoding="utf-8")
    n = len(payload["items"])
    done = sum(1 for i in payload["items"] if i["saved"])
    print(f"  {name}: 导出 {out.relative_to(C.ROOT)}（{n} 帧，已标 {done}，"
          f"{out.stat().st_size / 1e6:.1f} MB）")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="车身轴标注台（浏览器点两下）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  python -m src.annotate --serve --name video02\n"
               "  python -m src.annotate --export --all\n"
               "  python -m src.annotate --status\n")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--serve", action="store_true", help="起本地服务，边点边存（推荐）")
    g.add_argument("--export", action="store_true", help="导出自包含 HTML（离线可用）")
    g.add_argument("--status", action="store_true", help="只看进度与一致性体检")
    ap.add_argument("--name", help="素材名（--serve/--export 用；--serve 可省略）")
    ap.add_argument("--all", action="store_true", help="--export 时导出所有可标注素材")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    ap.add_argument("--force", action="store_true", help="忽略裁图与预处理缓存")
    args = ap.parse_args(argv)

    if args.status:
        return status_report([args.name] if args.name else None)
    if args.serve:
        return serve(args.name, port=args.port, open_browser=not args.no_open)

    names = [args.name] if args.name else ([s.name for s in C.annotation_videos()] if args.all else None)
    if not names:
        print("--export 需要 --name 或 --all")
        return 2
    ok = 0
    for n in names:
        try:
            export(n, force_crops=args.force)
            ok += 1
        except FileNotFoundError as e:
            print(f"  {n}: 跳过 —— {e}")
    if ok:
        print(f"\n用浏览器打开 outputs/annotations/web/<素材名>.html 即可标注（无需服务）。")
        print("标注后点「导出 CSV」，把文件放到 outputs/annotations/<素材名>.csv。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
