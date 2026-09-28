"""核验「裁图 ↔ 工作图」坐标换算是否真的对得上。

为什么不靠"看代码觉得对"：这一层出错的**症状是坐标整体平移，角度完全正常**，
所以下游的角度指标、标注残差、QA 全部不会报警 —— 只能被**针对性断言**抓到。

**主断言：模板匹配反解映射。**
取工作图上以检测质心为中心的 25×25 小片，放大到裁图的显示比例，
再在裁图里用 `matchTemplate` 找它**实际**在哪，与 `work_to_disp(cx, cy)` 声称的位置比。
两者之差就是映射误差（显示像素）。这条断言与"裁图有没有被夹边"无关 ——
夹边只让裁图不居中，映射本身仍然精确 —— 所以它是干净的。

辅助断言（便宜、能抓粗错）：`disp_to_work(work_to_disp(p)) == p`。
注意它是**恒等式**，任何 `ox/oy` 都满足，所以只能当"没写错符号"的烟雾测试，
不能单独当证据。

⚠ 曾经用过一条"质心应落在裁图中心"的断言，**它是错的**：`crop_rect_for` 会把
矩形夹到画面内（车贴边时），夹完就不居中了。本文件因此只统计"未夹边帧"的居中偏差。

用法：
    /opt/anaconda3/bin/python3 tools/verify_crops.py          # 只核验（渲染到临时目录）
    /opt/anaconda3/bin/python3 tools/verify_crops.py --stale  # 顺便列出几何过期的裁图缓存
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C        # noqa: E402
from src import crops as CR        # noqa: E402
from src import preprocess as P    # noqa: E402

PATCH = 25          # 工作图上的取样小片边长（px）
TOL_PX = 4.0        # 映射误差容差（显示像素）
MIN_SCORE = 0.55    # 模板匹配可信度下限，低了就跳过（沥青等低纹理区匹配不可靠）
MAX_FRAMES = 6      # 每段素材核验帧数上限（性质与帧号无关；每帧要重开视频解码）


def check(name: str, tmp: Path) -> tuple[list[str], list[float]]:
    """返回（问题列表，未夹边帧的居中偏差列表）。"""
    spec = C.get(name)
    res = P.preprocess(spec)
    qp = C.ANNOT_QUEUE_DIR / f"{name}.csv"
    if not qp.exists():
        return [f"{name}: 无标注队列，跳过"], []

    rows = []
    with qp.open(newline="") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line.rstrip("\n").split(","))
    head, body = rows[0], rows[1:][:MAX_FRAMES]
    idx = {c: i for i, c in enumerate(head)}
    ww, wh = res.work_size

    problems: list[str] = []
    centered: list[float] = []
    for r in body:
        k = int(float(r[idx["k"]]))
        cx, cy = float(r[idx["cx"]]), float(r[idx["cy"]])
        w, h = float(r[idx["w"]]), float(r[idx["h"]])
        rect = CR.crop_rect_for(res, cx, cy, w, h)
        v = CR.render(res, spec, k, rect, tmp / f"{name}_{k:05d}.jpg")

        # ---- 辅助：往返闭合 ----
        for (px, py) in ((cx, cy), (rect[0], rect[1]), (rect[2], rect[3])):
            X, Y = v.work_to_disp(px, py)
            bx, by = v.disp_to_work(X, Y)
            if float(np.hypot(bx - px, by - py)) > 1e-6:
                problems.append(f"{name} k={k}: 往返不闭合")
                break

        # ---- 主断言：模板匹配反解映射 ----
        half = PATCH // 2
        ix0, iy0 = int(round(cx)) - half, int(round(cy)) - half
        ix1, iy1 = ix0 + PATCH, iy0 + PATCH
        inside = (0 <= ix0 and ix1 <= ww and 0 <= iy0 and iy1 <= wh)
        clipped = (abs(rect[0] - (cx - 0.5 * C.ANNOT_PAD * max(w, h))) > 0.5
                   or abs(rect[1] - (cy - 0.5 * C.ANNOT_PAD * max(w, h))) > 0.5
                   or abs(rect[2] - (cx + 0.5 * C.ANNOT_PAD * max(w, h))) > 0.5
                   or abs(rect[3] - (cy + 0.5 * C.ANNOT_PAD * max(w, h))) > 0.5)
        if not inside:
            problems.append(f"{name} k={k}: 取样小片超出工作图（跳过映射断言）")
            continue

        work = P.read_work_frame(res, spec.path, res.kept_frames[k])
        tile = cv2.cvtColor(work[iy0:iy1, ix0:ix1], cv2.COLOR_BGR2GRAY)
        tw, th = max(4, int(round(PATCH * v.sx))), max(4, int(round(PATCH * v.sy)))
        tmpl = cv2.resize(tile, (tw, th), interpolation=cv2.INTER_CUBIC)
        crop = cv2.imread(str(v.path), cv2.IMREAD_GRAYSCALE)
        if crop is None or tw >= crop.shape[1] or th >= crop.shape[0]:
            problems.append(f"{name} k={k}: 裁图尺寸异常，跳过映射断言")
            continue

        resm = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(resm)
        X, Y = v.work_to_disp(cx, cy)
        exp = (X - tw / 2.0, Y - th / 2.0)
        err = float(np.hypot(loc[0] - exp[0], loc[1] - exp[1]))
        if score < MIN_SCORE:
            problems.append(f"{name} k={k}: 模板匹配太弱（score={score:.2f}），本次不计")
        elif err > TOL_PX:
            problems.append(
                f"{name} k={k}: 映射误差 {err:6.1f} px —— 声称在 ({exp[0]:.0f},{exp[1]:.0f})，"
                f"实际在 ({loc[0]},{loc[1]})（score={score:.2f}）")

        if not clipped:
            centered.append(float(np.hypot(X - v.iw / 2, Y - v.ih / 2)))

    l, t, r, b = res.crop
    extra = (f"  未夹边帧居中偏差 ≤ {max(centered):.1f} px" if centered else "  全帧夹边，居中不适用")
    if l or t:
        extra += f"  （裁移 l={l},t={t}：漏减会让坐标偏 {l if l else 0}/{t if t else 0} px）"
    print(f"  {name:>8}  crop={tuple(res.crop)}  核验 {len(body)} 帧" + extra)
    return problems, centered


def stale_report() -> int:
    """列出磁盘上几何指纹对不上的裁图缓存。"""
    n_stale = n_ok = 0
    per: dict[str, int] = {}
    root = C.ANNOT_CROP_DIR
    if not root.exists():
        print("\n（还没有裁图缓存）")
        return 0
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        try:
            want = CR.geom_key(P.preprocess(C.get(d.name)))
        except Exception:                              # noqa: BLE001
            continue
        for mp in sorted(d.glob("*.json")):
            try:
                ok = CR.CropView.from_json(mp.read_text(encoding="utf-8")).geom == want
            except Exception:                          # noqa: BLE001
                ok = False
            if ok:
                n_ok += 1
            else:
                n_stale += 1
                per[d.name] = per.get(d.name, 0) + 1
    print(f"\n裁图缓存几何指纹：一致 {n_ok}，过期 {n_stale}")
    for k, n in sorted(per.items()):
        print(f"  {k}: {n} 张过期（下次访问会按当前几何重出，只换换算参数，图像内容不变）")
    return n_stale


def main() -> None:
    do_stale = "--stale" in sys.argv
    names: list[str] = []
    root = C.ANNOT_CROP_DIR
    if root.exists():
        names = [d.name for d in sorted(root.iterdir())
                 if d.is_dir() and (C.ANNOT_QUEUE_DIR / f"{d.name}.csv").exists()]
    if not names:
        names = [s.name for s in C.annotation_videos()]

    print("=== 裁图坐标换算核验（渲染到临时目录，不动 outputs/）===")
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for n in names:
            try:
                p, _ = check(n, tmp)
                problems += p
            except Exception as e:                      # noqa: BLE001
                problems.append(f"{n}: 核验失败 {type(e).__name__}: {e}")

    if problems:
        print(f"\n✗ {len(problems)} 处不通过：")
        for p in problems[:40]:
            print("   " + p)
        if len(problems) > 40:
            print(f"   ……还有 {len(problems) - 40} 处")
    else:
        print(f"\n✓ 全部通过：工作图小片都能在裁图里被找到，映射误差 ≤ {TOL_PX} px")

    if do_stale:
        stale_report()


if __name__ == "__main__":
    main()
