"""命令行入口：把预处理、检测、运动学串成流水线。

用法（在 demo02 目录下执行）：

    # 跑主线素材（静止机位、非搁置）
    python -m src.pipeline --all-static

    # 指定素材
    python -m src.pipeline --videos video02 video12 video15

    # 只跑某一阶段；--force 忽略缓存重算
    python -m src.pipeline --videos video02 --stage preprocess --force

产物：
    outputs/preprocess/<name>.json        预处理结果（可复查）
    outputs/tracks/<name>.csv             逐帧运动学序列
    outputs/plots/<name>_kinematics.png   轨迹/速度/航向/滑移角 四联图
    outputs/plots/<name>_detect_check.png 检测抽帧目检图
    outputs/reports/M1_报告.md            汇总报告
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from . import config as C
from . import detect as D
from . import kinematics as K
from . import preprocess as P

STAGES = ("preprocess", "detect", "kinematics", "all")


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_kinematics(name: str, res: P.PreprocessResult, kin: dict, path: Path) -> None:
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{name}  运动学曲线（背景建模分支，无人工标注）", fontsize=13)
    t = np.array(res.t)

    ax = axes[0][0]
    sc = ax.scatter(kin["cx"], kin["cy"], c=t, s=6, cmap="viridis")
    ax.plot(kin["cx"], kin["cy"], lw=0.6, alpha=0.35, color="gray")
    ax.plot(kin["cx"][0], kin["cy"][0], "o", ms=9, mfc="none", mec="#E24B4A", mew=2)
    ax.plot(kin["cx"][-1], kin["cy"][-1], "s", ms=9, mfc="none", mec="#185FA5", mew=2)
    ax.annotate("起点", (kin["cx"][0], kin["cy"][0]), textcoords="offset points",
                xytext=(8, 8), color="#E24B4A", fontsize=10)
    ax.annotate("终点", (kin["cx"][-1], kin["cy"][-1]), textcoords="offset points",
                xytext=(8, -14), color="#185FA5", fontsize=10)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("x（工作图像素）")
    ax.set_ylabel("y（工作图像素，已翻转成图像坐标）")
    ax.set_title("车辆轨迹（颜色=时间）", fontsize=11)
    fig.colorbar(sc, ax=ax, label="t / s", shrink=0.85)

    ax = axes[0][1]
    ax.plot(t, kin["speed"], lw=1.4, color="#185FA5")
    ax.set_xlabel("t / s")
    ax.set_ylabel("速度 / (px/s)")
    ax.set_title("速度", fontsize=11)
    ax.grid(alpha=0.25)

    ax = axes[1][0]
    ax.plot(t, np.unwrap(np.radians(np.nan_to_num(kin["psi_vel"], nan=0.0))) * 180 / np.pi,
            lw=1.4, color="#0F6E56")
    ax.set_xlabel("t / s")
    ax.set_ylabel("ψ_vel / 度（已解卷绕）")
    ax.set_title("运动方向 ψ_vel（图像右为 0°，顺时针为正）", fontsize=11)
    ax.grid(alpha=0.25)

    ax = axes[1][1]
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.plot(t, kin["beta"], lw=1.2, color="#993C1D")
    ax.set_xlabel("t / s")
    ax.set_ylabel("β / 度")
    ax.set_ylim(-90, 90)
    ax.set_title("滑移角 β（车身轴用掩膜 PCA 基线，噪声大，仅供占位）", fontsize=11)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_detect_check(name: str, spec: C.VideoSpec, res: P.PreprocessResult,
                      dets: list[D.Detection], path: Path, n_show: int = 4) -> None:
    """抽帧把检测结果画到工作图上，用于目检。"""
    import cv2
    if res.n == 0:
        return
    picks = np.linspace(0, res.n - 1, n_show).astype(int)
    by_k = {d.k: d for d in dets}
    cap = cv2.VideoCapture(str(spec.path))
    tiles = []
    l, t, r, b = res.crop
    sx, sy = P.work_scale(res)      # 工作图 -> 原图（roi）的缩放比，画框前必须换算
    for k in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, res.kept_frames[k])
        ok, fr = cap.read()
        if not ok:
            continue
        roi = fr[t:fr.shape[0] - b, l:fr.shape[1] - r].copy()
        d = by_k.get(k)
        if d is not None:
            x1, y1 = int((d.cx - d.w / 2) * sx), int((d.cy - d.h / 2) * sy)
            x2, y2 = int((d.cx + d.w / 2) * sx), int((d.cy + d.h / 2) * sy)
            cxx, cyy = int(d.cx * sx), int(d.cy * sy)
            cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.circle(roi, (cxx, cyy), 6, (0, 255, 0), -1)
            ang = np.radians(d.axis_deg)
            L = max(d.w, d.h) * 0.75 * sx
            cv2.line(roi, (int(cxx - L * np.cos(ang)), int(cyy - L * np.sin(ang))),
                     (int(cxx + L * np.cos(ang)), int(cyy + L * np.sin(ang))), (0, 255, 255), 3)
            cv2.putText(roi, f"k={k} conf={d.conf:.2f}", (max(4, x1), max(34, y1 - 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        else:
            cv2.putText(roi, f"k={k} 未检出", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
        s = 640 / roi.shape[1]
        tiles.append(cv2.resize(roi, (640, int(roi.shape[0] * s))))
    cap.release()
    if not tiles:
        return
    h = sum(x.shape[0] for x in tiles)
    sheet = np.zeros((h, 640, 3), np.uint8)
    y = 0
    for x in tiles:
        sheet[y:y + x.shape[0]] = x
        y += x.shape[0]
    cv2.putText(sheet, f"{name} 红框=检测框 绿点=质心 黄线=掩膜主轴(粗初值)",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


# ---------------------------------------------------------------------------
# 单段素材处理
# ---------------------------------------------------------------------------
def process_one(spec: C.VideoSpec, stage: str = "all", force: bool = False,
                verbose: bool = False) -> dict:
    rec: dict = {"name": spec.name, "role": spec.role, "source": spec.source}
    res = P.preprocess(spec, force=force and stage in ("preprocess", "all"))
    rec["preprocess"] = res
    if stage == "preprocess":
        return rec

    dets, info = D.run_detection(spec, res, verbose=verbose)
    rec["detect"] = info
    rec["n_det"] = len(dets)
    if dets:
        sz = np.array([[d.w, d.h] for d in dets])
        rec["box_med"] = (float(np.median(sz[:, 0])), float(np.median(sz[:, 1])))
        rec["conf_med"] = float(np.median([d.conf for d in dets]))

    if spec.has_geotrax:
        try:
            geo = D.GeoTraxDetector(res, spec.geotrax_track)
            rec["cross"] = D.cross_validate(dets, geo)
        except Exception as e:  # 交叉校验失败不该阻断主流程
            rec["cross"] = {"error": str(e)}
    if stage == "detect":
        return rec

    body = K.load_annotation(spec.name, res.n)
    kin = K.compute(dets, res, body_axis=body)
    rec["kin"] = kin
    rec["annotated"] = body is not None

    K.write_series_csv(C.TRACK_DIR / f"{spec.name}.csv", res, kin)
    if kin["s"]["covered"] >= 5:
        plot_kinematics(spec.name, res, kin, C.PLOT_DIR / f"{spec.name}_kinematics.png")
    plot_detect_check(spec.name, spec, res, dets, C.PLOT_DIR / f"{spec.name}_detect_check.png")
    return rec


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------
def write_report(records: list[dict]) -> Path:
    lines = ["# M1 阶段报告：预处理 + 双分支检测 + 运动方向", ""]
    lines.append("由 `python -m src.pipeline` 自动生成。本阶段**不需要任何人工标注**，")
    lines.append("产出的是速度与航向曲线，用来先暴露素材本身的跟踪与比例尺问题。")
    lines.append("")
    lines.append("## 1. 预处理与检测总览")
    lines.append("")
    lines.append("| 素材 | 分辨率 | 原始帧 | 唯一帧 | 重复率 | 有效时长 | 背景残留 | "
                 "检出 | 覆盖率 | 车框中位 | 时序剔除 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        res = r["preprocess"]
        info = r.get("detect", {})
        n_det = r.get("n_det", 0)
        cov = f"{n_det / res.n * 100:.0f}%" if res.n else "-"
        box = r.get("box_med")
        box_s = f"{box[0]:.0f}×{box[1]:.0f}" if box else "-"
        use = res.trim[1] - res.trim[0]
        lines.append(
            f"| {r['name']} | {res.src_size[0]}×{res.src_size[1]} | {use} | {res.n} | "
            f"{res.dup_ratio * 100:.1f}% | {res.duration:.1f}s | "
            f"{res.bg_residual_frac * 100:.2f}% | {n_det} | {cov} | {box_s} | "
            f"{info.get('dropped', '-')} |")
    lines.append("")
    lines.append("> 「覆盖率」= 检出帧数 / 去重后唯一帧数。「背景残留」= 中值背景扣掉车辆后"
                 f"仍剩的差异像素占比，阈值 {C.BG_RESIDUAL_MAX * 100:.0f}%；超过即判为相机在动。")
    lines.append("")

    lines.append("## 2. 与 geo-trax 的交叉校验")
    lines.append("")
    lines.append("两分支相互独立（背景建模零训练；geo-trax 是检测+跟踪模型），"
                 "若结论一致，说明车辆定位可信。")
    lines.append("")
    lines.append("| 素材 | 最吻合的 geo-trax 轨迹 | 共同帧 | IoU 中位 | IoU 均值 | IoU>0.5 占比 |")
    lines.append("|---|---|---|---|---|---|")
    any_cross = False
    broken = []
    for r in records:
        c = r.get("cross")
        if not c or "error" in c or not c:
            continue
        any_cross = True
        mark = ""
        if r["name"] in C.GEOTRAX_BROKEN:
            broken.append(r["name"])
            mark = " ⚠"
        lines.append(f"| {r['name']}{mark} | ID {c['track_id']} | {c['frames']} | "
                     f"{c['iou_median']:.3f} | {c['iou_mean']:.3f} | {c['iou_over_05'] * 100:.0f}% |")
    if not any_cross:
        lines.append("| — | 无可用 geo-trax 结果 | — | — | — | — |")
    lines.append("")
    if broken:
        names = "、".join(f"`{b}`" for b in broken)
        lines.append(f"> ⚠ {names} 的 geo-trax 输出本身已知不可用"
                     "（目标相对尺度远超模型训练尺度，且地面重复砖纹让稳定化退化，"
                     "详见《项目规划.md》§12.2）。该行的 IoU 无参考价值，"
                     "**不代表背景建模有误** —— 这几段的我方检出覆盖率为 100%。")
        lines.append("")

    lines.append("## 3. 逐素材运动学摘要")
    lines.append("")
    lines.append("| 素材 | 有效段 | 内部最大缺口 | 位置离群剔除 | 速度 中位/p90 (px/s) | 航向覆盖 | 主轴 std | 标注 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in records:
        kin = r.get("kin")
        if not kin:
            continue
        s = kin["s"]
        sp = kin["speed"][~np.isnan(kin["speed"])]
        ax = kin["axis"][~np.isnan(kin["axis"])]
        sp_s = f"{np.median(sp):.0f} / {np.percentile(sp, 90):.0f}" if len(sp) else "-"
        ax_s = f"{np.std(ax):.0f}°" if len(ax) else "-"
        span = f"k {s['valid_from']}~{s['valid_to']}"
        out_s = (f"{kin['n_outliers']}（{kin['n_outliers'] / max(1, r.get('n_det', 1)) * 100:.0f}%）"
                 if kin.get("n_outliers") else "0")
        lines.append(f"| {r['name']} | {span} | {s['max_gap']} 帧 | {out_s} | {sp_s} | "
                     f"{s['covered']}/{s['n']} | {ax_s} | "
                     f"{'有' if r.get('annotated') else '无（用 PCA 基线）'} |")
    lines.append("")
    lines.append("> 「位置离群剔除」= 滑动窗口中值法判为偏离主轨迹、已置空并插值补回的帧数。"
                 "用于清掉车被画面边缘截断、掩膜临时并进阴影/杂物造成的假跳点。")
    lines.append("")

    lines.append("## 4. 已知局限（M1 未解决）")
    lines.append("")
    lines.append("1. **车身阴影并进检测框。** 顶光角度低时车辆在地面的投影与车身差异都很强，")
    lines.append("   会连成同一个连通域，把框撑大并让质心偏向阴影一侧。")
    lines.append("   目前已用「按差异强度加权的质心」压制（比取掩膜像素质心稳），但没有根治。")
    lines.append("   后果：框尺寸偏大、质心有随车体转动而摆动的小残差 → 速度曲线上有毛刺。")
    lines.append("2. **video15 有 12% 的帧被判为位置离群。** 剔除后速度曲线从 p90 2168 px/s "
                 "降到 215 px/s（10 倍），说明原来的尖峰主要是假跳点。")
    lines.append("   但**这 12% 究竟是「检测被车身阴影带偏」还是「原片本身有快动作段落」，"
                 "目前尚未区分**。")
    lines.append("   在人工逐帧看一遍之前，不要引用 video15 的速度绝对值。")
    lines.append("3. **video12 的轨迹仍不稳。** 位置离群只剔掉 3%，但逐帧位移的 p95/p50 仍达 19"
                 "（video02 为 1.8）。")
    lines.append("   背景模型残留差异约 4%（超过 3% 阈值）：相机在 16.5 s 内缓慢漂移约 23 px，")
    lines.append("   滚动背景只能缓解不能消除。这段素材的检测结果目前只适合看趋势，不适合做定量评估。")
    lines.append("")
    lines.append("## 5. 需要人工介入的事")
    lines.append("")
    lines.append("1. **车身朝向 ψ_body 必须人工标注。** 本阶段的 β 曲线用的是掩膜 PCA 主轴，")
    lines.append("   而车辆阴影会并进掩膜，使主轴标准差普遍偏大，**不可当结果使用**。")
    lines.append("   标注文件格式见 `outputs/annotations/README.md`。")
    lines.append("2. **比例尺尚未标定。** 速度目前是 px/s。要换算成 m/s，需要地面已知尺寸"
                 "（例如 video03 的靶心圆、video13/14 的轮胎痕圆环）。")
    lines.append("3. **漏检帧**：背景建模在车辆停住或与背景同色时会丢失目标，"
                 "报告里的「覆盖率」直接反映这一点。")
    lines.append("")
    lines.append("## 6. 逐素材警告")
    lines.append("")
    for r in records:
        ws = r["preprocess"].warnings
        if ws:
            lines.append(f"- **{r['name']}**：" + "；".join(ws))
    path = C.REPORT_DIR / "M1_报告.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


ANNOT_README = """# 人工标注说明

## 文件位置与格式

`outputs/annotations/<素材名>.csv`，两列，逗号分隔，可只标稀疏的若干帧：

```
k,axis_deg
0,42.5
5,47.0
10,55.3
```

- 第 1 列 `k`：**去重后的帧序号**（不是原始帧号）。取法见
  `outputs/tracks/<素材名>.csv` 的第一列，或用 `outputs/plots/<素材名>_kinematics.png`
  上的曲线横轴对照。
- 第 2 列 `axis_deg`：**车身轴向，无向，模 180°**。
  约定：图像 x 轴向右为 0°，顺时针为正（因为图像 y 轴向下，这与屏幕上的顺时针一致）。
  **只标"车身那条轴线"即可，不需要区分车头和车尾。**

## 为什么不用区分前后

滑移角 β = ψ_body − ψ_vel，而 ψ_body 是无向量。漂移时恒有 |β| < 90°，
所以程序会自动取"与运动方向夹角 ≤ 90°"的那一端作为车头 —— 180° 歧义自动消解。

## 关于帧号

`outputs/tracks/<素材名>.csv` 里有 `k` 与 `frame` 两列的对应关系，
`frame` 是原始视频帧号，`k` 是去重后序号。标注用 `k`，因为时间轴是按 `k` 重建的。
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="赛车漂移姿态识别 demo —— M1 流水线")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--videos", nargs="+", help="素材名，如 video02 video12")
    g.add_argument("--all-static", action="store_true",
                   help="所有登记的静止机位且非搁置的素材")
    g.add_argument("--all", action="store_true", help="全部 15 段素材")
    ap.add_argument("--stage", default="all", choices=STAGES)
    ap.add_argument("--force", action="store_true", help="忽略缓存，重算预处理")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.videos:
        specs = [C.get(v) for v in args.videos]
    elif args.all_static:
        specs = C.active_static_videos()
    else:
        specs = list(C.VIDEOS.values())

    # 标注说明文件
    (C.ANNOT_DIR / "README.md").write_text(ANNOT_README, encoding="utf-8")

    print(f"处理 {len(specs)} 段素材，阶段：{args.stage}")
    print("=" * 100)
    records = []
    for spec in specs:
        print(f"\n[{spec.name}] role={spec.role} 分支={spec.detector}")
        try:
            rec = process_one(spec, stage=args.stage, force=args.force, verbose=args.verbose)
        except Exception as e:
            print(f"  失败：{type(e).__name__}: {e}")
            continue
        records.append(rec)
        res = rec["preprocess"]
        print("  " + res.summary_line())
        for w in res.warnings:
            print(f"  [警告] {w}")
        if args.stage != "preprocess":
            print(f"  检出 {rec.get('n_det', 0)}/{res.n} 帧"
                  f"（时序剔除 {rec.get('detect', {}).get('dropped', '-')}）")
            c = rec.get("cross")
            if c and "error" not in c and c:
                print(f"  交叉校验 geo-trax ID {c['track_id']}：共同帧 {c['frames']}，"
                      f"IoU 中位 {c['iou_median']:.3f}，>0.5 占比 {c['iou_over_05'] * 100:.0f}%")
            kin = rec.get("kin")
            if kin:
                sp = kin["speed"][~np.isnan(kin["speed"])]
                if len(sp):
                    print(f"  速度 中位 {np.median(sp):.0f} px/s，"
                          f"p90 {np.percentile(sp, 90):.0f} px/s"
                          f"（{np.median(sp) / kin['size_px']:.2f} 车长/秒）")

    if records and args.stage == "all":
        rp = write_report(records)
        print(f"\n汇总报告：{rp.relative_to(C.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
