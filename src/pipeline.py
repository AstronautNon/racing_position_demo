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
    t = np.array(res.t)

    # 轴来源逐帧不同，图上的说法必须跟着实际来源走，
    # 不能写死成「无人工标注 / PCA 基线」—— 否则标完的素材会被误读成占位结果。
    kind = np.asarray(kin.get("axis_kind", np.zeros(res.n, dtype=int)))
    n_ann = int((kind == 1).sum())
    n_int = int((kind == 2).sum())
    if n_ann or n_int:
        src_txt = f"车身轴：人工标注 {n_ann} 帧 + 插值 {n_int} 帧"
    else:
        src_txt = "车身轴：掩膜 PCA 基线（占位，非结果）"
    fig.suptitle(f"{name}  运动学曲线（背景建模分支，{src_txt}）", fontsize=13)

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
    beta = np.asarray(kin["beta"], dtype=float)
    ax.plot(t, beta, lw=1.2, color="#993C1D", label="β 曲线")
    # 人工标的帧单独点出来：插值段是「匀速转动」假设推出来的，不能和目视观测平权
    if n_ann:
        m = (kind == 1) & np.isfinite(beta)
        ax.plot(t[m], beta[m], "o", ms=4.5, mfc="none", mec="#185FA5", mew=1.2,
                ls="none", label=f"人工标注帧（{int(m.sum())}）")
        ax.legend(fontsize=9, loc="best")
    ax.set_xlabel("t / s")
    ax.set_ylabel("β / 度")
    ax.set_ylim(-90, 90)
    ax.set_title("滑移角 β" + ("" if n_ann else "（车身轴用掩膜 PCA 基线，噪声大，仅供占位）"),
                 fontsize=11)
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

    # 必须用 load_annotation_detail 并把 annotated 掩膜一起传下去：
    # 只传角度序列的话 compute() 无从区分「人工标的」和「插值补的」，
    # axis_kind 会全部落成 2（插值），CSV 的 axis_src 列就成了假审计。
    ann = K.load_annotation_detail(spec.name, res.n)
    body = ann["axis"] if ann else None
    bmask = ann["annotated"] if ann else None
    kin = K.compute(dets, res, body_axis=body, body_mask=bmask)
    rec["kin"] = kin
    rec["annotated"] = ann is not None
    rec["annot_count"] = ann["count"] if ann else 0
    rec["axis_stats"] = K.annotate_stats(kin)

    K.write_series_csv(C.TRACK_DIR / f"{spec.name}.csv", res, kin)
    if kin["s"]["covered"] >= 5:
        plot_kinematics(spec.name, res, kin, C.PLOT_DIR / f"{spec.name}_kinematics.png")
    plot_detect_check(spec.name, spec, res, dets, C.PLOT_DIR / f"{spec.name}_detect_check.png")
    return rec


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------
def write_report(records: list[dict], partial: bool = False,
                 missing: list[tuple[str, str]] | None = None) -> Path:
    lines = ["# M1 阶段报告：预处理 + 双分支检测 + 运动方向", ""]
    if partial:
        # 跑单个素材时如果照原样写回 M1_报告.md，会把全量汇总冲成只有一行，
        # 下次打开的人会以为其它素材消失了。局部结果单独成文并显式声明范围。
        names = "、".join(f"`{r['name']}`" for r in records)
        lines.append(f"> ⚠ **本报告只覆盖 {len(records)} 段素材（{names}），是局部运行的结果，"
                     "不是全量汇总。**")
        lines.append("> 全量报告请跑 `python -m src.pipeline --all-static`；"
                     "本次结果写在本文件以免覆盖它。")
        lines.append("")
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
    lines.append("| 素材 | 有效段 | 内部最大缺口 | 位置离群剔除 | 速度 中位/p90 (px/s) | 航向覆盖 | 轴残差 | 标注 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for rec in records:
        kin = rec.get("kin")
        if not kin:
            continue
        s = kin["s"]
        sp = kin["speed"][~np.isnan(kin["speed"])]
        sp_s = f"{np.median(sp):.0f} / {np.percentile(sp, 90):.0f}" if len(sp) else "-"
        # 车身轴在整段里会真实转过几十度，普通 std / 圆 std 都被真实转动主导，
        # 测不出噪声。用"去掉平滑趋势后的残差散布"才能把 PCA 基线与人工标注拉开。
        resid = K.axis_residual_std(kin["axis"], kin["dt"])
        ax_s = f"{resid:.1f}°" if resid is not None else "-"
        span = f"k {s['valid_from']}~{s['valid_to']}"
        out_s = (f"{kin['n_outliers']}（{kin['n_outliers'] / max(1, rec.get('n_det', 1)) * 100:.0f}%）"
                 if kin.get("n_outliers") else "0")
        st = rec.get("axis_stats") or {}
        if st.get("annotated"):
            ann_s = f"{rec.get('annot_count', 0)} 帧人工"
            if st.get("interp"):
                ann_s += f" + {st['interp']} 插值"
        else:
            ann_s = "无（用 PCA 基线）"
        lines.append(f"| {rec['name']} | {span} | {s['max_gap']} 帧 | {out_s} | {sp_s} | "
                     f"{s['covered']}/{s['n']} | {ax_s} | {ann_s} |")
    lines.append("")
    lines.append("> 「位置离群剔除」= 滑动窗口中值法判为偏离主轨迹、已置空并插值补回的帧数。"
                 "用于清掉车被画面边缘截断、掩膜临时并进阴影/杂物造成的假跳点。")
    lines.append(">")
    lines.append("> 「轴残差」= 车身轴**去掉平滑趋势后**的残差标准差（0.5 s 窗口，在倍角域做平滑）。"
                 "**不能用普通 std 或圆 std 代替**：车身轴在一段素材里会真实转过几十度，"
                 "那两种度量都被真实转动主导 —— 实测 video09 的 PCA 基线与人工标注圆 std "
                 "是 12.1° vs 11.8°（几乎一样），但两者的 β 中位相差 7.15° 且方向一致，"
                 "是**系统性偏差**而非随机噪声。")
    lines.append(">")
    lines.append("> 该列只量「抖动」：video09 标注前的 PCA 基线为 3.8°，标注后为 1.6°。"
                 "拆开看，**人工帧残差 1.83°**（≈ 目视点两下的精度），"
                 "插值帧仅 0.13° —— 但插值段平滑是**按构造成立**的（倍角线性插值本身连续），"
                 "**不能反过来当作「插值很准」的证据**。所以该列衡量的是「轴序列抖不抖」，"
                 "不等于人工点选精度，更不等于 β 的精度。")
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
    done = [r for r in records if r.get("annot_count")]
    if done:
        lines.append("标注进度：" + "、".join(
            f"`{r['name']}` {r['annot_count']} 帧" for r in done)
            + "（逐帧来源见上表「标注」列与 `outputs/tracks/<素材名>.csv` 的 `axis_src`）。")
        lines.append("")
    lines.append("1. **车身朝向 ψ_body 必须人工标注。** 尚未标注的素材，其 β 曲线用的是掩膜 PCA 主轴，")
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
    lines.append("")
    if missing:
        lines.append("## 7. 本次未跑到的素材")
        lines.append("")
        lines.append("以下素材在本次运行时**源文件不在 `drift/`**，表中因此没有它们的行。")
        lines.append("它们此前生成的 `outputs/tracks/*.csv` 与曲线图仍在磁盘上，"
                     "但**可能已与最新代码不一致**，不要当作本次结果引用。")
        lines.append("")
        for n, why in missing:
            lines.append(f"- **{n}**：{why}")
        lines.append("")
    path = C.REPORT_DIR / ("M1_报告_局部.md" if partial else "M1_报告.md")
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
    missing = []
    for spec in specs:
        print(f"\n[{spec.name}] role={spec.role} 分支={spec.detector}")
        try:
            rec = process_one(spec, stage=args.stage, force=args.force, verbose=args.verbose)
        except FileNotFoundError as e:
            # 素材文件不在（改名/误删）时别静默跳过：报告里要留下痕迹，
            # 否则表格少一行没人会发现，旧产物还会被误当成最新结果。
            print(f"  [缺失] {e}")
            missing.append((spec.name, str(e)))
            continue
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
        full = {s.name for s in C.active_static_videos()}
        partial = {s.name for s in specs} != full
        rp = write_report(records, partial=partial, missing=missing)
        print(f"\n汇总报告：{rp.relative_to(C.ROOT)}" + ("（局部，未覆盖全量报告）" if partial else ""))
        if missing:
            print("未跑到的素材：" + "、".join(n for n, _ in missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
