"""生成 README「这套系统在算什么」一节用的概念示意图（两张 PNG）。

这两张图解释的是**读图的约定**，不是某段素材的结果，所以画的是示意、不是真实数据：

  · concept_axis_sources.png     —— 稀疏的人工标注如何变成逐帧序列
                                    （绿=观测 / 橙=插值 / 灰=留空）
  · concept_motion_direction.png —— 运动方向的基点：每帧掩膜质心 → 帧间位移 → ψ_vel

为什么单独写一个脚本，而不是往 README 里贴一张截图：
README 里的图必须**能从仓库重新生成**。截图会随代码改动静默失真，
而且没人知道它当初是怎么画出来的。

用法：
    /opt/anaconda3/bin/python3 tools/show_concept.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.transforms import Affine2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402

FIG_DIR = C.OUT_DIR / "reports" / "figures"

# 与项目其它出图脚本保持同一套字体回退链
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS",
                                   "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

GREEN = "#3B6D11"
ORANGE = "#BA7517"
GRAY = "#B4B2A9"
DIM = "#5F5E5A"
INK = "#2C2C2A"
BLUE = "#185FA5"
ARROW = "#378ADD"


def fig_axis_sources(out: Path) -> None:
    """上半：人工标注是稀疏的；下半：还原出的逐帧序列（含插值与长缺口）。"""
    # 示意用的已标帧号（取自一次真实标注的稀疏程度），最长的缺口超过 ANNOT_MAX_GAP
    ann = [2, 20, 38, 74, 96]
    gap_pair = (38, 74)

    fig, ax = plt.subplots(figsize=(11.0, 3.5), dpi=150)
    ax.set_xlim(-7, 104)
    ax.set_ylim(-0.42, 1.42)
    ax.axis("off")

    # ---- 行 1：人工标注帧 ----
    y1 = 1.02
    ax.plot([0, 100], [y1, y1], color="#D3D1C7", lw=1, zorder=1)
    for k in ann:
        ax.plot([k, k], [y1 - 0.07, y1 + 0.07], color=GREEN, lw=2.4,
                solid_capstyle="round", zorder=3)
    ax.text(-7, y1 + 0.17, "人工标注帧（稀疏）", fontsize=10.5, color=DIM,
            ha="left", va="bottom")
    ax.text(101, y1, "帧号 k →", fontsize=9.5, color=DIM, ha="left", va="center")

    # ---- 行 2：还原出的逐帧序列 ----
    y2 = 0.42
    # 两端不外推：只画到首末已标帧
    ax.plot([0, ann[0]], [y2, y2], color="#E5E3DC", lw=1.4,
            linestyle=(0, (1, 3)), zorder=1)
    ax.plot([ann[-1], 100], [y2, y2], color="#E5E3DC", lw=1.4,
            linestyle=(0, (1, 3)), zorder=1)
    for a, b in zip(ann[:-1], ann[1:]):
        if (a, b) == gap_pair:
            ax.plot([a, b], [y2, y2], color=GRAY, lw=2.0,
                    linestyle=(0, (5, 4)), zorder=2)
        else:
            ax.plot([a, b], [y2, y2], color=ORANGE, lw=3.0,
                    solid_capstyle="butt", zorder=2)
    ax.plot(ann, [y2] * len(ann), "o", color=GREEN, ms=6, zorder=4,
            markeredgecolor="white", markeredgewidth=1.0)
    ax.text(-7, y2 + 0.17, "还原成逐帧", fontsize=10.5, color=DIM,
            ha="left", va="bottom")

    gap = gap_pair[1] - gap_pair[0] - 1
    ax.annotate(f"缺口 {gap} 帧 > 上限 {C.ANNOT_MAX_GAP} → 不插值，留空",
                xy=((gap_pair[0] + gap_pair[1]) / 2, y2),
                xytext=((gap_pair[0] + gap_pair[1]) / 2, y2 + 0.30),
                fontsize=9.5, color=DIM, ha="center",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    ax.annotate("两端不外推（标到哪算到哪）", xy=(1, y2), xytext=(1, y2 - 0.30),
                fontsize=9.5, color=DIM, ha="left",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))

    # ---- 图例 ----
    handles = [
        plt.Line2D([], [], marker="o", color=GREEN, lw=0, ms=6,
                   label="人工标注（观测，可当结果）"),
        plt.Line2D([], [], color=ORANGE, lw=3.0, label="插值补齐（推测，别与观测平权）"),
        plt.Line2D([], [], color=GRAY, lw=2.0, linestyle=(0, (5, 4)),
                   label="留空（该帧没有轴）"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.005, -0.02),
              ncol=3, frameon=False, fontsize=10, handlelength=2.2,
              columnspacing=2.4)

    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _bezerier(p0, c, p1, ts):
    p0, c, p1 = (np.asarray(v, dtype=float) for v in (p0, c, p1))
    t = np.asarray(ts, dtype=float)[:, None]
    return ((1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1), t


def _tangent_angle(pts):
    d = np.gradient(pts, axis=0)
    return np.degrees(np.arctan2(d[:, 1], d[:, 0]))


def _car(ax, cx, cy, ang, L=15.0, W=6.6):
    """画一辆俯视的车（旋转矩形 + 车顶），角度为逆时针度数。"""
    rot = Affine2D().rotate_deg_around(cx, cy, ang) + ax.transData
    body = Rectangle((cx - L / 2, cy - W / 2), L, W, facecolor="white",
                     edgecolor="#888780", lw=0.8, zorder=4, transform=rot)
    cab = Rectangle((cx - L * 0.30, cy - W * 0.30), L * 0.44, W * 0.60,
                    facecolor="#D3D1C7", edgecolor="none", zorder=5,
                    transform=rot)
    ax.add_patch(body)
    ax.add_patch(cab)


def fig_motion_direction(out: Path) -> None:
    """运动方向的基点：每帧一个加权质心，方向 = 相邻质心的位移方位。"""
    ts_fine = np.linspace(0.0, 1.0, 200)
    path, _ = _bezerier((2, -2), (46, 26), (98, 32), ts_fine)

    ts_car = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cars, _ = _bezerier((2, -2), (46, 26), (98, 32), ts_car)
    angs = _tangent_angle(cars)

    fig, ax = plt.subplots(figsize=(11.0, 4.3), dpi=150)
    ax.set_xlim(-6, 106)
    ax.set_ylim(-12, 44)
    ax.set_aspect("equal")
    ax.axis("off")

    # 轨迹（平滑后）。标注放到左上方的空白处 —— 贴着起点会被"质心"那条标注压住
    ax.plot(path[:, 0], path[:, 1], color=GRAY, lw=1.2,
            linestyle=(0, (4, 3)), zorder=1)
    mid = path[50]
    ax.annotate("平滑后的质心轨迹", xy=(mid[0], mid[1]),
                xytext=(9, 17.5), fontsize=10, color=DIM, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))

    # 第 3 帧：把"区域 → 一个点"画出来
    k3 = 2
    ax.add_patch(Ellipse((cars[k3, 0], cars[k3, 1]), width=23, height=11,
                         angle=angs[k3], facecolor="#FAEEDA",
                         edgecolor="#EF9F27", lw=0.9, zorder=2, alpha=0.85))
    ax.annotate("每帧的检测掩膜（示意）\n实际形状随背景差分变化，常呈月牙带",
                xy=(cars[k3, 0] + 9, cars[k3, 1] - 4.6),
                xytext=(cars[k3, 0] + 9, cars[k3, 1] - 12),
                fontsize=9.5, color=ORANGE, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color="#EF9F27", lw=0.8))

    for (cx, cy), a in zip(cars, angs):
        _car(ax, cx, cy, a)

    # 帧间位移 = 速度方向
    for i in range(len(cars) - 1):
        v = cars[i + 1] - cars[i]
        ax.annotate("", xy=cars[i] + v * 0.88, xytext=cars[i] + v * 0.16,
                    arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.6,
                                    mutation_scale=13), zorder=6)

    ax.plot(cars[:, 0], cars[:, 1], "o", color=BLUE, ms=5.0, zorder=7,
            markeredgecolor="white", markeredgewidth=1.0)
    ax.annotate("质心（每帧的代表点）", xy=(cars[0, 0], cars[0, 1]),
                xytext=(cars[0, 0] - 6, cars[0, 1] - 8),
                fontsize=9.5, color=BLUE, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax.annotate("psi_vel = atan2(vy, vx)", xy=cars[-1],
                xytext=(cars[-1, 0] - 6, cars[-1, 1] + 6.5),
                fontsize=10, color=BLUE, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))

    handles = [
        plt.Line2D([], [], marker="o", color=BLUE, lw=0, ms=5.5,
                   label="掩膜加权质心（方向由此点求出）"),
        plt.Line2D([], [], color=ARROW, lw=1.6, label="帧间位移 → 速度方向"),
        plt.Line2D([], [], color=GRAY, lw=1.2, linestyle=(0, (4, 3)),
                   label="平滑后的轨迹"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10,
              handlelength=2.2)

    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outs = [FIG_DIR / "concept_axis_sources.png",
            FIG_DIR / "concept_motion_direction.png"]
    fig_axis_sources(outs[0])
    fig_motion_direction(outs[1])
    for p in outs:
        print(f"  写出 {p.relative_to(ROOT)}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
