"""诊断并尝试修正自动轴的 90° 分支翻转。

背景（见 项目规划.md §14.10）：

`--status` 报的「PCA 提示 vs 人工」**只取中位数**，video03 上显示 1.3°「尚可」，
但实际 p90 是 80.8° —— 因为月牙掩膜的主轴在若干帧里**整体转了 90°**
（PCA 把次轴当主轴）。这是**离散的分支翻转**，不是连续噪声：

    k=18   月牙 135.8° / 真值 71.7°   → 差 64.1°（翻的）
    k=20   月牙  52.7° / 真值 67.9°   → 差 15.2°（对的）

两帧之间月牙轴跳了 83°，而车身轴物理上是连续的。

车身轴是 mod 180 的无向量，所以「θ」与「θ+90」是**两个可选的候选分支**。
本脚本把"逐帧选哪个分支"写成一个一维离散优化（链式 DP）：

    代价 = Σ 平滑项(相邻帧的轴变化) + λ · 锚定项(与轮廓轴之差)

- 平滑项：车身轴物理连续，真翻转会表现为 ~90° 的跳变，必须罚；
- 锚定项：轮廓轴（颜色分割）是**绝对**估计，用来防止"整段都翻"时
  平滑项无法自救（纯平滑只保证连续，不保证对）。权重给弱值，
  因为它自己也不够准（见 §14.9）。

输出四种轴对真值的误差分布，用于判断这条路值不值得做进流水线。

用法：
    /opt/anaconda3/bin/python3 tools/probe_axis_branch.py video02 video03 video09
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import annotate as A          # noqa: E402
from src import config as C            # noqa: E402
from src import detect as D            # noqa: E402
from src import preprocess as P        # noqa: E402

MOD = C.BODY_AXIS_MOD
W_SMOOTH = 1.0     # 平滑项权重
W_ANCHOR = 0.35    # 锚定项权重（弱：轮廓轴自身也不够准）


def fold90(d: np.ndarray | float) -> np.ndarray | float:
    """把角度差折到 [-90, 90]（无向轴的距离）。"""
    return (d + 90.0) % MOD - 90.0


def dist(a: float, b: float) -> float:
    return abs((a - b + 90.0) % MOD - 90.0)


def resolve_branch(theta: list[float], phi: list[float | None],
                   ) -> tuple[list[float], list[int]]:
    """在 {θ, θ+90} 两个候选分支间做链式 DP，返回 (解出的轴, 分支选择)。

    调用方只传入**有检出**的帧，因此序列是一条连续链，不必处理断链回溯。
    """
    n = len(theta)
    INF = 1e18
    dp = [0.0, 0.0]
    back: list[list[int | None]] = []

    for k in range(n):
        cand = [theta[k] % MOD, (theta[k] + 90.0) % MOD]
        anchor = [0.0, 0.0]
        if phi[k] is not None:
            anchor = [W_ANCHOR * dist(cand[s], phi[k]) for s in (0, 1)]
        new = [INF, INF]
        bp: list[int | None] = [None, None]
        for s in (0, 1):
            if k == 0:
                new[s] = anchor[s]          # 链起点：只付锚定代价
                continue
            for sp in (0, 1):
                if dp[sp] >= INF:
                    continue
                smooth = W_SMOOTH * abs(fold90(cand[s] - prev_cand[sp]))
                c = dp[sp] + smooth + anchor[s]
                if c < new[s]:
                    new[s] = c
                    bp[s] = sp
        dp = new
        back.append(bp)
        prev_cand = cand

    sel = [0] * n
    s = int(np.argmin(dp))
    for k in range(n - 1, -1, -1):
        sel[k] = s
        bp = back[k]
        if bp[s] is None:
            break
        s = bp[s]

    out = [(theta[k] + 90.0 * sel[k]) % MOD for k in range(n)]
    return out, sel


def check(name: str) -> dict | None:
    spec = C.get(name)
    if spec is None or not spec.path.exists():
        print(f"\n[{name}] 源文件不存在，跳过")
        return None
    res = P.preprocess(spec)
    dets = D.BgSubDetector(res, spec.path).run()
    by_k = {d.k: d for d in dets}
    frames: dict[int, np.ndarray] = {}
    for k, _, _, wf in D.iter_work_frames(res, spec.path):
        if k in by_k:
            frames[k] = wf

    truth = {k: v["axis_deg"] for k, v in A.read_labels(name).items()
             if "axis_deg" in v}
    if not truth:
        print(f"\n[{name}] 无人工真值，跳过")
        return None

    ks: list[int] = []          # 只保留"有检出且能取到工作图"的帧
    theta: list[float] = []
    phi: list[float | None] = []
    for k in sorted(by_k):
        d = by_k[k]
        wf = frames.get(k)
        if wf is None:
            continue
        ks.append(k)
        theta.append(float(d.axis_deg))
        got = D.contour_axis(wf, d.cx, d.cy, d.w, d.h)
        phi.append(float(got[0]) if got else None)

    moon, sel = resolve_branch(theta, phi)

    picks = [k for k in ks if k in truth]
    pos = {k: i for i, k in enumerate(ks)}
    stats: dict[str, np.ndarray] = {}
    for tag, seq in (("月牙原始", theta), ("轮廓轴", phi), ("DP 解分支", moon)):
        v = np.array([dist(seq[pos[k]], truth[k]) for k in picks
                      if seq[pos[k]] is not None])
        if len(v):
            stats[tag] = v

    flipped = sum(sel[pos[k]] for k in picks)
    print(f"\n[{name}] 真值 {len(picks)} 帧，检出 {len(ks)} 帧，"
          f"DP 判定翻转 {flipped}/{len(picks)} 帧")
    print(f"  {'轴来源':<12}{'中位':>8}{'p90':>9}{'最大':>9}{'|Δ|>30°':>9}")
    print("  " + "-" * 47)
    for tag in ("月牙原始", "轮廓轴", "DP 解分支"):
        v = stats.get(tag)
        if v is None:
            continue
        print(f"  {tag:<12}{np.median(v):>7.2f}°{np.percentile(v, 90):>8.2f}°"
              f"{v.max():>8.2f}°{int((v > 30).sum()):>9}")
    return {"name": name, "n": len(picks), "flipped": flipped,
            **{f"{t}_med": float(np.median(v)) for t, v in stats.items()},
            **{f"{t}_p90": float(np.percentile(v, 90)) for t, v in stats.items()}}


def main(argv: list[str]) -> int:
    names = [a for a in argv[1:] if not a.startswith("-")] or \
            ["video02", "video03", "video09"]
    print("90° 分支翻转诊断（链式 DP：平滑项 + 轮廓轴弱锚定）")
    print(f"权重：平滑 {W_SMOOTH}，锚定 {W_ANCHOR}")
    print("=" * 78)
    for n in names:
        check(n)
    print("\n" + "=" * 78)
    print("若「DP 解分支」的 p90 相比「月牙原始」大幅下降，说明翻转是真问题")
    print("且可被时间连续性修复 —— 那就值得做进 detect.py 作为正式轴来源。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
