"""姿态质量审计：量「车头朝向」与「运动方向」这两半各自可不可信。

为什么单独成一个模块
--------------------
项目里所有下游（β 曲线、漂移评分）都建立在**人工标注的车身轴**上，
而人工点选本身没有独立校验；同时 β 又是由定头规则构造出来的、
必然落在 (−90, 90]，**边界上的帧看不出异常**。两件事都需要专门的度量，
且必须让流水线报告与命令行工具用**同一份判据**，否则两处数字会对不上。

判据分两块：

1. **航向（车身轴）的帧间变化。**
   车是刚体，帧率给定了自转角速度的上界。量相邻**已标**帧之间的轴变化、
   除以间隔，超出物理界的就是可疑点选（或真·快速自转，需人眼复核）。
   用 `axis_dist`（折 90° 的无向轴距离）—— 人工标的是无向轴；
   90° 分支翻转在这个度量下表现为 ~90° 跳变，同样会被抓出来。

   **前提：素材必须是实时时间轴。** 原片若经过加速/缩时处理，相邻帧之间
   真实经过的时间大于 1/eff_fps，帧间角度变化天然偏大，这个物理界**不成立**
   （实测教训：曾据此把 video15 的 k=14→15、15→16 判为"物理不可能的点选错误"，
   实为原片加速所致）。故本判据只在 `realtime=True` 时作数，
   非实时素材只报变化幅度、不作判定。见 `C.VideoSpec.realtime`。

2. **β 的边界饱和度。**
   `undirected_resolve` 的定头规则是「取与运动方向夹角 ≤ 90° 的那一端」，
   于是车头方向**不是观测来的、是由运动方向推出来的**：
     - 真·|β| ≥ 90（车尾朝前滑 / 自转）时，车头会被判反 180°，
       而 β 只会顶在 ±90 上，曲线看不出任何异常；
     - 所以必须报出 |β| 逼近 90° 的帧占比 —— 那才是「车头朝向」不可信的地方。
"""

from __future__ import annotations

import numpy as np

from . import kinematics as K

# 物理门槛：25 fps 下一帧 40 ms。漂移车峰值自转约 120°/s ≈ 4.8°/帧；
# 取 20°/帧（=500°/s）作为「几乎不可能是真自转」的界 —— 偏宽松，以免误杀真快动作。
# **仅对实时素材成立**（见模块 docstring 第 1 条与 C.VideoSpec.realtime）。
SPIKE_DEG_PER_FRAME = 20.0
BETA_NEAR = 70.0    # |β| 超过它 → 定头开始不稳
BETA_CRIT = 85.0    # |β| 超过它 → 车头朝向基本等于抛硬币


def pose_metrics(kin: dict, top_spikes: int = 6) -> dict:
    """从**算好的** kin 里量姿态质量。

    只看 kin，不重跑检测，因此可以在流水线里零成本调用。
    """
    dt = float(kin["dt"])
    realtime = bool(kin.get("realtime", True))
    kind = np.asarray(kin["axis_kind"])
    axis = np.asarray(kin["axis"], dtype=float)

    # 1) 人工帧之间的轴变化 ------------------------------------------------
    ks = np.nonzero((kind == 1) & np.isfinite(axis))[0]
    spikes: list[dict] = []
    rate_p50 = rate_p90 = rate_max = float("nan")
    top_change: dict | None = None
    if len(ks) >= 2:
        a = axis[ks]
        d = np.asarray(K.axis_dist(a[1:], a[:-1]), dtype=float)
        dk = np.diff(ks).astype(float)
        rate = d / dk                       # 度/帧
        rate_p50 = float(np.median(rate))
        rate_p90 = float(np.percentile(rate, 90))
        for i in np.argsort(-rate)[:top_spikes]:
            item = dict(k0=int(ks[i]), k1=int(ks[i + 1]), dk=int(dk[i]),
                        deg=float(d[i]), rate=float(rate[i]),
                        dps=float(rate[i] / dt))
            if top_change is None:
                top_change = item
                rate_max = item["rate"]
            # 物理界只在实时素材上有意义；非实时素材一律不产出"可疑跳变"
            if not realtime or rate[i] < SPIKE_DEG_PER_FRAME:
                break
            spikes.append(item)
        spikes.sort(key=lambda s: -s["rate"])

    resid = K.axis_residual_std(axis, dt)

    # 2) β 边界饱和度 -------------------------------------------------------
    beta = np.asarray(kin["beta"], dtype=float)
    ab = np.abs(beta[np.isfinite(beta)])
    nan = float("nan")          # 无有效帧时统一返回 NaN，调用方按 NaN 显示 "—"
    return dict(
        n_human=int(len(ks)), resid=resid, realtime=realtime,
        rate_p50=rate_p50, rate_p90=rate_p90, rate_max=rate_max,
        top_change=top_change, spikes=spikes,
        n_beta=int(ab.size),
        b_med=float(np.median(ab)) if ab.size else nan,
        b_p90=float(np.percentile(ab, 90)) if ab.size else nan,
        b_max=float(ab.max()) if ab.size else nan,
        f_near=float(np.mean(ab > BETA_NEAR)) if ab.size else nan,
        f_crit=float(np.mean(ab > BETA_CRIT)) if ab.size else nan,
        dt=dt,
    )
