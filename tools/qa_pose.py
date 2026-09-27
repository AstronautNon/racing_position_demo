"""姿态质量审计（命令行入口）。

判据本体在 `src/qa_pose.py`，与流水线报告用的是**同一份实现** ——
这样报告里的数字和这里跑出来的永远一致，不会各说各话。

用法：
    python3 tools/qa_pose.py                 # 全部已登记素材
    python3 tools/qa_pose.py video15 video03 # 指定素材
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src import detect as D  # noqa: E402
from src import kinematics as K  # noqa: E402
from src import preprocess as P  # noqa: E402
from src.qa_pose import BETA_CRIT, BETA_NEAR, SPIKE_DEG_PER_FRAME, pose_metrics  # noqa: E402


def analyze(name: str, verbose: bool = True) -> dict:
    spec = C.get(name)
    res = P.preprocess(spec)
    dets, _ = D.run_detection(spec, res)
    ann = K.load_annotation_detail(spec.name, res.n)
    if ann is None:
        kin = K.compute(dets, res)
    else:
        kin = K.compute(dets, res, body_axis=ann["axis"], body_mask=ann["annotated"])

    m = pose_metrics(kin)
    st = K.annotate_stats(kin)
    m.update(name=name, stats=st, eff_fps=res.eff_fps)

    if verbose:
        dt = m["dt"]
        tag = "" if m["realtime"] else "   ⚠ 时间轴非实时（原片加速）"
        print(f"\n{'='*92}\n{name}   有效 {res.eff_fps:.1f} fps（一帧 {dt*1000:.0f} ms）   "
              f"轴来源：人工 {st['annotated']} / 插值 {st['interp']} / "
              f"PCA {st['pca']} / 无 {st['none']}{tag}\n{'='*92}")
        if m["resid"] is not None:
            note = "" if m["realtime"] else "（按名义 0.5 s 窗口，加速素材上会偏大）"
            print(f"  轴残差（去趋势后抖动）   {m['resid']:.2f}°{note}")
        else:
            print("  轴残差  —（样本不足）")
        if m["n_human"] >= 2:
            if m["realtime"]:
                print(f"  人工帧间变化率 中位 {m['rate_p50']:.2f}°/帧 = {m['rate_p50']/dt:.0f}°/s"
                      f"   p90 {m['rate_p90']:.2f}°/帧 = {m['rate_p90']/dt:.0f}°/s")
            else:
                print(f"  人工帧间变化率 中位 {m['rate_p50']:.2f}°/帧"
                      f"   p90 {m['rate_p90']:.2f}°/帧   最大 {m['rate_max']:.2f}°/帧"
                      f"   （°/s 是名义值，此处不列）")
        else:
            print("  人工帧太少，无法量帧间变化率")
        if not m["realtime"]:
            tc = m["top_change"]
            where = f"（k={tc['k0']}→{tc['k1']}）" if tc else ""
            print(f"  — 该素材时间轴非实时，帧间变化率**不作物理判定** —— 相邻帧真实间隔"
                  f"大于 1/25 s，帧间角度变化大属正常{where}。")
            print("    只报数值、不判对错；如需判定需先知道加速倍率。")
        elif m["spikes"]:
            print(f"  ⚠ 可疑跳变 {len(m['spikes'])} 处（>{SPIKE_DEG_PER_FRAME}°/帧），"
                  f"逐条复核（真自转 or 点选错误）：")
            for s in m["spikes"]:
                print(f"      k={s['k0']:>3}→{s['k1']:<3}（隔 {s['dk']} 帧） "
                      f"轴变 {s['deg']:5.1f}°  = {s['rate']:5.1f}°/帧 = {s['dps']:6.0f}°/s")
        elif m["n_human"] >= 2:
            print("  ✓ 人工帧之间没有超物理界的轴跳变")
        print(f"  |β|  中位 {m['b_med']:.1f}°  p90 {m['b_p90']:.1f}°  max {m['b_max']:.1f}°")
        print(f"       |β|>{BETA_NEAR:.0f}°：{100*m['f_near']:.0f}% 的帧     "
              f"|β|>{BETA_CRIT:.0f}°：{100*m['f_crit']:.1f}% 的帧   ← 定头不稳区")
    return m


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    names = argv or [s.name for s in C.annotation_videos()]
    rows = [analyze(n) for n in names]

    print(f"\n{'='*92}\n汇总（按「轴残差」降序）\n{'='*92}")
    print(f"{'素材':<9}{'人工帧':>7}{'轴残差':>9}{'变化率中位':>12}{'变化率p90':>11}"
          f"{'可疑跳变':>9}{'|β|中位':>9}{'|β|p90':>9}{'|β|>70°':>9}{'|β|>85°':>9}")
    for r in sorted(rows, key=lambda x: -(x["resid"] or 0)):
        resid = "—" if r["resid"] is None else f"{r['resid']:.1f}°"
        mark = "" if r["realtime"] else " ᵃ"
        r50 = "—" if not np.isfinite(r["rate_p50"]) else f"{r['rate_p50']:.2f}°/帧"
        r90 = "—" if not np.isfinite(r["rate_p90"]) else f"{r['rate_p90']:.2f}°/帧"
        nsp = f"—ᵃ" if not r["realtime"] else str(len(r["spikes"]))
        print(f"{r['name'] + mark:<9}{r['n_human']:>7}{resid:>9}{r50:>12}{r90:>11}"
              f"{nsp:>9}{r['b_med']:>8.1f}°{r['b_p90']:>8.1f}°"
              f"{100*r['f_near']:>8.0f}%{100*r['f_crit']:>8.1f}%")
    print("\n  「轴残差」= 去掉 0.5 s 平滑趋势后的抖动（度），衡量轴序列稳不稳；")
    print(f"  「可疑跳变」= 相邻已标帧之间轴变 >{SPIKE_DEG_PER_FRAME:.0f}°/帧"
          f"（≈500°/s，远超赛车真实自转）。")
    nrt = [r for r in rows if not r["realtime"]]
    if nrt:
        print(f"\n  ᵃ 标记的素材时间轴**非实时**（原片经过加速/缩时处理）：相邻两帧真实间隔"
              f"大于名义的 1/25 s，")
        print("     帧间角度变化大属正常，故**不作物理判定**；其 t 与所有 °/s 也只是名义值。")
        for r in nrt:
            tc = r["top_change"]
            where = f"（最大处 k={tc['k0']}→{tc['k1']}，{tc['rate']:.1f}°/帧）" if tc else ""
            print(f"     {r['name']}：只作参考{where}")
    bad = [r for r in rows if r["spikes"]]
    if bad:
        print(f"\n  ⚠ 有 {len(bad)} 段素材存在可疑跳变，见上方逐条明细 ——")
        for r in bad:
            ks = "、".join(f"k={s['k0']}" for s in r["spikes"])
            print(f"      {r['name']}：{ks}（用标注台的「跳到 k」直接跳过去复核）")
    crit = [r for r in rows if r["f_crit"] > 0.05]
    if crit:
        print(f"  ⚠ 有 {len(crit)} 段素材 |β|>85° 的帧超过 5%：" +
              "、".join(f"{r['name']} {100*r['f_crit']:.0f}%" for r in crit) +
              "\n     这些帧的车头朝向由运动方向推定、|β| 已顶到 90 上限，需人眼确认。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
