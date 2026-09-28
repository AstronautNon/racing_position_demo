"""盲标抽检（独立测试集）：抽帧、评分。

为什么必须有这个
----------------
项目现在的 175 帧人工标注**同时**承担两个角色：调参的依据、报精度的依据。
这种情况下无论数字多好看都不能信 —— 它是在用同一批数据自证。
所以验收标准 S3 写的是"与人工**盲标**对比"（项目规划 §1），风险表里也写明
"只用盲标 MAE 汇报"（§10）。

做法：从**标注队列之外**另抽一批帧，让标注者不看任何提示地独立标一遍，
再与系统交付的输出（`outputs/tracks/<名>.csv` 的 `axis_deg`）比。

量到的是什么
------------
系统交付的车身轴是「稀疏人工标注 + 插值」拼出来的，所以这个 MAE 衡量的是
**端到端误差**：既含"人点选"的随机误差，也含"帧与帧之间靠插值"的误差。
这正是下游 β 实际承受的误差，比"重标同一批训练帧"诚实得多。

一条纪律
--------
抽出来的帧**一次定死**，中途不许换。换采样就等于偷看答案；
采样规则与帧号都记进 MANIFEST，便于复核。
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from . import annotate as A
from . import config as C
from . import detect as D
from . import kinematics as K
from . import preprocess as P
from . import select_frames as S


# ---------------------------------------------------------------------------
# 抽帧
# ---------------------------------------------------------------------------
def _cands(name: str) -> tuple[P.PreprocessResult, list[int], dict]:
    """返回 (预处理结果, 候选帧号, {k: 检出})。

    候选 = 检出帧中**不在已有标注队列、也不在已标集合**的帧。
    两道都排（队列 + 已标）是防呆：队列曾被人为改过，两个集合不一定一致。
    """
    spec = C.get(name)
    res = P.preprocess(spec)
    dets, _ = D.run_detection(spec, res)
    banned = set(S._queue_ks(name)) | S._labeled_ks(name)
    byk = {d.k: d for d in dets}
    cands = [d.k for d in dets if d.k not in banned]
    return res, cands, byk


def _alloc(n_total: int, weights: dict[str, int], min_each: int = 2) -> dict[str, int]:
    """按各素材的交付帧数比例分配抽帧数，用最大余数法凑到正好 n_total。"""
    names = sorted(weights)
    tot = sum(weights.values()) or 1
    raw = {n: n_total * weights[n] / tot for n in names}
    out = {n: max(min_each, int(np.floor(raw[n]))) for n in names}
    while sum(out.values()) > n_total:                 # 超了就削最"占便宜"的
        n = max(out, key=lambda x: out[x] - raw[x])
        if out[n] <= min_each:
            break
        out[n] -= 1
    order = sorted(names, key=lambda x: -(raw[x] - np.floor(raw[x])))
    i = 0
    while sum(out.values()) < n_total and order:        # 不够就按余数补
        out[order[i % len(order)]] += 1
        i += 1
    return out


def sample(cands: list[int], n_target: int) -> list[int]:
    """按 k 均匀铺开取 n_target 帧。

    **不用随机数**：均匀铺开更稳、可复现，且天然覆盖整段的姿态变化区间
    （随机抽样在小样本下容易全挤在一段里）。
    """
    if len(cands) <= n_target:
        return list(cands)
    idx = np.linspace(0, len(cands) - 1, n_target).round().astype(int)
    return sorted({cands[i] for i in idx})


def build(name: str, ks: list[int],
          res: P.PreprocessResult | None = None, byk: dict | None = None) -> Path:
    """把抽出的帧写成**与普通队列同格式**的 CSV，供标注台直接读。

    提示列与运动方向列**留空**：前者是系统自己的估计（给出来就是送答案），
    后者能被反推成"β 大概多少"（同属泄漏，见 `annotate.build_payload` 的说明）。
    """
    if res is None or byk is None:
        res, _, byk = _cands(name)
    out = A.blind_queue_path(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"# {name} 盲标抽检队列（独立测试集，不参与调参），"
                    f"由 tools/build_blind.py 生成"])
        w.writerow(["# 抽样规则：从已有标注队列**之外**的检出帧里按 k 均匀取 N 帧；"
                    "hint_axis_deg 与 psi_vel_deg 一律留空（盲标：不给任何提示）"])
        w.writerow(S.QUEUE_COLS)
        for k in ks:
            d = byk.get(k)
            if d is None:
                continue
            w.writerow([k, res.kept_frames[k], f"{res.t[k]:.4f}",
                        f"{d.cx:.2f}", f"{d.cy:.2f}", f"{d.w:.2f}", f"{d.h:.2f}",
                        f"{d.conf:.4f}", "", "", "", 0, f"crops/{name}/{k:05d}.jpg"])
    return out


def build_all(n_total: int = 30, seed: int = 20260928, verbose: bool = True) -> dict:
    weights = {s.name: len(A.read_labels(s.name)) for s in C.annotation_videos()}
    weights = {k: v for k, v in weights.items() if v}      # 只给已标注的素材抽
    quota = _alloc(n_total, weights) if weights else {}
    rows = []
    for name in sorted(quota):
        res, cands, byk = _cands(name)
        ks = sample(cands, quota[name])
        build(name, ks, res, byk)
        rows.append(dict(name=name, ks=ks, n_cand=len(cands), target=quota[name]))
        if verbose:
            print(f"  {name}: 目标 {quota[name]} 帧 → 实抽 {len(ks)} 帧"
                  f"（队列外候选 {len(cands)}）")
    mp = _write_manifest(rows, n_total, seed)
    if verbose:
        print(f"\n合计抽 {sum(len(r['ks']) for r in rows)} 帧；"
              f"记录见 {mp.relative_to(C.ROOT)}")
        print("开始盲标：")
        print("  /opt/anaconda3/bin/python3 -m src.annotate --serve --blind")
    return dict(rows=rows, manifest=mp)


def _write_manifest(rows: list[dict], n_total: int, seed: int) -> Path:
    p = C.ANNOT_BLIND_DIR / "MANIFEST.md"
    L = ["# 盲标抽检抽样记录（独立测试集）", "",
         f"生成时间：{datetime.now():%Y-%m-%d %H:%M}；随机种子 {seed}"
         "（当前为定序采样，种子只作留痕）。", "",
         "**这批帧不参与任何调参**，只用于报精度。抽定后不得更换。", "",
         "| 素材 | 抽帧数 | 队列外候选 | 帧号 k |", "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {len(r['ks'])} | {r['n_cand']} | "
                 + "、".join(str(k) for k in r["ks"]) + " |")
    L += ["", f"合计 **{sum(len(r['ks']) for r in rows)}** 帧（目标 {n_total}）。", "",
          "抽帧规则：候选 = 该素材检出帧中**不在已有标注队列、也不在已标集合**里的，",
          "按 k 升序用 `numpy.linspace` 均匀取 N 帧，因此覆盖整段的姿态变化区间。", ""]
    p.write_text("\n".join(L), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 评分
# ---------------------------------------------------------------------------
def _num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _read_track_axis(name: str) -> dict[int, dict]:
    """读系统交付的逐帧输出（`outputs/tracks/<名>.csv`）。"""
    p = C.TRACK_DIR / f"{name}.csv"
    out: dict[int, dict] = {}
    if not p.exists():
        return out
    with p.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r and not r["k"].startswith("#")]
    for r in rows:
        try:
            out[int(r["k"])] = dict(axis=float(r["axis_deg"]),
                                    src=r.get("axis_src", ""),
                                    beta=_num(r.get("beta_deg")),
                                    psi_vel=_num(r.get("psi_vel_deg")))
        except (ValueError, KeyError):
            continue
    return out


def has_blind(name: str) -> bool:
    return A.blind_labels_path(name).exists()


def score(verbose: bool = True) -> dict:
    """盲标轴 vs 系统交付轴的误差。返回汇总 dict，并写一份 markdown。"""
    per: list[dict] = []
    all_err: list[float] = []
    for spec in C.annotation_videos():
        if not has_blind(spec.name):
            continue
        lb = A.read_blind_labels(spec.name)
        track = _read_track_axis(spec.name)
        errs, ks, srcs = [], [], []
        for k, rec in sorted(lb.items()):
            t = track.get(k)
            if not t or not np.isfinite(t["axis"]):
                continue
            e = float(K.axis_dist(rec["axis_deg"], t["axis"]))
            if np.isfinite(e):
                errs.append(e)
                ks.append(k)
                srcs.append(t["src"])
        if not errs:
            continue
        a = np.asarray(errs)
        per.append(dict(name=spec.name, n=int(a.size), mae=float(a.mean()),
                        med=float(np.median(a)), p90=float(np.percentile(a, 90)),
                        mx=float(a.max()), ks=ks, srcs=srcs))
        all_err.extend(errs)

    pooled = np.asarray(all_err) if all_err else np.asarray([])
    summary = dict(
        per=per, n=int(pooled.size),
        mae=float(pooled.mean()) if pooled.size else float("nan"),
        med=float(np.median(pooled)) if pooled.size else float("nan"),
        p90=float(np.percentile(pooled, 90)) if pooled.size else float("nan"),
        mx=float(pooled.max()) if pooled.size else float("nan"))
    if verbose:
        _print_summary(summary)
    summary["report"] = _write_report(summary)
    return summary


def _print_summary(s: dict) -> None:
    if not s["per"]:
        print("还没有盲标标注。两步：")
        print("  1) 抽帧： /opt/anaconda3/bin/python3 tools/build_blind.py")
        print("  2) 标注： /opt/anaconda3/bin/python3 -m src.annotate --serve --blind")
        return
    print(f"{'素材':<10}{'帧数':>6}{'MAE':>9}{'中位':>9}{'p90':>9}{'最大':>9}"
          f"  系统轴来源")
    for r in s["per"]:
        src: dict[str, int] = {}
        for v in r["srcs"]:
            src[v] = src.get(v, 0) + 1
        print(f"{r['name']:<10}{r['n']:>6}{r['mae']:>8.2f}°{r['med']:>8.2f}°"
              f"{r['p90']:>8.2f}°{r['mx']:>8.2f}°  "
              + "/".join(f"{k} {v}" for k, v in sorted(src.items())))
    print(f"{'合计':<10}{s['n']:>6}{s['mae']:>8.2f}°{s['med']:>8.2f}°"
          f"{s['p90']:>8.2f}°{s['mx']:>8.2f}°")
    print("\n  用 axis_dist（折 90° 的无向轴距离）度量。"
          "参照：人工点选自身的残差约 1.8°。")


def _write_report(s: dict) -> Path:
    p = C.REPORT_DIR / "D8_盲标精度.md"
    L = ["# D8 盲标精度：系统交付 vs 独立盲标", "",
         "盲标帧从**已有标注队列之外**抽取；标注时不给任何提示、也不给运动方向；",
         "**不参与任何调参**（采样记录见 `outputs/annotations/blind/MANIFEST.md`）。", ""]
    if not s["per"]:
        L += ["**尚未完成盲标**，本表为空。", "",
              "抽帧： `python tools/build_blind.py`；"
              "标注： `python -m src.annotate --serve --blind`。", ""]
        p.write_text("\n".join(L), encoding="utf-8")
        return p
    L += ["## 误差（车身轴，度数）", "",
          "| 素材 | 盲标帧 | MAE | 中位 | p90 | 最大 | 系统轴来源 |",
          "|---|---|---|---|---|---|---|"]
    for r in s["per"]:
        src: dict[str, int] = {}
        for v in r["srcs"]:
            src[v] = src.get(v, 0) + 1
        L.append(f"| {r['name']} | {r['n']} | **{r['mae']:.2f}°** | {r['med']:.2f}° | "
                 f"{r['p90']:.2f}° | {r['mx']:.2f}° | "
                 + "、".join(f"{k} {v}" for k, v in sorted(src.items())) + " |")
    L.append(f"| **合计** | **{s['n']}** | **{s['mae']:.2f}°** | {s['med']:.2f}° | "
             f"{s['p90']:.2f}° | {s['mx']:.2f}° | — |")
    L += ["", "## 怎么读这个数", "",
          "- 「系统轴来源」是系统交付值在该帧的来源：`annotated`（正好有人工标注）、",
          "  `interp`（由附近标注插值而来）。**MAE 主要由插值帧决定** ——",
          "  它是「稀疏标注 + 插值」这套方案的端到端误差，",
          "  也正是下游 β 实际承受的误差。",
          "- 参照：人工点选自身的残差约 1.8°。盲标 MAE 若明显高于它，",
          "  说明误差主要来自**插值**而非点选 —— 那就该加密标注而不是换模型。",
          "- 本表**不含**已搁置的 `video12`，也不含未标注的 video07/08。", ""]
    p.write_text("\n".join(L), encoding="utf-8")
    return p
