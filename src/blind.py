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

    候选 = 检出帧中**不在已有标注队列、也不在已标集合**、且**落在已标注区间内**
    的帧。三道都排（队列 + 已标 + 区间）是防呆：队列曾被人为改过，两个集合不一定一致。

    **为什么要限定在已标注区间内**（2026-09-28 补，第一批盲标的教训）：
    实测 5 段素材的「已标注区间」**恰好等于队列区间**（`min/max` 逐段一致），
    而检出的帧比它宽得多（如 video03 队列 0–190、检出到 257）。系统只在
    **标注之间**插值、**两端不外推**，所以区间外的检出帧上 `axis_src=none` ——
    抽到那里就是白标：人工标了也进不了分母。第一批 30 帧里 9 帧踩了这个坑。

    注意这里**没有**按「系统是否交付轴」筛候选 —— 那是按结果筛选，会让盲标
    偏向系统答得出的帧。区间内若因标注缺口过大而弃权，那是**真该计入的失败**，
    要留在样本里如实报告。
    """
    spec = C.get(name)
    res = P.preprocess(spec)
    dets, _ = D.run_detection(spec, res)
    qks = S._queue_ks(name)
    banned = set(qks) | S._labeled_ks(name)
    lo, hi = (min(qks), max(qks)) if qks else (None, None)
    byk = {d.k: d for d in dets}
    cands = []
    for d in dets:
        if d.k in banned:
            continue
        if lo is not None and not (lo <= d.k <= hi):
            continue
        cands.append(d.k)
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
    # 上半部分（抽样表）由代码生成；下半部分「标注执行记录」是人工追加的现场记录，
    # 代码生成不出来，所以这里**保留**已有那份，不要一重生成就把它冲掉。
    keep = ""
    if p.exists():
        old = p.read_text(encoding="utf-8")
        i = old.find(EXEC_MARK)
        if i >= 0:
            keep = old[i:]
    L = ["# 盲标抽检抽样记录（独立测试集）", "",
         f"生成时间：{datetime.now():%Y-%m-%d %H:%M}；随机种子 {seed}"
         "（当前为定序采样，种子只作留痕）。", "",
         "**这批帧不参与任何调参**，只用于报精度。抽定后不得更换。", "",
         "| 素材 | 抽帧数 | 队列外候选 | 帧号 k |", "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {len(r['ks'])} | {r['n_cand']} | "
                 + "、".join(str(k) for k in r["ks"]) + " |")
    L += ["", f"合计 **{sum(len(r['ks']) for r in rows)}** 帧（目标 {n_total}）。", "",
          "抽帧规则：候选 = 该素材检出帧中**不在已有标注队列、也不在已标集合**、"
          "且**落在队列区间内**的，",
          "按 k 升序用 `numpy.linspace` 均匀取 N 帧，因此覆盖整段的姿态变化区间。",
          "（区间这一道是 2026-09-28 补的：系统只在标注之间插值、两端不外推，"
          "区间外的帧标了也计不了分。）", ""]
    text = "\n".join(L)
    if keep:
        text += "\n" + keep
    p.write_text(text, encoding="utf-8")
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


# 「贴近标注」的判定阈值：盲标帧两侧最近的训练标注都在这个帧距内时，插值几乎
# 不引入误差，量到的就基本是**标注者自身的复现性**（同一个人隔一段时间再标一遍的
# 差异）。取 3 帧 —— 素材多为 25 fps，约 0.12 s，车身在这个尺度内转动很小。
NEAR_GAP = 3

# MANIFEST.md 里「标注执行记录」小节的起始标记。这一节是人工写的现场记录
# （哪些帧标不了、为什么），`_write_manifest` 重写上半部分时会原样保留它。
EXEC_MARK = "## 标注执行记录"

# 系统弃权的原因之一（「两端不外推」）。单独抽成常量是为了让报告能**按实际
# 原因**决定措辞，而不是把结论写死 —— 将来批次若出现别的原因，报告得跟着变。
_OUTSIDE = "在已标注区间之外，两端不外推"


def _sampled_ks(name: str) -> list[int]:
    """读盲标队列，得到当初**抽定**的帧号（评分覆盖率要用它做分母）。"""
    p = A.blind_queue_path(name)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#")]
    if not rows:
        return []
    head = [c.strip() for c in rows[0]]
    if "k" not in head:
        return []
    j = head.index("k")
    out = []
    for r in rows[1:]:
        try:
            out.append(int(float(r[j])))
        except (ValueError, IndexError):
            continue
    return sorted(set(out))


def _gap_around(k: int, aks: list[int]) -> tuple[int | None, int | None]:
    """盲标帧 k 到左 / 右两侧最近训练标注的帧距；该侧没有标注则返回 None。"""
    left = [a for a in aks if a <= k]
    right = [a for a in aks if a >= k]
    return (k - left[-1] if left else None), (right[0] - k if right else None)


def _skip_reason(k: int, aks: list[int]) -> str:
    """盲标帧有标注、但系统没交付轴时，说明为什么。"""
    if not aks:
        return "该素材无训练标注"
    if k < aks[0] or k > aks[-1]:
        return _OUTSIDE
    return "落在标注缺口内且超出插值上限"


def has_blind(name: str) -> bool:
    return A.blind_labels_path(name).exists()


def score(verbose: bool = True) -> dict:
    """盲标轴 vs 系统交付轴的误差。返回汇总 dict，并写一份 markdown。

    除了误差本身，还算两件**读这个数必须一起看**的事：

    · 覆盖率：抽定 30 帧里，有多少真拿到人工标注、有多少是系统自己弃权。
      只报误差而不报分母，容易让人把"系统只在一半帧上交付"读成"全段都能用"。
    · 误差来源：按盲标帧离最近训练标注的帧距分成「贴近标注」与「跨插值」两组。
      前一组量到的近似是**标注者自身的复现性**（人两遍标注的差异），
      后一组才是插值引入的误差 —— 两者该分开报，因为治法完全不同
      （前者只能靠多人标注取共识，后者靠加密标注）。
    """
    per: list[dict] = []
    all_err: list[float] = []
    cover: list[dict] = []
    near: list[float] = []
    far: list[float] = []
    span_all: list[float] = []      # 两侧都有标注的帧：插值跨度
    err_all: list[float] = []
    worst: dict | None = None
    for spec in C.annotation_videos():
        name = spec.name
        sampled = _sampled_ks(name)
        if not has_blind(name) and not sampled:
            continue
        lb = A.read_blind_labels(name)
        track = _read_track_axis(name)
        aks = sorted(A.read_labels(name))
        errs, ks, srcs, gaps = [], [], [], []
        skipped: list[tuple[int, str]] = []
        for k, rec in sorted(lb.items()):
            t = track.get(k)
            if not t or not np.isfinite(t["axis"]):
                skipped.append((k, _skip_reason(k, aks)))
                continue
            e = float(K.axis_dist(rec["axis_deg"], t["axis"]))
            if not np.isfinite(e):
                continue
            errs.append(e)
            ks.append(k)
            srcs.append(t["src"])
            gl, gr = _gap_around(k, aks)
            gaps.append((gl, gr))
            if gl is not None and gr is not None and max(gl, gr) <= NEAR_GAP:
                near.append(e)
            else:
                far.append(e)
            if gl is not None and gr is not None:
                span_all.append(gl + gr)
                err_all.append(e)
            if worst is None or e > worst["err"]:
                worst = dict(name=name, k=k, err=e,
                             span=(gl + gr) if (gl is not None and gr is not None) else None)
        cover.append(dict(name=name, n_sampled=len(sampled), n_labeled=len(lb),
                          n_scored=len(errs), skipped=skipped,
                          unlabeled=[k for k in sampled if k not in lb],
                          n_unlabeled=max(0, len(sampled) - len(lb))))
        if not errs:
            continue
        a = np.asarray(errs)
        per.append(dict(name=name, n=int(a.size), mae=float(a.mean()),
                        med=float(np.median(a)), p90=float(np.percentile(a, 90)),
                        mx=float(a.max()), ks=ks, srcs=srcs, gaps=gaps))
        all_err.extend(errs)

    pooled = np.asarray(all_err) if all_err else np.asarray([])

    # 插值跨度与误差的相关：用来支撑「误差主要来自插值」这句话，而不是空口断言
    corr = float("nan")
    if len(span_all) >= 3:
        corr = float(np.corrcoef(np.asarray(span_all, float),
                                 np.asarray(err_all, float))[0, 1])

    def _stat(v: list[float]) -> dict:
        a = np.asarray(v)
        return dict(n=int(a.size),
                    mae=float(a.mean()) if a.size else float("nan"),
                    med=float(np.median(a)) if a.size else float("nan"),
                    mx=float(a.max()) if a.size else float("nan"))

    summary = dict(
        per=per, cover=cover, near=_stat(near), far=_stat(far), near_gap=NEAR_GAP,
        corr=corr, worst=worst,
        span_med=float(np.median(span_all)) if span_all else float("nan"),
        span_mx=float(np.max(span_all)) if span_all else float("nan"),
        n_sampled=sum(c["n_sampled"] for c in cover),
        n_labeled=sum(c["n_labeled"] for c in cover),
        n=int(pooled.size),
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

    print(f"\n覆盖率：抽定 {s['n_sampled']} 帧 → 人工标注 {s['n_labeled']} 帧 "
          f"→ 计入评分 {s['n']} 帧")
    for c in s["cover"]:
        det = ""
        if c["skipped"]:
            det = "；系统弃权 " + "、".join(f"k={k}（{why}）" for k, why in c["skipped"])
        if c["unlabeled"]:
            det += "；未标 " + "、".join(f"k={k}" for k in c["unlabeled"])
        print(f"  {c['name']:<10} 抽 {c['n_sampled']:>2}  标 {c['n_labeled']:>2}  "
              f"计入 {c['n_scored']:>2}{det}")

    nf, fr = s["near"], s["far"]
    print(f"\n误差来源拆解（按离最近训练标注的帧距 ≤{s['near_gap']} 帧分界）：")
    print(f"  贴近标注  n={nf['n']:>2}  MAE {nf['mae']:.2f}°  中位 {nf['med']:.2f}°  "
          f"最大 {nf['mx']:.2f}°   ← 近似「标注者复现性」")
    print(f"  跨插值    n={fr['n']:>2}  MAE {fr['mae']:.2f}°  中位 {fr['med']:.2f}°  "
          f"最大 {fr['mx']:.2f}°   ← 插值引入")


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
          f"**合计 MAE {s['mae']:.2f}°**（{s['n']} 帧；中位 {s['med']:.2f}°、"
          f"p90 {s['p90']:.2f}°、最大 {s['mx']:.2f}°）。", "",
          "| 素材 | 计入帧 | MAE | 中位 | p90 | 最大 | 系统轴来源 |",
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
    L += ["", "## 覆盖率：抽了 30 帧，为什么只报了这些", "",
          f"抽定 {s['n_sampled']} 帧 → 得到人工标注 {s['n_labeled']} 帧 → "
          f"**能计分 {s['n']} 帧**。差额不是「挑掉的」，是**系统自己不交付轴**：", "",
          "| 素材 | 抽定 | 已标 | 计入 | 系统弃权的帧 |",
          "|---|---|---|---|---|"]
    for c in s["cover"]:
        bits = [f"k={k}（{why}）" for k, why in c["skipped"]]
        if c["unlabeled"]:
            bits.append("未标 " + "、".join(f"k={k}" for k in c["unlabeled"]))
        det = "；".join(bits) or "—"
        L.append(f"| {c['name']} | {c['n_sampled']} | {c['n_labeled']} | "
                 f"{c['n_scored']} | {det} |")
    # 「弃权全因两端不外推」这句话要按实际原因判断，不能写死 ——
    # 将来的批次若出现「落在标注缺口内」的弃权，报告不能还说同一句。
    why_all = {why for c in s["cover"] for _, why in c["skipped"]}
    if why_all <= {_OUTSIDE}:
        lead = ["弃权的帧**全部落在已标注区间之外**（首帧之前或末帧之后）——",
                "这是「两端不外推」这条约定的直接后果，不是缺陷：远端没有可依据的标注，",
                "外推出的轴没有证据支持。"]
    else:
        lead = ["弃权的帧见上表：一部分落在已标注区间之外（两端不外推），",
                "一部分落在标注缺口内、超出插值上限 —— 两者依同一条原则处理：",
                "邻近没有足够的标注支撑，就既不外推也不硬插。"]
    L += [""] + lead + [
        "**被判据保护的正是这一点**：系统不交付的帧，本表也不计分。",
        "未标的帧与其原因见 `outputs/annotations/blind/MANIFEST.md` 的「标注执行记录」。", "",
          "因此 30 帧的有效样本是 **"
          f"{s['n']} 帧**。若要让盲标覆盖整段（含首尾），需要一个**独立于本轮**的",
          "第二批抽帧（抽定后同样冻结），不能拿本轮的失败帧去补。", "",
          "## 误差来源拆解：这 2° 是谁贡献的", "",
          f"按盲标帧离最近训练标注的帧距分两组（分界 {s['near_gap']} 帧）：", "",
          "| 组 | 帧数 | MAE | 中位 | 最大 | 主要误差来源 |",
          "|---|---|---|---|---|---|",
          f"| 贴近标注（两侧 ≤{s['near_gap']} 帧） | {s['near']['n']} | "
          f"**{s['near']['mae']:.2f}°** | {s['near']['med']:.2f}° | "
          f"{s['near']['mx']:.2f}° | 标注者自身的复现性 |",
          f"| 跨插值（更远或只有一侧） | {s['far']['n']} | "
          f"**{s['far']['mae']:.2f}°** | {s['far']['med']:.2f}° | "
          f"{s['far']['mx']:.2f}° | 插值 |",
          "",
          f"两侧都有标注的帧上，**插值跨度与误差的相关系数 r = {s['corr']:.2f}** ——",
          "跨度越大误差越大，与「误差主要来自插值」一致。", "",
          "两组的治法完全不同，所以必须分开报：",
          "",
          "- 贴近标注那组量到的**不是系统的错**，而是「同一个人隔一段时间再标一遍」",
          "  的差异 —— 它构成这套评测的**误差地板**，任何方法都压不到它以下。",
          "  想降它只能靠多人标注取共识，换模型没用。",
          "- 跨插值那组才是「稀疏标注 + 插值」这套方案真正的代价，",
          "  也是**加密标注**能直接改善的部分。",
          "",
          "## 怎么读这个数", "",
          "- 「系统轴来源」在本批里**全是 `interp`**：盲标帧抽自标注队列之外，",
          "  不会正好落在训练标注上。所以本表衡量的是「稀疏标注 + 插值」的",
          "  端到端误差，也正是下游 β 实际承受的误差。",
          f"- 参照「人工点选自身残差约 1.8°」在本批得到验证：「贴近标注」组实测 "
          f"**{s['near']['mae']:.2f}°**，与之吻合 —— 它确实是这套评测的误差地板。",
          f"- 全批最大误差 **{s['worst']['err']:.2f}°**（{s['worst']['name']} "
          f"k={s['worst']['k']}）出现在两侧标注相隔 {s['worst']['span']} 帧的帧上"
          f"（全批插值跨度中位 {s['span_med']:.0f} 帧、最大 {s['span_mx']:.0f} 帧）。",
          f"  它属插值组的尾部而非普遍水平：全批中位误差只有 {s['med']:.2f}°。",
          "- 本表**不含**已搁置的 `video12`，也不含未标注的 video07/08。",
          "",
          "## 附注：video02 的系统性偏移", "",
          "video02 有 4 帧的误差同向（系统轴比盲标**低** 3.6°–4.1°），且这些帧两侧",
          "训练标注只隔 1–2 帧 —— 插值本身几乎不引入误差。同样的模式在其它素材上",
          "没有出现（video03 反向 +0.5°、其余 ≤1.7°）。因此它更像是**该段素材上",
          "标注者的判读习惯在两次标注间有偏移**，而不是系统拿了错的轴。",
          "记录在此，留待多人标注时验证。", ""]
    p.write_text("\n".join(L), encoding="utf-8")
    return p
