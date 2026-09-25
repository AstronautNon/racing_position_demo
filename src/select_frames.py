"""待标注帧选择：把"该标哪几帧"从人的直觉变成可复现的算法。

为什么需要它
------------
人工标注的预算有限（默认约 230 帧）。随机抽帧会把大部分额度浪费在
**朝向相同**的帧上 —— 而车身轴的多样性正是滑移角训练/评估最需要的东西
（直线、小角度、大角度、左右两个方向都要覆盖）。

做法：**贪心最远点采样**（farthest point sampling）。特征由四项拼成：

    朝向代理  [cos 2θ, sin 2θ]      θ 取掩膜 PCA 主轴
    画面位置  [cx/W, cy/H]          别把帧都堆在轨迹的同一段
    时间      [t/T]                 覆盖整段时长
    外观      48x48 归一化灰度       避免选出几乎一样的画面

距离定义为前四项的欧氏距离，再加上外观的 `1 - |余弦相似|`。
每轮选出"与已选集合最小距离"最大的候选帧。

关于朝向代理的一个诚实说明
--------------------------
掩膜 PCA 主轴的标准差普遍在 44°~83°（车辆阴影并进掩膜所致），
**它本身不能当朝向用**。但作为"选帧的多样性指标"仍然有效：
它在统计上确实与真实朝向相关，用它做分层的效果明显好于随机抽帧
（对比见输出的朝向分布直方图）。真正可用的朝向仍然只能来自人工标注。

用法：
    python -m src.select_frames --all            # 按登记的配额生成队列
    python -m src.select_frames --videos video02 --budget 20
产物：
    outputs/annotations/queue/<name>.csv         待标队列（含提示信息）
    outputs/annotations/crops/<name>/*.jpg       裁好的待标图（顺便预热标注台缓存）
    outputs/plots/<name>_annot_queue.png         队列预览图
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from . import crops as CR
from . import detect as D
from . import kinematics as K
from . import preprocess as P

FEAT_SCHEMA = 2


# ---------------------------------------------------------------------------
# 候选帧筛选
# ---------------------------------------------------------------------------
def candidate_frames(spec: C.VideoSpec, res: P.PreprocessResult,
                     dets: list[D.Detection], kin: dict) -> list[D.Detection]:
    """挑出"值得标、也标得准"的帧。

    五条门槛，每条都是为了不浪费标注额度：
      - 置信度低于该素材 25 分位 → 掩膜本身可疑，框未必在车上；
      - 车框太短（< 工作图宽的 3%）→ 点选误差会被放大，标了也不准；
      - 位置被判过离群 → 轨迹不可信，说明这一帧检测出了问题；
      - **车框宽高比 > ANNOT_MAX_BOX_ASPECT → 掩膜是运动月牙带而不是车**；
      - 车框超出画面太多（超过框长边的 ANNOT_MAX_OVERHANG）→ 车被截断，车身轴看不全。

    最后一条**不**用"距边缘留白"来判。检测框会被车身阴影撑大（实测 v15 框宽
    比车身核心宽 18%），框贴边往往只是阴影尾巴越界，车本身完整在画面内。
    按"距边 1%"筛会把这类帧全部误杀（v15 只剩 7 帧候选，v12 只剩 96 帧）；
    改用"越界幅度占框长边的比例"后，v15 恢复到 42 帧、v12 恢复到 186 帧。
    """
    if not dets:
        return []
    ww, wh = res.work_size
    conf = np.array([d.conf for d in dets])
    conf_min = float(np.quantile(conf, C.ANNOT_MIN_CONF_FRAC))
    box_min = C.ANNOT_MIN_BOX_FRAC * ww
    rejected = kin.get("rejected", np.zeros(res.n, dtype=bool))

    out = []
    for d in dets:
        if d.conf < conf_min:
            continue
        if max(d.w, d.h) < box_min:
            continue
        if 0 <= d.k < len(rejected) and rejected[d.k]:
            continue
        if not (np.isfinite(d.cx) and np.isfinite(d.cy)):
            continue
        # 宽高比超限 ⇒ 掩膜是"运动月牙带"而不是车（见 C.ANNOT_MAX_BOX_ASPECT）。
        # 放在越界判定之前：月牙带的框又长又低，越界与否已无意义。
        if max(d.w, d.h) > C.ANNOT_MAX_BOX_ASPECT * max(1.0, min(d.w, d.h)):
            continue
        x1, y1, x2, y2 = d.bbox
        overhang = max(0.0, -x1, -y1, x2 - ww, y2 - wh)
        if overhang > C.ANNOT_MAX_OVERHANG * max(d.w, d.h):
            continue
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# 外观特征（读一遍视频，结果缓存在 npz）
# ---------------------------------------------------------------------------
def _appearance(spec: C.VideoSpec, res: P.PreprocessResult,
                ks: list[int]) -> np.ndarray:
    cache = C.CACHE_DIR / f"annot_feat_{spec.name}.npz"
    want = np.array(ks, dtype=int)
    if cache.exists():
        try:
            z = np.load(cache)
            if int(z["schema"]) == FEAT_SCHEMA and int(z["n"]) == res.n \
                    and np.array_equal(z["ks"], want):
                return z["app"]
        except (KeyError, ValueError, OSError):
            pass

    size = C.ANNOT_APPEAR_SIZE
    need = set(ks)
    got: dict[int, np.ndarray] = {}
    for k, _, _, fr in P.iter_work_frames(res, spec.path):
        if k not in need:
            continue
        g = cv2.cvtColor(cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
        g -= g.mean()
        n = float(np.linalg.norm(g))
        got[k] = g / n if n > 1e-6 else g
    app = np.stack([got.get(k, np.zeros(size * size, np.float32)) for k in ks])
    np.savez_compressed(cache, schema=FEAT_SCHEMA, n=res.n, ks=want, app=app)
    return app


# ---------------------------------------------------------------------------
# 贪心最远点采样
# ---------------------------------------------------------------------------
def _fill_gaps(sel: list[int], cands: list[D.Detection], dist_to,
               max_gap: int) -> tuple[list[int], list[float], list[tuple[int, int]]]:
    """补上超过 max_gap 的空洞。

    返回 (新 sel, 新增帧的 gain, 无法补的空洞 [(k_a, k_b), ...])。

    为什么需要单独一步：最远点采样（farthest point sampling）只优化"朝向多样性"，
    完全不知道下游插值器有最大间隔约束。一段朝向变化平缓（但车子确实在动的）区间
    对"多样性"贡献小，会被整段跳掉 —— 而那正是需要锚点的地方。

    实测 video12：k 9→65 与 114→171 各断 56/57 帧，区间检出率 100%、
    帧间差异中位 1.49（其余区间 1.16，反而更大），且 k=9 的提示轴 177.7°
    到 k=65 的 58.0° 差了约 60° —— 信息确实丢了，不是"那里没内容"。

    补法：反复找当前**最大**的超限间隔，在它内部挑"离已选帧最远"的那一帧插入
    （与主循环同一准则），直到所有间隔 ≤ max_gap。

    预算处理：**覆盖优先**，允许结果超过 budget —— 静默留洞（下游 β 整段为空）
    比多标几帧危险得多。超出量由调用方报出来。
    区间内没有候选（检出中断）时无法补，记入第三项返回。
    """
    ks = np.array([d.k for d in cands])
    gains_add: list[float] = []
    blocked: list[tuple[int, int]] = []
    if max_gap <= 0:
        return sel, gains_add, blocked

    while True:
        order = sorted(sel, key=lambda i: ks[i])
        worst = None
        for a, b in zip(order[:-1], order[1:]):
            gap = int(ks[b] - ks[a])
            if gap <= max_gap or (int(ks[a]), int(ks[b])) in blocked:
                continue
            if worst is None or gap > worst[0]:
                worst = (gap, a, b)
        if worst is None:
            return sel, gains_add, blocked
        _, a, b = worst
        inside = [i for i in range(len(cands)) if ks[a] < ks[i] < ks[b]]
        if not inside:
            blocked.append((int(ks[a]), int(ks[b])))
            continue
        gains = [min(dist_to(i, j) for j in sel) for i in inside]
        best = inside[int(np.argmax(gains))]
        gains_add.append(float(max(gains)))
        sel = sel + [best]


def greedy_select(cands: list[D.Detection], res: P.PreprocessResult,
                  kin: dict, app: np.ndarray, budget: int,
                  *, max_gap: int | None = None, verbose: bool = False,
                  name: str = "") -> tuple[list[int], np.ndarray]:
    """返回（按选取顺序排列的候选下标, 每次选取时的最小距离）。

    max_gap 非 None 时，在最远点采样之后再跑一遍 `_fill_gaps` 补齐空洞
    （见该函数说明）。补进来的帧排在最后，prio 因此更大 —— 队列里"越靠后越次要"
    的既有语义不变，但**补进来的帧不能跳过**：它们就是用来堵洞的，跳过就白补了。
    """
    m = len(cands)
    budget = min(budget, m)
    if budget <= 0:
        return [], np.zeros(0)

    ww, wh = res.work_size
    T = max(1e-6, res.duration)
    axis = np.array([d.axis_deg for d in cands], dtype=float)
    ax = np.stack([np.cos(np.radians(2 * axis)), np.sin(np.radians(2 * axis))], 1) * C.ANNOT_W_AXIS
    pos = np.array([[d.cx / ww, d.cy / wh] for d in cands]) * C.ANNOT_W_POS
    tim = np.array([[d.t / T] for d in cands]) * C.ANNOT_W_TIME
    geo = np.concatenate([ax, pos, tim], axis=1)

    def dist_to(i: int, j: int) -> float:
        g = float(np.linalg.norm(geo[i] - geo[j]))
        a = 1.0 - abs(float(app[i] @ app[j]))
        return g + C.ANNOT_W_APPEAR * a

    # 起点取置信度最高的一帧：它是"最确信框对了"的那一帧，
    # 从它出发向外扩，最远点采样的结果对起点不敏感。
    conf = np.array([d.conf for d in cands])
    sel = [int(np.argmax(conf))]
    mind = np.array([dist_to(i, sel[0]) for i in range(m)])
    mind[sel[0]] = -1.0
    gains = [float("inf")]
    while len(sel) < budget:
        nxt = int(np.argmax(mind))
        if mind[nxt] <= 0:
            break
        gains.append(float(mind[nxt]))
        sel.append(nxt)
        for i in range(m):
            d = dist_to(i, nxt)
            if d < mind[i]:
                mind[i] = d

    n_diverse = len(sel)
    if max_gap is None:
        max_gap = C.ANNOT_SELECT_MAX_GAP
    sel, extra, blocked = _fill_gaps(sel, cands, dist_to, max_gap)
    gains.extend(extra)

    over = max(0, len(sel) - budget)
    if over and verbose:
        print(f"  [{name}] 覆盖优先：为满足最大间隔 {max_gap} 帧，"
              f"队列从配额 {budget} 扩到 {len(sel)} 帧（超 {over} 帧，请一并标注）")
    if blocked and verbose:
        print(f"  [{name}] 有 {len(blocked)} 个间隔区间内无候选（检出中断），无法补齐："
              + ", ".join(f"k{a}→k{b}" for a, b in blocked))
    if verbose and len(sel) > n_diverse:
        print(f"  [{name}] 最远点采样选出 {n_diverse} 帧，"
              f"覆盖修复补入 {len(sel) - n_diverse} 帧")
    return sel, np.array(gains)


# ---------------------------------------------------------------------------
# 队列产物
# ---------------------------------------------------------------------------
QUEUE_COLS = ["k", "frame", "t", "cx", "cy", "w", "h", "conf",
              "hint_axis_deg", "psi_vel_deg", "speed_blps", "prio", "img"]


def _queue_ks(name: str) -> list[int]:
    """读已有队列的 k 列表（不存在或读不了则返回空）。"""
    p = C.ANNOT_QUEUE_DIR / f"{name}.csv"
    if not p.exists():
        return []
    try:
        with p.open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.reader(f)
                    if r and not r[0].lstrip().startswith("#")]
    except OSError:
        return []
    out: list[int] = []
    for row in rows[1:]:                     # 跳过表头
        if not row:
            continue
        try:
            out.append(int(float(row[0])))
        except ValueError:
            continue
    return out


def _labeled_ks(name: str) -> set[int]:
    """已标注的 k 集合。直接读 CSV，不走 annotate，避免循环导入。"""
    p = C.ANNOT_DIR / f"{name}.csv"
    if not p.exists():
        return set()
    try:
        with p.open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.reader(f)
                    if r and not r[0].lstrip().startswith("#")]
    except OSError:
        return set()
    out: set[int] = set()
    for row in rows[1:]:
        if not row:
            continue
        try:
            out.add(int(float(row[0])))
        except ValueError:
            continue
    return out


def build_queue(name: str, budget: int | None = None, force: bool = False,
                make_sheet: bool = True, verbose: bool = True,
                rewrite: bool = False) -> dict:
    # 已完成标注的队列**冻结**，一帧都不动。
    #
    # 为什么需要这条：选帧参数（质量门槛、权重、配额）以后必然会调，每次调都可能
    # 选出不同的帧集。若照常重写，会出现"旧帧被挤出队列、队列里混进没标过的新帧"，
    # 于是 `--status` 把已完成的素材显示成未完成，还要人把同一段再标一遍 ——
    # 实测过一次：加了宽高比门槛后，video03 与 video12 各有 7 个已标帧被挤出队列，
    # 进度从 45/45、55/55 掉到 38/45、48/56（标注数据本身没丢，但队列不再对应人工作品）。
    # 已完成的人工成果不该被算法顺手改掉；要重选请显式 `--rewrite` 或删掉队列文件。
    prev_ks = _queue_ks(name)
    if prev_ks and not rewrite and set(prev_ks) <= _labeled_ks(name):
        if verbose:
            print(f"  {name}: 队列 {len(prev_ks)} 帧已全部标注 → 保持不变"
                  f"（要重选加 --rewrite）")
        return dict(name=name, n_cand=None, n_sel=len(prev_ks), frozen=True,
                    sel=prev_ks, queue=C.ANNOT_QUEUE_DIR / f"{name}.csv")

    spec = C.get(name)
    res = P.preprocess(spec)
    dets, _ = D.run_detection(spec, res)
    kin = K.compute(dets, res)
    cands = candidate_frames(spec, res, dets, kin)
    if not cands:
        if verbose:
            print(f"  {name}: 无满足门槛的候选帧")
        return dict(name=name, n_cand=0, n_sel=0)

    budget = C.annot_quota(name) if budget is None else budget
    app = _appearance(spec, res, [d.k for d in cands])
    sel, gains = greedy_select(cands, res, kin, app, budget,
                               verbose=verbose, name=name)
    n_greedy = len(sel)
    sel_cands = [cands[i] for i in sel]

    # 与旧队列取并集：**绝不因为采样参数变了就把旧帧丢掉**。
    # 旧帧里可能有已经人工标注的（哪怕这一帧没通过新门槛 —— 人能看见车、
    # 检测器看不见，正是需要人工的原因）。丢掉它等于把人的劳动判为无效，
    # 还会让 --status 把已标过的帧报成"未标"。
    # 只并"曾进过队列"的帧，不会把新门槛挡掉的候选混进来影响本次选择。
    have = {d.k for d in sel_cands}
    all_det = {d.k: d for d in dets}
    merged: list[D.Detection] = []
    for k in prev_ks:
        if k in have:
            continue
        d = all_det.get(k)
        if d is None:
            if verbose:
                print(f"  [{name}] 旧队列帧 k={k} 已无检出，无法并入（其标注仍供管线使用）")
            continue
        merged.append(d)
        have.add(k)
    if merged and verbose:
        print(f"  [{name}] 并入 {len(merged)} 个旧队列帧"
              f"（k={merged[0].k}…{merged[-1].k}），避免重复标注")
    sel_cands = sel_cands + merged
    # 贪心幅度曲线只对应"本次新选出"的部分：并集帧不是按该准则选出来的，
    # 混进去会让对数轴出现 log(0)。
    gains = gains[:n_greedy]

    qp = C.ANNOT_QUEUE_DIR / f"{name}.csv"
    with qp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"# {name} 待标注队列，由 python -m src.select_frames 生成"])
        w.writerow([f"# 共 {len(sel)} 帧（候选 {len(cands)}），按选取顺序排列；"
                    f"hint_axis_deg 是掩膜 PCA 主轴，仅作粗初值，不可当结果"])
        w.writerow([f"# 采样准则：最远点采样保证朝向多样性，再补帧保证相邻间隔"
                    f" ≤ {C.ANNOT_SELECT_MAX_GAP} 帧（插值上限 {C.ANNOT_MAX_GAP} 的 75%）。"
                    f"**补进来的帧排在队列末尾（prio 最大），但必须标** —— 跳过就留空洞"])
        w.writerow([f"# 质量门槛：置信度 > 25 分位、框长边 ≥ {C.ANNOT_MIN_BOX_FRAC:.0%} 工作图宽、"
                    f"宽高比 ≤ {C.ANNOT_MAX_BOX_ASPECT:.1f}（超过就是运动月牙带而不是车）、"
                    f"越界幅度 ≤ {C.ANNOT_MAX_OVERHANG:.0%} 框长边、位置未判离群"])
        if merged:
            w.writerow([f"# 末尾 {len(merged)} 帧是旧队列保留下来的（k="
                        f"{','.join(str(d.k) for d in merged)}）："
                        f"已标过的帧一律不丢弃"])
        w.writerow(QUEUE_COLS)
        for prio, d in enumerate(sel_cands):
            sp = kin["speed"][d.k]
            pv = kin["psi_vel"][d.k]
            w.writerow([
                d.k, d.frame, f"{d.t:.4f}", f"{d.cx:.2f}", f"{d.cy:.2f}",
                f"{d.w:.2f}", f"{d.h:.2f}", f"{d.conf:.4f}",
                f"{d.axis_deg:.2f}",
                "" if not np.isfinite(pv) else f"{pv:.2f}",
                "" if not np.isfinite(sp) else f"{sp / kin['size_px']:.3f}"
                if kin.get("size_px") else "",
                prio, f"crops/{name}/{d.k:05d}.jpg"])

    # 裁图一并生成，顺带把标注台的缓存预热掉
    views = []
    for d in sel_cands:
        try:
            views.append(CR.ensure(res, spec, d.k, d.cx, d.cy, d.w, d.h, force=force))
        except (IOError, ValueError):
            views.append(None)

    if make_sheet:
        sheet = queue_sheet(spec.name, res, sel_cands, views, gains,
                            C.PLOT_DIR / f"{name}_annot_queue.png",
                            all_cands=cands)
    else:
        sheet = None

    if verbose:
        print(f"  {name}: 候选 {len(cands)} / 检出 {len(dets)} → 选出 {len(sel)} 帧")
        if sheet:
            print(f"         预览图 {sheet.relative_to(C.ROOT)}")
    return dict(name=name, n_cand=len(cands), n_sel=len(sel),
                n_det=len(dets), sheet=sheet, queue=qp,
                sel=[d.k for d in sel_cands])


# ---------------------------------------------------------------------------
# 队列预览图
# ---------------------------------------------------------------------------
def queue_sheet(name: str, res: P.PreprocessResult, cands: list[D.Detection],
                views: list, gains: np.ndarray, path: Path,
                all_cands: list[D.Detection] | None = None,
                cols: int = 8, tile: int = 190) -> Path | None:
    """把队列画成一张联络图，并在顶部给出选帧前后的朝向分布对比。

    `cands` 是**被选中**的帧，`all_cands` 是全部候选（用于对照直方图）。
    两者对比能直接看出贪心采样是否真的把朝向覆盖拉平了。
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    n = len(cands)
    if n == 0:
        return None
    rows = int(np.ceil(n / cols))
    body_h = rows * (tile / 100.0)
    head = 2.4
    top_frac = 1.0 - head / (body_h + head)
    fig = plt.figure(figsize=(cols * 2.0, body_h + head))
    fig.suptitle(f"{name}  待标注队列（{n} 帧）  "
                 f"红框=检测框  黄虚线=掩膜 PCA 主轴（粗初值，仅供参照）",
                 fontsize=12, y=0.996)

    # 朝向覆盖对比：全部候选 vs 被选中
    ax = fig.add_axes([0.055, top_frac + 0.004, 0.40, (head - 0.75) / (body_h + head)])
    bins = np.arange(0, 181, 15)
    if all_cands:
        ax.hist(np.array([d.axis_deg for d in all_cands]), bins=bins, alpha=0.55,
                color="#B4B2A9", label=f"全部候选（{len(all_cands)}）")
    ax.hist(np.array([d.axis_deg for d in cands]), bins=bins, alpha=0.80,
            color="#1D9E75", label=f"被选中（{n}）")
    ax.set_title("朝向代理分布（PCA 主轴；只表示覆盖是否均匀，不是可用朝向）", fontsize=9)
    ax.set_xlabel("轴角 / 度（mod 180）", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right")

    # 贪心幅度：第一个点之后单调不增，说明采样已收敛到"覆盖最稀疏处"
    axg = fig.add_axes([0.555, top_frac + 0.004, 0.40, (head - 0.75) / (body_h + head)])
    g = np.array(gains[1:], dtype=float) if len(gains) > 1 else np.zeros(0)
    if len(g):
        axg.plot(np.arange(1, len(g) + 1), g, lw=1.4, color="#185FA5")
        axg.set_yscale("log")
    axg.set_title("贪心幅度（每次选取时的最小距离，对数轴）", fontsize=9)
    axg.set_xlabel("选取序号", fontsize=8)
    axg.tick_params(labelsize=7)
    axg.grid(alpha=0.25, which="both")

    gs = fig.add_gridspec(rows, cols, top=top_frac - 0.004, bottom=0.004,
                          left=0.006, right=0.994, hspace=0.16, wspace=0.05)
    for i, (d, v) in enumerate(zip(cands, views)):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ax.axis("off")
        if v is not None:
            img = cv2.imread(str(v.path))
            if img is not None:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                x1, y1 = v.work_to_disp(d.cx - d.w / 2, d.cy - d.h / 2)
                x2, y2 = v.work_to_disp(d.cx + d.w / 2, d.cy + d.h / 2)
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                           ec="#E24B4A", lw=1.0))
                p1, p2 = CR.axis_segment(d.cx, d.cy, d.axis_deg, max(d.w, d.h) * 0.95)
                a1 = v.work_to_disp(*p1)
                a2 = v.work_to_disp(*p2)
                ax.plot([a1[0], a2[0]], [a1[1], a2[1]], "--", color="#F2C230", lw=1.0)
                ax.set_xlim(0, v.iw)
                ax.set_ylim(v.ih, 0)
        ax.set_title(f"#{i}  k={d.k}", fontsize=7, pad=1.5)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成待标注帧队列")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--videos", nargs="+")
    g.add_argument("--all", action="store_true", help="所有登记为可标注的素材")
    ap.add_argument("--budget", type=int, default=None, help="覆盖登记的配额")
    ap.add_argument("--force", action="store_true", help="忽略裁图与特征缓存")
    ap.add_argument("--rewrite", action="store_true",
                    help="允许重写已完成标注的队列（默认冻结，保护人工成果）")
    ap.add_argument("--no-sheet", action="store_true")
    args = ap.parse_args(argv)

    specs = [C.get(v) for v in args.videos] if args.videos else C.annotation_videos()
    print(f"生成待标注队列：{len(specs)} 段素材")
    tot = 0
    for spec in specs:
        print(f"[{spec.name}]")
        try:
            r = build_queue(spec.name, budget=args.budget, force=args.force,
                            make_sheet=not args.no_sheet, rewrite=args.rewrite)
            tot += r.get("n_sel", 0)
        except Exception as e:
            print(f"  失败：{type(e).__name__}: {e}")
    print(f"\n合计 {tot} 帧待标注，队列在 {C.ANNOT_QUEUE_DIR.relative_to(C.ROOT)}/")
    print("下一步： python -m src.annotate --all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
