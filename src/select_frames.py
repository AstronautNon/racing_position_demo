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

    四条门槛，每条都是为了不浪费标注额度：
      - 置信度低于该素材 25 分位 → 掩膜本身可疑，框未必在车上；
      - 车框太短（< 工作图宽的 3%）→ 点选误差会被放大，标了也不准；
      - 位置被判过离群 → 轨迹不可信，说明这一帧检测出了问题；
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
def greedy_select(cands: list[D.Detection], res: P.PreprocessResult,
                  kin: dict, app: np.ndarray, budget: int) -> tuple[list[int], np.ndarray]:
    """返回（按选取顺序排列的候选下标, 每次选取时的最小距离）。"""
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
    return sel, np.array(gains)


# ---------------------------------------------------------------------------
# 队列产物
# ---------------------------------------------------------------------------
QUEUE_COLS = ["k", "frame", "t", "cx", "cy", "w", "h", "conf",
              "hint_axis_deg", "psi_vel_deg", "speed_blps", "prio", "img"]


def build_queue(name: str, budget: int | None = None, force: bool = False,
                make_sheet: bool = True, verbose: bool = True) -> dict:
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
    sel, gains = greedy_select(cands, res, kin, app, budget)
    sel_cands = [cands[i] for i in sel]

    qp = C.ANNOT_QUEUE_DIR / f"{name}.csv"
    with qp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"# {name} 待标注队列，由 python -m src.select_frames 生成"])
        w.writerow([f"# 共 {len(sel)} 帧（候选 {len(cands)}），按选取顺序排列；"
                    f"hint_axis_deg 是掩膜 PCA 主轴，仅作粗初值，不可当结果"])
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
    ap.add_argument("--no-sheet", action="store_true")
    args = ap.parse_args(argv)

    specs = [C.get(v) for v in args.videos] if args.videos else C.annotation_videos()
    print(f"生成待标注队列：{len(specs)} 段素材")
    tot = 0
    for spec in specs:
        print(f"[{spec.name}]")
        try:
            r = build_queue(spec.name, budget=args.budget, force=args.force,
                            make_sheet=not args.no_sheet)
            tot += r.get("n_sel", 0)
        except Exception as e:
            print(f"  失败：{type(e).__name__}: {e}")
    print(f"\n合计 {tot} 帧待标注，队列在 {C.ANNOT_QUEUE_DIR.relative_to(C.ROOT)}/")
    print("下一步： python -m src.annotate --all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
