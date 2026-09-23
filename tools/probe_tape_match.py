"""素材溯源：找出某段素材是从哪个母带剪出来的。

为什么需要它：

1. **素材随时可能重下/重剪。** 知道"素材 ← 母带"这条链，才能在需要时
   （换清晰度、加长片段、去字幕重剪）回到母带去重新导出，而不是到处找文件。
2. **避免重复劳动。** 新下的素材如果已经剪过，溯源能立刻发现。
3. **看清母带还剩多少没用。** 母带往往比剪出来的片段长得多（例：103 s 的母带只用了 27 s），
   那些没用到的部分就是现成的候选素材池，不必再去网上找。
4. 与 `verify_material.py` 互补：那个管"素材有没有被换过"，这个管"素材从哪来"。

原理：素材是母带的子区间，内容逐帧对应。把素材的中间帧缩成 160×90 灰度，
在母带里逐帧搜最小 MSE 即可定位。同源时 MSE 落在 1 量级（只剩编码噪声），
不同源时都在 1000 量级，区分度极干净。

用法：
    # 溯源单段素材（自动在常见下载目录里找候选母带）
    /opt/anaconda3/bin/python3 tools/probe_tape_match.py video10

    # 溯源多段
    /opt/anaconda3/bin/python3 tools/probe_tape_match.py video10 video11 video13

    # 全部已登记素材 × 指定母带
    /opt/anaconda3/bin/python3 tools/probe_tape_match.py --all --tape ~/Desktop/26570394849-1-100113.mp4

    # 不用素材名，直接把某个文件当查询
    /opt/anaconda3/bin/python3 tools/probe_tape_match.py --query some_clip.mp4

结果里的「时间」是母带内的位置，「折算时间」按母带 fps 换算，可直接拿去重剪。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402

W, H = 160, 90
STEP = 2                       # 母带扫描步长（帧）
MATCH_MSE = 100.0              # 同源判定
NEAR_MSE = 600.0               # 疑似

# 候选母带自动发现的目录（不放 drift/，那里是素材本身）
SCAN_DIRS = [Path.home() / "Desktop", Path.home() / "Movies", Path.home() / "Downloads"]
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def gray_frame(path: Path, idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(cv2.resize(fr, (W, H)), cv2.COLOR_BGR2GRAY).astype(np.float32)


def probe(path: Path) -> tuple[int, float]:
    """返回 (帧数, fps)。"""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    cap.release()
    return n, fps


def scan_tape(tape: Path, queries: dict[str, np.ndarray],
              step: int = STEP) -> tuple[int, float, dict[str, tuple[float, int]]]:
    """扫一遍母带，同时为所有查询取最小 MSE（避免每段素材重扫）。"""
    n, fps = probe(tape)
    best = {k: (1e18, -1) for k in queries}
    cap = cv2.VideoCapture(str(tape))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv2.cvtColor(cv2.resize(fr, (W, H)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            for k, q in queries.items():
                d = float(np.mean((g - q) ** 2))
                if d < best[k][0]:
                    best[k] = (d, i)
        i += 1
    cap.release()
    return n, fps, best


def find_tapes() -> list[Path]:
    out = []
    for d in SCAN_DIRS:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in VIDEO_EXT and p.is_file() and p.stat().st_size > 2 << 20:
                out.append(p)
    return out


def build_queries(names: list[str], explicit_query: Path | None) -> dict[str, tuple[Path, np.ndarray]]:
    qs: dict[str, tuple[Path, np.ndarray]] = {}
    if explicit_query:
        n, _ = probe(explicit_query)
        g = gray_frame(explicit_query, int(n * 0.5))
        if g is not None:
            qs[explicit_query.name] = (explicit_query, g)
        return qs
    for name in names:
        spec = C.get(name)
        if spec is None or not spec.path.exists():
            print(f"  跳过 {name}：未登记或文件不存在")
            continue
        n, _ = probe(spec.path)
        g = gray_frame(spec.path, int(n * 0.5))
        if g is not None:
            qs[name] = (spec.path, g)
    return qs


def main(argv: list[str]) -> int:
    tapes: list[Path] = []
    # --tape 只吃紧跟其后的一个路径；要指定多个就重复写 --tape（或逗号分隔）
    while "--tape" in argv:
        i = argv.index("--tape")
        if i + 1 >= len(argv):
            print("--tape 后面缺少路径")
            return 2
        for seg in argv[i + 1].split(","):
            if seg.strip():
                tapes.append(Path(seg.strip()).expanduser())
        argv = argv[:i] + argv[i + 2:]

    explicit = None
    if "--query" in argv:
        i = argv.index("--query")
        explicit = Path(argv[i + 1]).expanduser()
        argv = argv[:i] + argv[i + 2:]

    args = [a for a in argv[1:] if not a.startswith("-")]
    if not explicit:
        if "--all" in argv:
            names = [s.name for s in C.active_static_videos()]
        elif args:
            names = args
        else:
            print(__doc__)
            return 2
    else:
        names = []

    if not tapes:
        tapes = find_tapes()

    qs = build_queries(names, explicit)
    if not qs:
        print("没有可用的查询素材")
        return 2
    queries = {k: v[1] for k, v in qs.items()}

    print(f"查询 {len(queries)} 段素材，候选母带 {len(tapes)} 个")
    print("（MSE<100 = 同源；<600 = 疑似；否则不同源）")
    print("=" * 86)
    print(f"{'素材':<10}{'母带':<34}{'素材帧号':>9}{'时间':>9}{'MSE':>10}  判定")
    print("-" * 86)
    for t in tapes:
        try:
            n, fps, best = scan_tape(t, queries)
        except Exception as e:                                   # noqa: BLE001
            print(f"{'':<10}{t.name[:33]:<34}  读取失败：{type(e).__name__}: {e}")
            continue
        for k in queries:
            d, i = best[k]
            v = "**同源**" if d < MATCH_MSE else ("疑似" if d < NEAR_MSE else "不同源")
            tname = t.name if len(t.name) <= 33 else t.name[:30] + "..."
            print(f"{k:<10}{tname:<34}{i:>9}{i / fps:>8.2f}s{d:>10.1f}  {v}")
        print(f"{'':<10}（母带共 {n} 帧 / {n / fps:.1f}s @{fps:.2f}fps）")
        print("-" * 86)
    print("提示：同一母带上若多段素材都命中，注意它们的区间是否重叠（重叠=重复剪同一段）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
