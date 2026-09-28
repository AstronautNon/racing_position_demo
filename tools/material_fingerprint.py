"""给 drift/ 下的素材留一份「指纹清单」—— 素材不入 git，但指纹要入 git。

为什么需要它
------------
`drift/*.mov` 与 `drift/*.mp4` 被 `.gitignore` 排除（体积 377 MiB，且 video05
越过 GitHub 单文件 100 MiB 硬上限，见 README「素材」一节），所以这些文件
**不在版本库里**，删了就真没了。而一旦素材被重新下载、重新剪辑、换分辨率或
改名，旧标注与旧轨迹会在**看不出来的情况下**变成错的（README 约定 7）。

本工具把「当初是哪一份」写成一个小到可以入库的记录，日后一条命令即可核验。

与 tools/verify_material.py 的分工
---------------------------------
  verify_material.py      —— 逐帧重出裁图比对：**能发现**素材变了，慢，且依赖
                             `outputs/annotations/crops/` 缓存还在。
  material_fingerprint.py —— 只读文件本身算 sha256：快、不依赖任何缓存、
                             **能指出是哪一份**变了、且不要求素材曾被处理过。
互补关系：指纹先筛（秒级），裁图比对再定案（分钟级）。

产物
----
  drift/fingerprints.json   机器可读，记录 bytes / sha256 / 容器元信息
  drift/MANIFEST.md         同一份数据渲染成表，给人看（两者均由本脚本生成）

用法
----
    /opt/anaconda3/bin/python3 tools/material_fingerprint.py            # 试算，只打印
    /opt/anaconda3/bin/python3 tools/material_fingerprint.py --write     # 写入上述两个文件
    /opt/anaconda3/bin/python3 tools/material_fingerprint.py --check     # 核验现有清单

退出码：0 正常 / 核验通过；1 `--check` 发现不一致；2 参数错误或清单缺失。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C          # noqa: E402

MANIFEST_JSON = C.DRIFT_DIR / "fingerprints.json"
MANIFEST_MD = C.DRIFT_DIR / "MANIFEST.md"
EXTS = (".mov", ".mp4")
CHUNK = 1 << 20          # 1 MiB
HASH_IN_MD = 12          # Markdown 表里只印哈希前 N 位，完整值在 JSON

SCHEMA = 1


# ---------------------------------------------------------------------------
# 采集
# ---------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    """分块读，避免把 100 MB 的片子整个塞进内存。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def probe(path: Path) -> dict:
    """只读容器元信息（不解码），失败则返回空字典。"""
    cap = cv2_video_capture(path)
    if cap is None:
        return {}
    try:
        def g(prop) -> float:
            try:
                return float(cap.get(prop) or 0.0)
            except Exception:                       # noqa: BLE001
                return 0.0
        fps = g(cv2_prop("FPS"))
        frames = int(g(cv2_prop("FRAME_COUNT")))
        w = int(g(cv2_prop("FRAME_WIDTH")))
        h = int(g(cv2_prop("FRAME_HEIGHT")))
    finally:
        cap.release()
    out = dict(width=w, height=h, frames=frames,
               fps=round(fps, 3) if fps else 0.0,
               duration_s=round(frames / fps, 2) if fps > 0 else None)
    return out


def cv2_video_capture(path: Path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def cv2_prop(name: str) -> int:
    import cv2
    return getattr(cv2, f"CAP_PROP_{name}")


def material_files() -> list[Path]:
    """drift/ **顶层**的素材文件（与 .gitignore 的作用域一致，不含 results/）。"""
    return sorted(p for p in C.DRIFT_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in EXTS and not p.name.startswith("."))


def preproc_note(name: str) -> dict | None:
    """预处理缓存里的指纹字段（若该素材处理过）。"""
    p = C.OUT_DIR / "preprocess" / f"{name}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return dict(total_frames=d.get("total_frames"), kept_n=len(d.get("kept_frames") or []),
                eff_fps=d.get("eff_fps"), dup_ratio=d.get("dup_ratio"),
                bg_residual_frac=d.get("bg_residual_frac"))


def record(path: Path, do_hash: bool = True) -> dict:
    st = path.stat()
    spec = C.VIDEOS.get(path.stem)
    rec: dict = dict(
        bytes=st.st_size,
        mtime=dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    )
    if do_hash:
        rec["sha256"] = sha256_of(path)
    rec.update(probe(path))
    if spec is not None:
        rec["spec"] = dict(camera=spec.camera, detector=spec.detector, source=spec.source,
                           role=spec.role, realtime=spec.realtime,
                           trim_head=spec.trim_head, trim_tail=spec.trim_tail)
    else:
        rec["spec"] = None
    pre = preproc_note(path.stem)
    if pre:
        rec["preprocess"] = pre
    return rec


def collect(do_hash: bool = True) -> dict:
    files = material_files()
    recs = {p.name: record(p, do_hash) for p in files}
    # 登记的素材：文件可能叫 .mov 也可能叫 .mp4，两种都算在场
    missing = [s.name for s in C.VIDEOS.values()
               if not any((C.DRIFT_DIR / f"{s.name}{e}").exists() for e in EXTS)]
    return dict(
        schema=SCHEMA,
        generated=dt.date.today().isoformat(),
        generator="tools/material_fingerprint.py",
        purpose=("素材文件不入 git（体积 + GitHub 单文件 100 MiB 硬上限，见 README「素材」）。"
                 "本文件是它们的指纹记录，用来核验「现在 drift/ 里的这份，还是当初产出"
                 "标注与轨迹的那一份吗」。记录里的 hash 是文件全字节哈希，"
                 "换分辨率 / 换剪辑点 / 改内容都会变；**改文件名不会变**，"
                 "故本记录按**文件名**索引。"),
        total_bytes=sum(r["bytes"] for r in recs.values()),
        files=recs,
        registered_but_absent=missing,
        unregistered=[n for n in recs if C.VIDEOS.get(Path(n).stem) is None],
    )


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def mib(n: int) -> str:
    return f"{n / 1048576:.1f}"


def render_md(data: dict) -> str:
    lines: list[str] = []
    lines.append("# drift/ 素材指纹清单")
    lines.append("")
    lines.append("> 本文件由 `tools/material_fingerprint.py` 生成，**请勿手改**。"
                 "重新生成：`--write`；核验：`--check`。")
    lines.append("")
    lines.append(data["purpose"])
    lines.append("")
    lines.append(f"- 生成日期：{data['generated']}")
    lines.append(f"- 文件数：{len(data['files'])}，合计 **{mib(data['total_bytes'])} MiB**")
    lines.append("- 哈希算法：SHA-256（完整值见同目录 `fingerprints.json`，下表只印前 "
                 f"{HASH_IN_MD} 位）")
    lines.append("")
    lines.append("核验方式（**换过素材、重下、重剪、改分辨率之后务必跑一次**）：")
    lines.append("")
    lines.append("```bash")
    lines.append("/opt/anaconda3/bin/python3 tools/material_fingerprint.py --check")
    lines.append("# 退出码 0 = 全部还是当初那一份；1 = 有文件变了 / 缺失 / 清单未覆盖")
    lines.append("```")
    lines.append("")
    lines.append("## 清单")
    lines.append("")
    lines.append("| 素材 | 大小 (MiB) | SHA-256 | 分辨率 | 帧率 | 帧数 | 时长 (s) | 登记 | 分析缓存 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name in sorted(data["files"]):
        r = data["files"][name]
        spec = r.get("spec")
        if spec:
            reg = spec["role"] + ("" if spec["realtime"] else " ·非实时")
        else:
            reg = "**未登记**"
        pre = r.get("preprocess")
        cache = (f"{pre['eff_fps']:.1f} fps / {pre['kept_n']} 帧"
                 if pre and pre.get("eff_fps") else "—")
        res = f"{r.get('width', 0)}×{r.get('height', 0)}"
        sha = (r.get("sha256") or "")[:HASH_IN_MD]
        lines.append(f"| `{name}` | {mib(r['bytes'])} | `{sha}` | {res} | "
                     f"{round(r.get('fps', 0), 2):g} | {r.get('frames', 0)} | "
                     f"{r.get('duration_s') if r.get('duration_s') is not None else '—'} | "
                     f"{reg} | {cache} |")
    lines.append("")
    lines.append("列义：「登记」＝ `src/config.py` 里的 `role`（非实时＝原片做过加速，"
                 "见 README 约定 12）；「分析缓存」＝ `outputs/preprocess/<名>.json` 的"
                 "有效帧数与有效帧率，「—」表示该素材还没跑过预处理。")
    lines.append("")

    if data["registered_but_absent"]:
        lines.append("## 登记了但 drift/ 里没有文件")
        lines.append("")
        for n in data["registered_but_absent"]:
            lines.append(f"- `{n}` —— 文件缺失。旧标注与轨迹仍在，但跑不了；"
                         "找回前先跑 `tools/verify_material.py` 确认是不是同一份。")
        lines.append("")
    if data["unregistered"]:
        lines.append("## drift/ 里有但 config.py 未登记")
        lines.append("")
        for n in data["unregistered"]:
            lines.append(f"- `{n}` —— 未登记，流水线不会处理它。")
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 核验
# ---------------------------------------------------------------------------
def check(manifest: Path = MANIFEST_JSON) -> int:
    if not manifest.exists():
        print(f"清单不存在：{manifest}\n先跑 --write 生成。")
        return 2
    old = json.loads(manifest.read_text(encoding="utf-8")).get("files", {})
    print(f"核验 drift/ 素材指纹（对照 {manifest.name}，{len(old)} 条记录）")
    print("=" * 78)

    changed = missing = 0
    seen: set[str] = set()
    for name in sorted(old):
        p = C.DRIFT_DIR / name
        seen.add(name)
        if not p.exists():
            print(f"  [缺失] {name:<16} 清单里有，drift/ 里找不到")
            missing += 1
            continue
        now = record(p)
        o = old[name]
        why: list[str] = []
        if now["bytes"] != o.get("bytes"):
            why.append(f"大小 {o.get('bytes')} → {now['bytes']}")
        if now.get("sha256") != o.get("sha256"):
            why.append("内容哈希不符")
        if why:
            print(f"  [已变] {name:<16} {'；'.join(why)}  ← 旧标注/轨迹按约定 7 需作废重标")
            changed += 1
        else:
            print(f"  [一致] {name:<16} {now['bytes']:>12,} B  sha {now['sha256'][:HASH_IN_MD]}…")

    fresh = [p.name for p in material_files() if p.name not in seen]
    for n in fresh:
        print(f"  [新增] {n:<16} 清单里没有它 —— 清单已过期，跑 --write 补上")
    print("=" * 78)
    if changed or missing or fresh:
        print(f"结论：{changed} 个已变、{missing} 个缺失、{len(fresh)} 个未覆盖 → 清单需要处理")
        return 1
    print("结论：全部一致 —— drift/ 里就是当初产出标注与轨迹的那一份。")
    return 0


# ---------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    if "--check" in argv:
        return check()
    if not any(a in argv for a in ("--write", "--dry-run", "--print")):
        print(__doc__)
        return 2

    do_hash = "--no-hash" not in argv
    data = collect(do_hash=do_hash)
    if "--write" in argv:
        MANIFEST_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        MANIFEST_MD.write_text(render_md(data), encoding="utf-8")
        print(f"已写入 {MANIFEST_JSON.relative_to(ROOT)} 与 {MANIFEST_MD.relative_to(ROOT)}")
    print(f"{len(data['files'])} 个文件，合计 {mib(data['total_bytes'])} MiB")
    for name in sorted(data["files"]):
        r = data["files"][name]
        sha = (r.get("sha256") or "—")[:HASH_IN_MD]
        print(f"  {name:<16} {mib(r['bytes']):>8} MiB  {r.get('width', 0)}×{r.get('height', 0)}"
              f"  {r.get('frames', 0):>5} 帧  sha {sha}")
    if data["registered_but_absent"]:
        print("登记了但文件缺失：" + ", ".join(data["registered_but_absent"]))
    if data["unregistered"]:
        print("drift/ 里有但未登记：" + ", ".join(data["unregistered"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
