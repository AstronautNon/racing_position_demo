"""一次性的坐标归一：把历史标注里的点选坐标还原成**真正的工作图坐标**。

背景（详见 项目规划 §14.16）：`crops.render()` 曾把裁图原点算成 `fx0/sxf`，
漏减了黑边裁移 `l`（应为 `(fx0-l)/sxf`）。后果：

  · 存进标注 CSV 的 `x1..y2` 整体多算了 `(l/sxf, t/syf)`；
  · **`axis_deg` 完全不受影响** —— 两点同向平移，差向量不变，所以角度、
    插值、β、所有报告数字都是对的（这也正是它藏了这么久的原因）。

本工具把那些坐标减回正确值，让 CSV 的坐标列与代码修好之后的约定一致。

**为什么不担心"改坏标注劳动成果"**：迁移只做平移 `Δ = (-l/sxf, -t/syf)`，
因此 `axis_deg` 必须逐帧**严格不变**。工具对每一行都重算一遍角度并比对，
任何一行对不上就整体拒绝写入 —— 这条校验是硬门槛，不是提示。

用法：
    /opt/anaconda3/bin/python3 tools/migrate_annot_coords.py            # 试算（默认，不写文件）
    /opt/anaconda3/bin/python3 tools/migrate_annot_coords.py --apply    # 真正写入（先备份 .bak）
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C        # noqa: E402
from src import crops as CR        # noqa: E402
from src import preprocess as P    # noqa: E402

MARK = "coord_fix"      # 已迁移的标记，避免二次平移
# 角度不变性校验的容差。**不能取 1e-6**：CSV 的坐标只存 2 位小数，平移后重新
# 四舍五入会引入 ~0.01 px 的位置量化，对一条 ~65 px 长的轴折算约 0.01/65 ≈ 0.009°。
# 而真正"改坏了"的量级是**度**，所以 0.05° 这个门槛既容得下量化噪声，
# 又比要抓的错误小两个数量级。
TOL_DEG = 0.05


def targets() -> list[Path]:
    out = sorted(C.ANNOT_DIR.glob("video*.csv"))
    blind = C.ANNOT_BLIND_DIR
    if blind.exists():
        out += sorted(blind.glob("labels_video*.csv"))
    return out


def delta_for(name: str) -> tuple[float, float]:
    """该素材历史上多算的坐标偏移 `(l/sxf, t/syf)`。"""
    res = P.preprocess(C.get(name))
    l, t, _, _ = res.crop
    sxf, syf = P.work_scale(res)
    return l / sxf, t / syf


def process(p: Path, apply: bool) -> tuple[int, float, float, str]:
    """返回 (迁移行数, dx, dy, 说明)。"""
    name = p.name.split("_")[-1].removesuffix(".csv")
    if not name.startswith("video"):
        return 0, 0.0, 0.0, "文件名不是素材名，跳过"

    lines = p.read_text(encoding="utf-8").splitlines()
    if any(MARK in ln and ln.lstrip().startswith("#") for ln in lines):
        return 0, 0.0, 0.0, "已迁移过，跳过"

    dx, dy = delta_for(name)
    if abs(dx) < 0.05 and abs(dy) < 0.05:
        return 0, dx, dy, "该素材没有黑边裁移，无需迁移"

    out_lines: list[str] = []
    n = 0
    worst = 0.0
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            out_lines.append(ln)
            continue
        row = next(csv.reader([ln]))
        if row[0].strip() == "k":              # 表头
            out_lines.append(ln)
            continue
        try:
            k = int(float(row[0]))
            axis = float(row[1])
            x1, y1, x2, y2 = (float(v) for v in row[2:6])
        except (ValueError, IndexError):
            out_lines.append(ln)
            continue

        nx1, ny1, nx2, ny2 = x1 - dx, y1 - dy, x2 - dx, y2 - dy
        new_axis = CR.points_to_axis_deg((nx1, ny1), (nx2, ny2))
        # 硬校验：平移不改变方向，角度必须逐帧不变
        d = abs((new_axis - axis + 90.0) % 180.0 - 90.0)
        worst = max(worst, d)
        if d > TOL_DEG:
            raise AssertionError(
                f"{p.name} k={k}: 迁移后角度变了 {d:.6f}°（{axis:.6f} → {new_axis:.6f}）")

        row[2], row[3], row[4], row[5] = (f"{nx1:.2f}", f"{ny1:.2f}",
                                          f"{nx2:.2f}", f"{ny2:.2f}")
        out_lines.append(",".join(row))
        n += 1

    if n and apply:
        shutil.copy2(p, p.with_suffix(p.suffix + ".pre_coordfix.bak"))
        marker = (f"# {MARK}: 2026-09-28 已减去黑边裁移 (l/sxf, t/syf)=({dx:.2f},{dy:.2f})；"
                  f"坐标现在与工作图一致，axis_deg 未变（迁移时逐帧校验 ≤ {worst:.1e}）")
        p.write_text("\n".join([out_lines[0], marker] + out_lines[1:]) + "\n",
                     encoding="utf-8")

    return n, dx, dy, f"角度不变性校验最大偏差 {worst:.2e}°"


def main() -> None:
    apply = "--apply" in sys.argv
    print("=== 标注点选坐标归一（" + ("写入" if apply else "试算，不写文件") + "）===")
    total = failed = 0
    for p in targets():
        try:
            n, dx, dy, note = process(p, apply)
        except AssertionError as e:
            print(f"  ✗ {p.name}: {e}")
            failed += 1
            continue
        flag = "→" if (n and apply) else ("·" if n else " ")
        print(f"  {flag} {p.name:<24} {n:>3} 行   Δ=({dx:6.2f},{dy:6.2f})   {note}")
        total += n
    print(f"\n合计 {total} 行" + ("已迁移。" if apply else "待迁移（加 --apply 执行）。")
          + (f"  **{failed} 个文件校验未过，未处理**" if failed else ""))
    if not apply and total:
        print("提示：先确认标注台已关闭再 --apply，避免它的内存状态覆盖磁盘。")


if __name__ == "__main__":
    main()
