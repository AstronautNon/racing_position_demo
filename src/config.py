"""项目级配置：路径、角度约定、素材登记表。

素材登记表（VIDEOS）把《项目规划.md》三轮素材体检的结论固化下来，
后续所有模块都读它，避免把"哪段素材能用、哪段相机在动"散落在各处代码里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = ROOT / "drift"
GEOTRAX_DIR = DRIFT_DIR / "results"          # geo-trax 历史输出
OUT_DIR = ROOT / "outputs"
PREPROC_DIR = OUT_DIR / "preprocess"
TRACK_DIR = OUT_DIR / "tracks"
PLOT_DIR = OUT_DIR / "plots"
REPORT_DIR = OUT_DIR / "reports"
ANNOT_DIR = OUT_DIR / "annotations"          # 人工标注（车身轴）落在这里
ANNOT_QUEUE_DIR = ANNOT_DIR / "queue"        # 待标注帧队列（选帧结果）
ANNOT_CROP_DIR = ANNOT_DIR / "crops"         # 裁好的待标图（标注台按需生成并缓存）
ANNOT_WEB_DIR = ANNOT_DIR / "web"            # 离线版标注台导出目录
CACHE_DIR = OUT_DIR / "cache"

for _d in (PREPROC_DIR, TRACK_DIR, PLOT_DIR, REPORT_DIR, ANNOT_DIR,
           ANNOT_QUEUE_DIR, ANNOT_CROP_DIR, ANNOT_WEB_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 全局约定
# ---------------------------------------------------------------------------
# 角度：图像 x 轴向右为 0°，顺时针为正，范围 [0, 360)。
# 由于图像 y 轴向下，atan2(dy, dx) 天然就是"顺时针为正"，无需翻转符号。
# 滑移角 beta = psi_body - psi_vel，顺时针为正（与 demo01 保持一致）。
ANGLE_MOD = 360.0

# 车身轴是无向的（mod 180°）。漂移时恒有 |beta| < 90°，
# 故取"与运动方向夹角 <= 90°"的那一端即为车头 —— 180° 歧义可自动消解。
BODY_AXIS_MOD = 180.0

# 预处理：重复帧判据（详见《项目规划.md》§13.1）
# 相邻帧缩到 DEDUP_SIZE 灰度后，|diff| > DEDUP_PIXEL_THR 的像素占比
# 小于 DEDUP_FRAC 即判为重复帧（同一原始画面被重复写入）。
DEDUP_SIZE = (480, 270)
DEDUP_PIXEL_THR = 20
DEDUP_FRAC = 5e-4

# 检测：静止机位背景建模参数
# 阈值经参数扫描确定：45 在 video02 上使与 geo-trax 的 IoU 达 0.933，
# 同时把 video12 的轨迹跳点比（逐帧位移 p95/p50）从 10.3 压到 5.1。
# 更低（32）对 video02 略好但 video12 明显变差；更高（65）会丢失车身弱对比部分。
BG_THRESHOLD = 45          # 与背景的最大通道差阈值
BG_CLOSE_K = 17            # 闭运算核，用来把车顶/车窗造成的掩膜碎片桥接起来
BG_OPEN_K = 5              # 开运算核，用来去掉字幕笔画等细结构
BG_BLOCK = 40              # 每处理多少帧重建一次背景
BG_WINDOW = 60             # 重建时向前后各取多少帧（按保留帧索引计）
BG_SAMPLE_STEP = 2         # 重建时的抽样步长
MIN_AREA_FRAC = 0.0012     # 最小连通域面积占工作图比例
MAX_AREA_FRAC = 0.50       # 最大连通域面积占工作图比例
MIN_AREA_PX = 120          # 绝对面积下限
# 长宽比上限：俯视/斜视下的车辆长宽比一般在 1:1 ~ 4:1。
# 实测视频里的字幕带长宽比可达 13:1（如 video02 的 309x23），必须排除。
MAX_ASPECT = 6.0
# 背景模型可用性阈值：中值背景扣掉车辆后仍残留的差异像素占比
# 实测 01/02/03/07/08/09/13/14/15 为 0.02%~1.1%，04/05/06 为 10%~21%，
# 10/11 约 5.3%，video12 约 3.0%（缓慢漂移，需靠滚动背景兜住）。
BG_RESIDUAL_MAX = 0.03

# 运动学
SMOOTH_WINDOW = 7          # Savitzky-Golay 窗口（奇数）
SPEED_MIN_PXS = 3.0        # 速度低于此值(px/s)时航向视为无定义

# ---------------------------------------------------------------------------
# 人工标注（车身轴）
# ---------------------------------------------------------------------------
# 标注预算：按素材的样本量与"朝向精度潜力"分配。
# video03 车最大（占宽 36.6%）→ 朝向精度潜力最高，值得多标；
# video12 样本最多（265 帧可用）；video15 有 12% 帧被判位置离群，故少一些。
ANNOT_QUOTA = {
    "video12": 55,
    "video02": 50,
    "video03": 45,
    "video15": 35,
    "video01": 25,
    "video09": 20,
}
ANNOT_BUDGET_DEFAULT = 40          # 未在 ANNOT_QUOTA 里的素材用这个值

# 选帧：贪心最远点采样，特征 = [朝向代理(倍角), 画面位置, 时间, 外观]
# 权重体现"什么对标注最有价值"：朝向覆盖第一，其次是别把帧都堆在轨迹同一处。
ANNOT_W_AXIS = 1.00
ANNOT_W_POS = 0.70
ANNOT_W_TIME = 0.80
ANNOT_W_APPEAR = 0.60
ANNOT_APPEAR_SIZE = 48             # 外观特征缩略图边长

# 选帧质量门槛
ANNOT_MIN_CONF_FRAC = 0.25         # 置信度须高于"该素材置信度分布的 25 分位"
ANNOT_MIN_BOX_FRAC = 0.03          # 车框长边须不小于工作图宽度的 3%

# 车框宽高比上限。**这是"掩膜到底是不是车"的判据，不只是形状偏好。**
# 轴对齐外框的宽高比有几何上界：一辆长宽比 r 的车无论转到什么角度，
# 其外框宽高比都不超过 r（θ=0°/90° 取到上界，45° 时降到约 √2·r/(r+1) 的最小值）。
# 市售车 r ≈ 2~2.5，所以正常检出的宽高比应当 ≤ 2.5。
# 而"车辆快速横移时掩膜退化成沿运动方向拉长的月牙带"能给出 4.7~5.8：
# 实测 video12 有 6 帧框宽 671~731 px（占 960 px 工作图的 70~76%），
# 那根本不是车 —— 拿它当标注目标，等于让用户对着一条带子点车身。
# 取 3.0 留出余量（真实车 ≤2.5），同时把月牙带挡在外面。
ANNOT_MAX_BOX_ASPECT = 3.0
# 车框允许超出画面的幅度（占车框长边的比例）。
# 不用"距画面边缘留白"，因为检测框会被车身阴影撑大：实测 video15 有 50/62 帧
# 框越界，但越界幅度中位只有 7.6%、p90 13.1%、max 15.2%，车本身其实在画面内。
# 按"距边 1%"筛会误杀 43 帧；按"越界幅度 ≤ 10%"筛既保住这些帧，
# 又仍能剔除车身真被截去大半的情况（剩下 10% 的可见部分仍足以定出车身轴）。
ANNOT_MAX_OVERHANG = 0.10

# 标注台渲染
ANNOT_PAD = 1.90                   # 裁图半边长 = PAD/2 × 车框长边（留出上下文）
ANNOT_DISP_MAX = 1000              # 显示图长边上限（超出则缩小）
ANNOT_DISP_MIN = 560               # 显示图长边下限（不足则放大，方便点选）
ANNOT_JPEG_Q = 88

# 人工标注的稀疏插值上限：相邻两个已标帧间隔超过这么多帧就不插值。
# 车身轴的角速度在稳态漂移下近似恒定，短缺口插值可靠；
# 缺口太长则"线性转动"这个假设站不住，宁可留 NaN。
ANNOT_MAX_GAP = 24

# 选帧时的最大间隔（帧）。**这是与 ANNOT_MAX_GAP 不同的约束，别混**：
#   ANNOT_MAX_GAP    是"插值器最多敢跨多长缺口"，作用在插值阶段；
#   ANNOT_SELECT_MAX_GAP 是"选帧结果本身不许留下多长的空洞"，作用在采样阶段。
# 早先只有前者，于是最远点采样可以为了朝向多样性跳掉一整段 ——
# 实测 video12 队列在 k 9→65 与 114→171 处各断了 56/57 帧（区间检出率 100%、
# 帧间差异还比别处更大，跳得毫无依据），而插值跨不过 24 帧，
# 结果 β 在这 111 帧（占全片 39%）整段为空，标注预算白花。
# 取 0.75×ANNOT_MAX_GAP 留出余量：正好卡在上限上的缺口没有容错空间。
ANNOT_SELECT_MAX_GAP = 18

# 工作图宽度：统一按"原图宽度一半"缩放，兼顾精度与速度
WORK_WIDTH_MIN = 640
WORK_WIDTH_MAX = 1352


def work_width_for(width: int) -> int:
    """由原始画面宽度决定工作图宽度。"""
    return min(WORK_WIDTH_MAX, max(WORK_WIDTH_MIN, width // 2))


# geo-trax 在这些素材上的输出已知不可用（原因见《项目规划.md》§12.2：
# 目标相对尺度过大 + 地面重复砖纹导致稳定化退化），
# 因此它们的交叉校验数值没有参考价值，报告里要显式说明。
GEOTRAX_BROKEN = {"video03"}


# ---------------------------------------------------------------------------
# 素材登记表
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VideoSpec:
    """单段素材的登记信息。"""

    name: str
    camera: str            # "static" | "moving"
    detector: str          # "bgsub" | "geotrax"
    source: str            # 来源分组标识，用于"按来源划分数据集"
    role: str              # main / aux / special / sample / hard / shelved / low
    trim_head: int = 0     # 丢弃开头 N 帧
    trim_tail: int = 0     # 丢弃末尾 N 帧
    mask_bands: tuple = ()  # 检测时排除的横带 ((y0比例, y1比例), ...)
    notes: str = ""

    @property
    def path(self) -> Path:
        """素材文件路径，自动匹配 .mov / .mp4。"""
        for ext in (".mov", ".mp4"):
            p = DRIFT_DIR / f"{self.name}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"找不到素材文件：{self.name}.mov / .mp4")

    @property
    def geotrax_track(self) -> Path:
        return GEOTRAX_DIR / f"{self.name}.txt"

    @property
    def has_geotrax(self) -> bool:
        p = self.geotrax_track
        return p.exists() and p.stat().st_size > 0


VIDEOS: dict[str, VideoSpec] = {
    v.name: v
    for v in [
        # --- 旧素材（第一批，全部是屏幕录制，重复帧 50%~72%） ---
        VideoSpec("video01", "static", "bgsub", "S6", "aux",
                  notes="宝马广告横滑，竖屏内容嵌横屏，左右各约 879/68 px 黑边"),
        VideoSpec("video02", "static", "bgsub", "S7", "main",
                  notes="近正射略倾，甩尾。geo-trax 中真车是 ID 31，ID 1 是路边停的车"),
        VideoSpec("video03", "static", "bgsub", "S8", "special",
                  notes="近正射，绕蓝色锥桶画圆，地面有靶心可标定。车占画面宽 37.8%，"
                        "geo-trax 完全失效，必须走背景建模"),
        VideoSpec("video04", "moving", "geotrax", "S9", "shelved",
                  notes="倾斜俯视，林间弯道，未漂移；相机位移 18% 画面宽"),
        VideoSpec("video05", "moving", "geotrax", "S10", "shelved",
                  notes="明显斜视，车小烟浓，低空人群；相机位移 22%"),
        VideoSpec("video06", "moving", "geotrax", "S11", "shelved",
                  notes="明显斜视；相机位移 27%"),
        # --- 新素材（第二批，直接下原文件，重复帧 3.5%~31%） ---
        VideoSpec("video07", "static", "bgsub", "S1", "low",
                  notes="英文 skid pad 教学全景点机位。车身仅占画面宽 3.0%，"
                        "朝向标注误差会被放大，训练价值低"),
        VideoSpec("video08", "static", "bgsub", "S1", "low",
                  notes="与 video07 同一拍摄；车身占宽 5.6%"),
        VideoSpec("video09", "static", "bgsub", "S1", "sample",
                  notes="同场地近景正俯视，车身占宽 20.6%，但有效仅 1.2 s，适合当演示片段"),
        VideoSpec("video10", "moving", "geotrax", "S2", "hard",
                  notes="中文教学，多锥桶 skid pad，A.R.T 水印；相机在绕车运动（位移 14.6%），"
                        "用于检验相机运动下的鲁棒性"),
        VideoSpec("video11", "moving", "geotrax", "S2", "hard",
                  notes="与 video10 同源；相机位移 12.8%"),
        VideoSpec("video12", "static", "bgsub", "S3", "main",
                  trim_tail=6,
                  notes="正俯视无人机，碎石地定圆，有效 11.6 s（最长）。"
                        "相机缓慢漂移约 23 px，末尾 6 帧发生镜头切换（已裁）"),
        VideoSpec("video13", "static", "bgsub", "S4", "shelved",
                  notes="正俯视八字（动作价值最高），但仅 640x360，车身约 50 px，"
                        "标注精度受限，待回源找 1080p"),
        VideoSpec("video14", "static", "bgsub", "S4", "shelved",
                  notes="与 video13 同源，八字；仅 640x360"),
        VideoSpec("video15", "static", "bgsub", "S5", "main",
                  trim_tail=14,
                  notes="正俯视定圆，地面有轮胎圆环痕，车身占宽 25.6%。"
                        "上下黑边（上 31 / 下 52 px）。末尾 14 帧是片尾数字倒计时大字幕（已裁）"),
    ]
}


def get(name: str) -> VideoSpec:
    if name not in VIDEOS:
        raise KeyError(f"未登记的素材：{name}；可选 {sorted(VIDEOS)}")
    return VIDEOS[name]


def static_videos() -> list[VideoSpec]:
    return [v for v in VIDEOS.values() if v.camera == "static"]


def active_static_videos() -> list[VideoSpec]:
    """静止机位中非搁置的素材。"""
    return [v for v in static_videos() if v.role != "shelved"]


def work_videos() -> list[VideoSpec]:
    """M1 主线处理的素材：静止机位、非搁置。"""
    return active_static_videos()


# 默认进入标注队列的素材：静止机位里 role 为 main / special / aux / sample 的。
# 不含 low（车身仅占宽 3%~5%，朝向标注误差会被放大）与 shelved（待回源 1080p）。
ANNOT_ROLES = ("main", "special", "aux", "sample")


def annotation_videos() -> list[VideoSpec]:
    return [v for v in static_videos() if v.role in ANNOT_ROLES]


def annot_quota(name: str) -> int:
    return ANNOT_QUOTA.get(name, ANNOT_BUDGET_DEFAULT)
