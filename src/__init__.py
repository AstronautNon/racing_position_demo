"""赛车漂移姿态识别 demo —— 源码包。

模块划分：
    config       素材登记表与全局约定
    preprocess   黑边裁剪、重复帧剔除、时间轴重建
    detect       双分支车辆检测（静止机位背景建模 / 相机运动走 geo-trax）
    kinematics   轨迹平滑、速度与航向 psi_vel、滑移角 beta
    pipeline     命令行入口，把上面几层串起来
"""

__version__ = "0.1.0"
