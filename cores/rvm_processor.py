"""
RVM (Robust Video Matting) 背景移除处理器
基于 PyTorch 的轻量级背景分割模型
"""
import torch
import cv2
import numpy as np
from typing import Tuple
from loguru import logger

from cores.rvm.model import MattingNetwork


class RVMProcessor:
    """RVM 背景移除处理器（单帧）"""

    def __init__(self, checkpoint_path: str, variant: str = 'resnet50', device: str = 'cuda'):
        """
        初始化 RVM 处理器

        Args:
            checkpoint_path: 模型权重路径
            variant: 模型变体 ('resnet50' 或 'mobilenetv3')
            device: 设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"RVM 处理器使用设备: {self.device}")

        # 加载模型
        self.model = MattingNetwork(variant=variant).eval().to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        logger.info(f"RVM 模型加载成功: {variant} - {checkpoint_path}")

        # 循环状态（用于视频序列）
        self.rec = [None] * 4

    @torch.no_grad()
    def process_single_frame(self,
                            frame: np.ndarray,
                            bg_color: Tuple[int, int, int] = (0, 255, 0),
                            downsample_ratio: float = 0.5) -> np.ndarray:
        """
        处理单帧图像，替换背景为纯色

        Args:
            frame: BGR 图像 (H, W, 3)
            bg_color: 背景颜色 (B, G, R)
            downsample_ratio: 推理缩放比例（0.5 = 2倍下采样）

        Returns:
            BGR 图像，背景已替换为 bg_color
        """
        # 1. BGR -> RGB -> Tensor
        src = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        src = torch.from_numpy(src).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 2. 推理
        fgr, pha, *self.rec = self.model(src, *self.rec, downsample_ratio=downsample_ratio)

        # 3. 合成：fgr * pha + bg * (1 - pha)
        fgr = fgr.squeeze(0).permute(1, 2, 0).cpu().numpy()
        pha = pha.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # 背景颜色归一化（RGB）
        bg = np.array(bg_color[::-1], dtype=np.float32) / 255.0  # BGR -> RGB

        # Alpha 合成
        com = fgr * pha + bg * (1 - pha)
        com = (com * 255).clip(0, 255).astype(np.uint8)

        # 4. RGB -> BGR
        com = cv2.cvtColor(com, cv2.COLOR_RGB2BGR)

        return com

    def reset_state(self):
        """重置循环状态（处理新视频时调用）"""
        self.rec = [None] * 4
