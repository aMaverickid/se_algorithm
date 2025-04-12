"""
深度估计模块，用于生成深度图
"""
import logging
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline, DPTForDepthEstimation, DPTImageProcessor
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

class DepthEstimator:
    """深度估计器，用于生成深度图"""
    
    def __init__(self, model_name="Intel/dpt-large", device=config.DEVICE):
        """
        初始化深度估计器
        
        Args:
            model_name (str): 深度估计模型名称
            device (str): 设备类型，'cuda'或'cpu'
        """
        self.device = device
        self.model_name = model_name
        
        logger.info(f"加载深度估计模型: {model_name}")
        self.depth_estimator = pipeline(
            "depth-estimation", 
            model=model_name,
            device=0 if device == "cuda" else -1,
            model_kwargs={"cache_dir": config.MODEL_CACHE_DIR}
        )
    
    def estimate_depth(self, image):
        """
        估计图像的深度图
        
        Args:
            image: PIL图像或图像路径
            
        Returns:
            深度图（PIL图像）
        """
        # 确保图像为PIL格式
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        if not isinstance(image, Image.Image):
            raise ValueError("image必须是PIL.Image.Image类型或有效的图像路径")
        
        # 执行深度估计
        depth_image = self.depth_estimator(image)["depth"]
        
        # 标准化深度图以用于ControlNet
        depth_image = self._normalize_depth(depth_image)
        
        return depth_image
    
    def _normalize_depth(self, depth_image):
        """
        标准化深度图（将深度值标准化到0-255范围）
        
        Args:
            depth_image: 深度图PIL图像
            
        Returns:
            标准化后的深度图
        """
        # 转换为numpy数组
        depth_np = np.array(depth_image)
        
        # 标准化到0-255范围
        depth_min = depth_np.min()
        depth_max = depth_np.max()
        
        if depth_max > depth_min:
            depth_np = ((depth_np - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_np = np.zeros_like(depth_np, dtype=np.uint8)
        
        # 应用颜色映射以更好地可视化
        depth_colored = cv2.applyColorMap(depth_np, cv2.COLORMAP_TURBO)
        depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        
        # 转回PIL格式
        return Image.fromarray(depth_colored_rgb)
    
    def enhance_depth(self, depth_image, contrast=1.5, brightness=0.0):
        """
        增强深度图的对比度和亮度
        
        Args:
            depth_image: 深度图PIL图像
            contrast (float): 对比度增强因子
            brightness (float): 亮度调整值
            
        Returns:
            增强后的深度图
        """
        # 转换为numpy数组
        depth_np = np.array(depth_image)
        
        # 增强对比度和亮度
        depth_np = cv2.convertScaleAbs(depth_np, alpha=contrast, beta=brightness)
        
        # 转回PIL格式
        return Image.fromarray(depth_np)
    
    def blend_depth_with_original(self, original_image, depth_image, alpha=0.7):
        """
        将深度图与原始图像混合
        
        Args:
            original_image: 原始PIL图像
            depth_image: 深度图PIL图像
            alpha (float): 混合因子，0.0表示完全使用原始图像，1.0表示完全使用深度图
            
        Returns:
            混合后的图像
        """
        # 确保图像尺寸一致
        if original_image.size != depth_image.size:
            depth_image = depth_image.resize(original_image.size, Image.LANCZOS)
        
        # 转换为numpy数组
        original_np = np.array(original_image)
        depth_np = np.array(depth_image)
        
        # 执行混合
        blended = cv2.addWeighted(original_np, 1 - alpha, depth_np, alpha, 0)
        
        # 转回PIL格式
        return Image.fromarray(blended)
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'depth_estimator'):
            del self.depth_estimator
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 