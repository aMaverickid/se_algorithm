"""
IP-Adapter模型的基础类
"""
import os
import logging
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
import numpy as np
from diffusers import DiffusionPipeline
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

class BaseIPAdapter(ABC):
    """IP-Adapter模型的抽象基类"""
    
    def __init__(self, device=config.DEVICE):
        """
        初始化IP-Adapter基础模型
        
        Args:
            device (str): 运行模型的设备，'cuda'或'cpu'
        """
        self.device = device
        self.torch_dtype = torch.float16 if device == "cuda" else torch.float32
        self.image_encoder = None # 图像编码器 用来提取图像特征
        self.image_processor = None # 图像处理器 用来处理图像
        self.pipeline = None # 扩散模型
        self._load_image_encoder()
    
    def _load_image_encoder(self):
        """加载IP-Adapter使用的图像编码器"""
        logger.info("加载IP-Adapter图像编码器")
        
        try:
            self.image_processor = CLIPImageProcessor.from_pretrained(
                config.CLIP_MODEL_PATH,
                torch_dtype=self.torch_dtype
            )
            logger.info("成功从本地加载CLIP图像处理器")
            
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                config.CLIP_MODEL_PATH,
                torch_dtype=self.torch_dtype
            ).to(self.device)
            logger.info("成功从本地加载CLIP图像编码器")
        except Exception as e:
            logger.error(f"从本地加载模型失败: {e}")
            raise RuntimeError(f"加载CLIP模型失败: {e}")
    
    def preprocess_image(self, image, target_size=224):
        """
        预处理输入图像
        
        Args:
            image: PIL图像或图像路径
            target_size: 目标尺寸
            
        Returns:
            处理后的PIL图像
        """
        if isinstance(image, str):
            if os.path.exists(image):
                image = Image.open(image).convert("RGB")
            else:
                raise ValueError(f"图像路径不存在: {image}")
        
        if not isinstance(image, Image.Image):
            raise ValueError("image必须是PIL.Image.Image类型或有效的图像路径")
        
        # 确保图像尺寸合适
        if image.size != (target_size, target_size):
            image = image.resize((target_size, target_size), Image.LANCZOS)
        
        return image
    
    def encode_image(self, image):
        """
        编码图像为特征向量
        
        Args:
            image: PIL图像或图像路径
            
        Returns:
            图像特征向量
        """
        image = self.preprocess_image(image)
        inputs = self.image_processor(images=image, return_tensors="pt").to(
            self.device, dtype=self.torch_dtype
        )
        with torch.no_grad():
            image_embeds = self.image_encoder(**inputs).image_embeds
        return image_embeds
    
    @abstractmethod
    def generate(self, *args, **kwargs):
        """生成图像的抽象方法，需要子类实现"""
        pass
    
    def save_image(self, image, output_path):
        """
        保存生成的图像
        
        Args:
            image: 生成的PIL图像
            output_path: 输出路径
        
        Returns:
            保存的图像路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存图像
        image.save(output_path)
        logger.info(f"图像已保存到: {output_path}")
        
        return output_path
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline
        if hasattr(self, 'image_encoder') and self.image_encoder is not None:
            del self.image_encoder
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 