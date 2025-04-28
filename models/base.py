"""
IP-Adapter模型的基础类
"""
import os
import logging
import torch
from abc import ABC, abstractmethod
from pathlib import Path
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
        self.pipeline = None # 扩散模型
        self.scale = 0.7  # 默认IP-Adapter缩放因子
        self.ip_model_type = "plus"  # 默认IP-Adapter模型类型 其类型包括base, plus, plus_face
        self.ip_adapter_path = None  # 将在子类中设置具体路径
    
    def load_ip_adapter(self):
        """加载IP-Adapter模型"""
        # 模型类型到文件名的映射
        model_type_map = {
            "base": "ip-adapter_sd15.safetensors",
            "plus": "ip-adapter-plus_sd15.safetensors",
            "plus_face": "ip-adapter-plus-face_sd15.safetensors",
            "full_face": "ip-adapter-full-face_sd15.safetensors"
        }
        
        # 确保模型类型有效
        if self.ip_model_type not in model_type_map:
            logger.warning(f"未知的IP-Adapter模型类型: {self.ip_model_type}，默认使用'plus'")
            self.ip_model_type = "plus"
        
        model_filename = model_type_map[self.ip_model_type]
        ip_adapter_dir = Path(self.ip_adapter_path)
        
        models_dir = ip_adapter_dir / "models"
        model_path = models_dir / model_filename
        ip_adapter_dir = models_dir
        
        logger.info(f"加载IP-Adapter模型: {model_path}")
        logger.info(f"使用模型类型: {self.ip_model_type}")
        
        # 加载IP-Adapter模型
        self.pipeline.load_ip_adapter(
            pretrained_model_name_or_path_or_dict=str(ip_adapter_dir),
            subfolder="",
            weight_name=model_filename
        )
    
    @abstractmethod
    def generate(self, *args, **kwargs):
        """生成图像的抽象方法，需要子类实现"""
        pass
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 