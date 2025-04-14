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
    
    def _load_ip_adapter(self):
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
        
        # 确定模型文件名
        model_filename = model_type_map[self.ip_model_type]
        
        # 构建完整的模型文件路径
        ip_adapter_dir = Path(self.ip_adapter_path)
        
        # 如果目录中包含models子目录，添加到路径中
        models_dir = ip_adapter_dir / "models"
        if models_dir.exists() and models_dir.is_dir():
            model_path = models_dir / model_filename
            ip_adapter_dir = models_dir
        else:
            model_path = ip_adapter_dir / model_filename
        
        logger.info(f"加载IP-Adapter模型: {model_path}")
        logger.info(f"使用模型类型: {self.ip_model_type}")
        
        # 检查模型文件是否存在
        if not model_path.exists():
            available_models = []
            try:
                if models_dir.exists() and models_dir.is_dir():
                    available_models = [f.name for f in models_dir.glob("*.safetensors") if f.is_file()]
                elif ip_adapter_dir.exists() and ip_adapter_dir.is_dir():
                    available_models = [f.name for f in ip_adapter_dir.glob("*.safetensors") if f.is_file()]
                
                if available_models:
                    logger.error(f"模型文件 {model_filename} 不存在，可用模型有: {', '.join(available_models)}")
                else:
                    logger.error(f"模型文件 {model_filename} 不存在，且目录中没有找到可用的.safetensors模型文件")
            except Exception as e:
                logger.error(f"检查可用模型时出错: {str(e)}")
            
            raise FileNotFoundError(f"无法找到IP-Adapter模型文件: {model_path}")
        
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