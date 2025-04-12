"""
模型工具函数
"""
import os
import logging
import torch
import json
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

def get_device():
    """
    获取当前可用的设备
    
    Returns:
        设备类型字符串，'cuda'或'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    else:
        logger.warning("CUDA不可用，将使用CPU进行推理。这可能会导致性能显著降低。")
        return "cpu"

def get_torch_dtype(device=config.DEVICE):
    """
    根据设备获取合适的PyTorch数据类型
    
    Args:
        device (str): 设备类型
        
    Returns:
        PyTorch数据类型
    """
    if device == "cuda":
        return torch.float16
    else:
        return torch.float32

def get_ip_adapter_models(ip_adapter_path):
    """
    获取IP-Adapter模型的文件路径
    
    Args:
        ip_adapter_path: IP-Adapter模型目录路径
        
    Returns:
        包含模型文件路径的字典
    """
    if isinstance(ip_adapter_path, str):
        ip_adapter_path = Path(ip_adapter_path)
    
    # 定义常见的IP-Adapter模型文件名
    model_files = {
        "image_proj": "image_proj.safetensors",
        "ip_adapter": "ip_adapter.safetensors",
        "ip_adapter_plus": "ip_adapter_plus.safetensors",
        "ip_adapter_plus_face": "ip_adapter_plus_face.safetensors",
        "ip_adapter_full_face": "ip_adapter_full_face.safetensors",
        "ip_adapter_composition": "ip_adapter_composition.safetensors"
    }
    
    # 检查文件是否存在并返回完整路径
    model_paths = {}
    for key, filename in model_files.items():
        file_path = ip_adapter_path / filename
        if file_path.exists():
            model_paths[key] = str(file_path)
    
    if not model_paths:
        logger.warning(f"在{ip_adapter_path}中未找到任何IP-Adapter模型文件")
    
    return model_paths

def save_model_config(model_name, model_config, config_dir=None):
    """
    保存模型配置到JSON文件
    
    Args:
        model_name (str): 模型名称
        model_config (dict): 模型配置
        config_dir: 配置文件保存目录
        
    Returns:
        配置文件路径
    """
    if config_dir is None:
        config_dir = Path(config.MODEL_CACHE_DIR) / "configs"
    
    # 确保目录存在
    os.makedirs(config_dir, exist_ok=True)
    
    # 配置文件路径
    config_path = Path(config_dir) / f"{model_name}_config.json"
    
    # 保存配置
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=4)
    
    return str(config_path)

def load_model_config(model_name, config_dir=None):
    """
    从JSON文件加载模型配置
    
    Args:
        model_name (str): 模型名称
        config_dir: 配置文件目录
        
    Returns:
        模型配置字典，如果文件不存在则返回None
    """
    if config_dir is None:
        config_dir = Path(config.MODEL_CACHE_DIR) / "configs"
    
    # 配置文件路径
    config_path = Path(config_dir) / f"{model_name}_config.json"
    
    # 检查文件是否存在
    if not config_path.exists():
        logger.warning(f"模型配置文件不存在: {config_path}")
        return None
    
    # 加载配置
    with open(config_path, "r") as f:
        model_config = json.load(f)
    
    return model_config

def cleanup_memory():
    """
    清理GPU内存
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.info("已清理GPU内存")

def get_model_memory_usage(model):
    """
    获取模型的内存使用量
    
    Args:
        model: PyTorch模型
        
    Returns:
        内存使用量（MB）
    """
    if not torch.cuda.is_available():
        return 0
    
    # 计算模型参数占用的内存
    mem_params = sum([p.nelement() * p.element_size() for p in model.parameters()])
    
    # 计算模型缓冲区占用的内存
    mem_bufs = sum([b.nelement() * b.element_size() for b in model.buffers()])
    
    # 转换为MB
    mem_total = (mem_params + mem_bufs) / 1024 / 1024
    
    return mem_total 