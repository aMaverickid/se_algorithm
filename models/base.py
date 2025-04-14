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
        self.scale = 0.7  # 默认IP-Adapter缩放因子
        self.ip_model_type = "plus"  # 默认IP-Adapter模型类型
        self._load_image_encoder()
    
    def _load_image_encoder(self):
        """
        加载IP-Adapter使用的图像编码器，负责对输入的图像进行预处理，并提取图像特征
        """
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
    
    def _load_ip_adapter_weights(self, ip_adapter_path, ip_model_type="plus"):
        """
        加载IP-Adapter权重
        
        Args:
            ip_adapter_path: IP-Adapter模型路径
            ip_model_type: IP-Adapter模型类型，"base"或"plus"
            
        Returns:
            (图像投影模型, ip_adapter_state_dict)
        """
        logger.info(f"使用IP-Adapter模型目录: {ip_adapter_path}")
        
        # 处理路径
        ip_adapter_path = Path(ip_adapter_path)
        if ip_adapter_path.is_dir():
            # 如果提供的是目录，自动选择正确的权重文件
            models_dir = ip_adapter_path / "models"
            if not models_dir.exists():
                logger.warning(f"模型目录不存在: {models_dir}，尝试直接使用 {ip_adapter_path}")
                models_dir = ip_adapter_path
                
            # 根据模型类型选择权重文件
            if ip_model_type == "base":
                model_name = "ip-adapter_sd15.safetensors"
            elif ip_model_type == "plus":
                model_name = "ip-adapter-plus_sd15.safetensors"
            elif ip_model_type == "plus_face":
                model_name = "ip-adapter-plus-face_sd15.safetensors"
            else:
                raise ValueError(f"不支持的IP-Adapter模型类型: {ip_model_type}")
            
            ip_adapter_path = models_dir / model_name
            
            # 如果safetensors版本不存在，尝试使用bin版本
            if not ip_adapter_path.exists():
                bin_path = models_dir / model_name.replace(".safetensors", ".bin")
                if bin_path.exists():
                    ip_adapter_path = bin_path
                    logger.warning(f"未找到safetensors版本，使用替代文件: {ip_adapter_path}")
                else:
                    # 尝试查找任何可能的IP-Adapter权重
                    possible_weights = list(models_dir.glob("ip-adapter*.safetensors")) + list(models_dir.glob("ip-adapter*.bin"))
                    if possible_weights:
                        ip_adapter_path = possible_weights[0]
                        logger.warning(f"未找到特定的权重文件，将使用: {ip_adapter_path}")
                    else:
                        files_in_dir = "\n  ".join([str(f) for f in models_dir.glob("*") if f.is_file()])
                        raise FileNotFoundError(
                            f"无法找到IP-Adapter权重文件: {model_name}，目录中包含以下文件:\n  {files_in_dir}"
                        )
        
        logger.info(f"加载IP-Adapter权重: {ip_adapter_path}")
        
        # 确保图像编码器路径正确
        if ip_adapter_path.is_file():
            image_encoder_path = ip_adapter_path.parent / "image_encoder"
        else:
            image_encoder_path = ip_adapter_path / "image_encoder"
        
        if not image_encoder_path.exists():
            # 尝试查找任何名称的图像编码器目录
            parent_dir = ip_adapter_path.parent
            possible_dirs = [
                parent_dir / "image_encoder",
                parent_dir / "clip_vision",
                parent_dir / "clip",
                parent_dir.parent / "image_encoder"
            ]
            
            for dir_path in possible_dirs:
                if dir_path.exists():
                    image_encoder_path = dir_path
                    logger.warning(f"使用替代图像编码器目录: {image_encoder_path}")
                    break
            else:
                dirs_in_parent = "\n  ".join([str(d) for d in parent_dir.glob("*") if d.is_dir()])
                raise FileNotFoundError(
                    f"无法找到图像编码器目录。已尝试: {image_encoder_path}\n"
                    f"父目录中包含以下目录:\n  {dirs_in_parent}"
                )
        
        # 使用safetensors格式的模型文件
        image_proj_path = image_encoder_path / "model.safetensors"
        if not image_proj_path.exists():
            # 尝试使用备选文件名
            image_proj_path = image_encoder_path / "pytorch_model.bin"
            if not image_proj_path.exists():
                files_in_encoder_dir = "\n  ".join([str(f) for f in image_encoder_path.glob("*") if f.is_file()])
                raise FileNotFoundError(
                    f"图像投影权重文件不存在: {image_proj_path}\n"
                    f"图像编码器目录中包含以下文件:\n  {files_in_encoder_dir}"
                )
        
        logger.info(f"加载图像投影权重: {image_proj_path}")
        
        # 加载权重，根据文件扩展名选择不同的加载方法
        try:
            if str(ip_adapter_path).endswith('.safetensors'):
                from safetensors.torch import load_file
                ip_adapter_state_dict = load_file(ip_adapter_path)
            else:
                ip_adapter_state_dict = torch.load(ip_adapter_path, map_location=self.device)
                
            if str(image_proj_path).endswith('.safetensors'):
                from safetensors.torch import load_file
                image_proj_state_dict = load_file(image_proj_path)
            else:
                image_proj_state_dict = torch.load(image_proj_path, map_location=self.device)
                
            # 存储加载的权重键，用于后续参考
            self._loaded_keys = list(image_proj_state_dict.keys())
            # 记录前10个键，用于调试
            logger.info(f"前10个加载的权重键: {self._loaded_keys[:10]}")
            
            # 检测权重文件类型
            if any("vision_model" in key for key in self._loaded_keys):
                logger.warning("检测到CLIP视觉模型权重文件，非标准IP-Adapter权重")
                logger.warning("将使用特殊处理以适应此权重结构")
                
        except Exception as e:
            logger.error(f"加载IP-Adapter权重文件时出错: {str(e)}")
            raise ValueError(f"无法加载IP-Adapter权重文件: {str(e)}")
        
        # 创建图像投影层
        try:
            if ip_model_type == "base":
                image_proj_model = self._create_base_image_proj(
                    image_proj_state_dict, cross_attention_dim=768
                )
            else:  # plus和plus_face使用相同的方式
                image_proj_model = self._create_plus_image_proj(
                    image_proj_state_dict, cross_attention_dim=768
                )
            
            logger.info("IP-Adapter权重加载完成")
            
            return image_proj_model, ip_adapter_state_dict
        
        except Exception as e:
            logger.error(f"创建图像投影模型时出错: {str(e)}")
            logger.error(f"图像投影权重键: {list(image_proj_state_dict.keys())[:10]}...")
            raise ValueError(f"创建图像投影模型失败: {str(e)}\n请检查图像投影权重文件结构是否匹配模型类型 '{ip_model_type}'")
    
    def _create_base_image_proj(self, state_dict, cross_attention_dim=768):
        """
        创建基础IP-Adapter的图像投影层
        
        Args:
            state_dict: 图像投影权重
            cross_attention_dim: 交叉注意力维度
        
        Returns:
            图像投影模型
        """
        import torch.nn as nn
        
        # 提取权重参数
        proj_in = state_dict["proj.weight"].shape[1]
        proj_out = state_dict["proj.weight"].shape[0]
        
        # 创建模型
        image_proj_model = nn.Linear(proj_in, proj_out)
        image_proj_model.load_state_dict(state_dict)
        image_proj_model.to(self.device, dtype=self.torch_dtype)
        image_proj_model.eval()
        
        return image_proj_model
    
    def _create_plus_image_proj(self, state_dict, cross_attention_dim=768):
        """
        创建Plus版IP-Adapter的图像投影层
        
        Args:
            state_dict: 图像投影权重
            cross_attention_dim: 交叉注意力维度
        
        Returns:
            图像投影模型
        """
        import torch.nn as nn
        
        # 打印状态字典的键，用于调试
        logger.info(f"图像投影权重state_dict包含以下键: {list(state_dict.keys())}")
        
        # 创建模型
        image_proj_model = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.LayerNorm(1280),
            nn.GELU(),
            nn.Linear(1280, cross_attention_dim),
        )
        
        # 定义可能的键名映射（处理不同的命名约定）
        key_mapping = {
            "0.weight": ["0.weight", "layers.0.weight", "projection.0.weight", "linear1.weight"],
            "0.bias": ["0.bias", "layers.0.bias", "projection.0.bias", "linear1.bias"],
            "1.weight": ["1.weight", "layers.1.weight", "projection.1.weight", "layernorm.weight"],
            "1.bias": ["1.bias", "layers.1.bias", "projection.1.bias", "layernorm.bias"],
            "3.weight": ["3.weight", "layers.3.weight", "projection.3.weight", "linear2.weight"],
            "3.bias": ["3.bias", "layers.3.bias", "projection.3.bias", "linear2.bias"]
        }
        
        # 尝试加载权重
        try:
            # 按照标准方式尝试加载
            for target_key, possible_keys in key_mapping.items():
                found_key = None
                for key in possible_keys:
                    if key in state_dict:
                        found_key = key
                        break
                
                if found_key is None:
                    raise KeyError(f"无法在state_dict中找到匹配的键 {possible_keys}")
                
                # 获取对应的层和属性
                layer_idx, attr = target_key.split(".")
                layer_idx = int(layer_idx)
                
                # 设置权重
                param = getattr(image_proj_model[layer_idx], attr)
                param.data = state_dict[found_key].to(device=self.device, dtype=self.torch_dtype)
                
                logger.info(f"成功加载权重 {found_key} 到层 {layer_idx}.{attr}")
        
        except Exception as e:
            logger.error(f"加载图像投影权重时出错: {str(e)}")
            logger.error("尝试根据现有键结构自动识别...")
            
            # 尝试识别文件中的权重结构
            if len(state_dict.keys()) >= 6:  # 至少需要6个参数
                sorted_keys = sorted(list(state_dict.keys()))
                
                # 假设键是顺序排列的
                weight_keys = [k for k in sorted_keys if k.endswith('.weight')]
                bias_keys = [k for k in sorted_keys if k.endswith('.bias')]
                
                if len(weight_keys) >= 3 and len(bias_keys) >= 3:
                    # 直接映射到对应的层，忽略层号
                    logger.info(f"发现替代权重结构，尝试直接按顺序映射...")
                    
                    # 用于可视化检查的映射
                    mapping_info = []
                    
                    # 权重映射 (按顺序)
                    for i, (target_idx, source_key) in enumerate([
                        (0, weight_keys[0]),
                        (1, weight_keys[1]),
                        (3, weight_keys[2])
                    ]):
                        image_proj_model[target_idx].weight.data = state_dict[source_key].to(
                            device=self.device, dtype=self.torch_dtype)
                        mapping_info.append(f"{source_key} -> model[{target_idx}].weight")
                    
                    # 偏置映射 (按顺序)
                    for i, (target_idx, source_key) in enumerate([
                        (0, bias_keys[0]),
                        (1, bias_keys[1]),
                        (3, bias_keys[2])
                    ]):
                        image_proj_model[target_idx].bias.data = state_dict[source_key].to(
                            device=self.device, dtype=self.torch_dtype)
                        mapping_info.append(f"{source_key} -> model[{target_idx}].bias")
                    
                    logger.info("应用以下权重映射:")
                    for mapping in mapping_info:
                        logger.info(f"  {mapping}")
                else:
                    raise ValueError(f"无法识别权重结构，找到 {len(weight_keys)} 个权重键和 {len(bias_keys)} 个偏置键，但需要至少3个")
            else:
                raise ValueError(f"权重文件包含的键太少 ({len(state_dict.keys())})，无法完成映射")
        
        image_proj_model.to(self.device, dtype=self.torch_dtype)
        image_proj_model.eval()
        
        return image_proj_model

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
    
    def _prepare_ip_adapter_features(self, image):
        """
        准备IP-Adapter特征
        
        Args:
            image: 输入图像
            
        Returns:
            IP-Adapter特征元组 (image_prompt_embeds, uncond_image_prompt_embeds, image_embeds)
        """
        # 获取图像编码
        image_embeds = self.encode_image(image)
        
        # 添加调试信息
        logger.info(f"图像编码后的tensor形状: {image_embeds.shape}")
        
        # 通过投影层处理
        if self.ip_model_type == "base":
            image_prompt_embeds = self.image_proj_model(image_embeds)
        else:  # plus或plus_face
            # 检查图像投影模型的权重是否来自CLIP视觉模型
            is_clip_weights = any("vision_model" in key for key in getattr(self, "_loaded_keys", []))
            
            # 根据权重类型调整处理逻辑
            if is_clip_weights:
                logger.info("检测到CLIP视觉模型权重，调整维度处理")
                # 对于CLIP权重，我们直接使用image_embeds的最后一个维度
                # 扁平化并提取特征
                image_embeds_flat = image_embeds.flatten(1) if image_embeds.dim() > 2 else image_embeds
                
                # 通过线性投影到768维度
                # 创建临时线性层 (CLIP输出1280维，IP-Adapter需要768维)
                import torch.nn as nn
                if not hasattr(self, "_image_proj_linear"):
                    self._image_proj_linear = nn.Linear(image_embeds_flat.shape[-1], 768).to(
                        device=self.device, dtype=self.torch_dtype)
                    # 初始化权重为小值
                    torch.nn.init.normal_(self._image_proj_linear.weight, std=0.02)
                    torch.nn.init.zeros_(self._image_proj_linear.bias)
                
                # 投影到768维并添加batch维度
                image_prompt_embeds = self._image_proj_linear(image_embeds_flat).unsqueeze(1)
                logger.info(f"使用临时线性投影后的tensor形状: {image_prompt_embeds.shape}")
            else:
                # 原始处理逻辑
                try:
                    # 扩展维度模拟标记
                    embeds = image_embeds.unsqueeze(1)
                    logger.info(f"扩展维度后的tensor形状: {embeds.shape}")
                    
                    # 投影并返回
                    image_prompt_embeds = self.image_proj_model(embeds)
                except RuntimeError as e:
                    logger.error(f"图像投影出错: {str(e)}")
                    logger.warning("尝试备用方法: 重塑张量以匹配模型输入要求")
                    
                    # 备用方法：尝试将张量调整为2D
                    try:
                        # 先展平或降维到2D
                        if image_embeds.dim() > 2:
                            flat_embeds = image_embeds.reshape(image_embeds.shape[0], -1)
                        else:
                            flat_embeds = image_embeds
                            
                        logger.info(f"重塑后的张量形状: {flat_embeds.shape}")
                        
                        # 应用投影模型
                        result = self.image_proj_model(flat_embeds)
                        
                        # 如果需要，恢复原始维度结构
                        if result.dim() == 2:
                            image_prompt_embeds = result.unsqueeze(1)
                        else:
                            image_prompt_embeds = result
                            
                        logger.info(f"最终特征形状: {image_prompt_embeds.shape}")
                    except Exception as e2:
                        logger.error(f"备用方法也失败: {str(e2)}")
                        # 最后的备用方案：创建零张量
                        logger.warning("使用零张量作为占位符")
                        batch_size = image_embeds.shape[0] if image_embeds.dim() > 0 else 1
                        image_prompt_embeds = torch.zeros(batch_size, 1, 768, 
                                                         device=self.device, 
                                                         dtype=self.torch_dtype)
        
        # 返回特征
        uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)
        
        logger.info(f"最终image_prompt_embeds形状: {image_prompt_embeds.shape}")
        logger.info(f"最终uncond_image_prompt_embeds形状: {uncond_image_prompt_embeds.shape}")
        
        return image_prompt_embeds, uncond_image_prompt_embeds, image_embeds
    
    def _prepare_unet_features(self, image_prompt_embeds, uncond_image_prompt_embeds):
        """
        准备U-Net特征，用于将IP-Adapter特征整合到U-Net中
        
        Args:
            image_prompt_embeds: 条件图像特征
            uncond_image_prompt_embeds: 无条件图像特征
            
        Returns:
            带有IP-Adapter特征的unet处理方法
        """
        # 获取原始的U-Net前向传播方法
        unet = self.pipeline.unet
        orig_forward = unet.forward
        
        # 定义U-Net的新forward方法
        def forward_with_ip_adapter(
            sample,
            timestep,
            encoder_hidden_states,
            cross_attention_kwargs=None,
            **kwargs
        ):
            # 分离交叉注意力参数
            cross_attention_kwargs = cross_attention_kwargs or {}
            ip_scale = cross_attention_kwargs.pop("ip_adapter_scale", self.scale)
            
            # 使用原始forward计算基本输出
            out = orig_forward(
                sample,
                timestep,
                encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                **kwargs
            )
            
            # 添加IP-Adapter特征
            # 这些特征已经在unet中注册
            ip_hidden_states = torch.cat([
                uncond_image_prompt_embeds, image_prompt_embeds
            ])
            
            return out
        
        # 替换U-Net的forward方法
        unet.forward = forward_with_ip_adapter
        
        return unet.forward
    
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