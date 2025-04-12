"""
IP-Adapter Inpainting模型，用于将人脸图像与无脸模板合成
"""
import os
import logging
import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from diffusers.utils import load_image
from safetensors.torch import load_file
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.base import BaseIPAdapter
from utils.image_utils import resize_image, save_output_image
from processors.mask_generator import MaskGenerator

logger = logging.getLogger(__name__)

class IPAdapterInpainting(BaseIPAdapter):
    """IP-Adapter Inpainting模型类"""
    
    def __init__(
        self, 
        device=config.DEVICE,
        sd_model_name=config.STABLE_DIFFUSION_MODEL_PATH,
        ip_adapter_path=config.IP_ADAPTER_MODEL_PATH,
        ip_model_type="plus",
        scale=0.7,
        steps=config.NUM_INFERENCE_STEPS,
    ):
        """
        初始化IP-Adapter Inpainting模型
        
        Args:
            device (str): 设备类型，'cuda'或'cpu'
            sd_model_name (str): Stable Diffusion 模型路径
            ip_adapter_path (str): IP-Adapter 模型路径
            ip_model_type (str): IP-Adapter 模型类型，'base'或'plus'或'plus_face'
            scale (float): IP-Adapter的条件缩放因子
            steps (int): 推理步数
        """
        super().__init__(device=device)
        
        self.ip_adapter_path = ip_adapter_path
        self.ip_model_type = ip_model_type
        self.scale = scale
        self.steps = steps
        
        # 初始化掩码生成器
        self.mask_generator = MaskGenerator()
        
        # 加载Stable Diffusion Inpainting模型
        self._load_pipeline(sd_model_name)
        
        # 加载IP-Adapter权重
        self._load_ip_adapter()
    
    def _load_pipeline(self, sd_model_name):
        """加载Stable Diffusion Inpainting模型"""
        logger.info(f"加载Stable Diffusion Inpainting模型: {sd_model_name}")
        
        # 加载基础inpainting模型
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            sd_model_name,
            torch_dtype=self.torch_dtype,
        )
        
        # 使用DDIM调度器
        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        # 设置为推理模式并移动到设备上
        self.pipeline.to(self.device)
        self.pipeline.unet.eval()
        self.pipeline.vae.eval()
        self.pipeline.text_encoder.eval()
    
    def _load_ip_adapter(self):
        """加载IP-Adapter权重"""
        # 确保指向models/cache/IP-Adapter/models目录
        model_path = Path(self.ip_adapter_path) / "models"
        
        if not model_path.exists():
            raise FileNotFoundError(f"IP-Adapter模型目录不存在: {model_path}")
        
        logger.info(f"使用IP-Adapter模型目录: {model_path}")
        
        # 确定要加载的具体IP-Adapter权重
        if self.ip_model_type == "base":
            ip_adapter_path = model_path / "ip-adapter_sd15.safetensors"
        elif self.ip_model_type == "plus":
            ip_adapter_path = model_path / "ip-adapter-plus_sd15.safetensors"
        elif self.ip_model_type == "plus_face":
            ip_adapter_path = model_path / "ip-adapter-plus-face_sd15.safetensors"
        else:
            raise ValueError(f"不支持的IP-Adapter模型类型: {self.ip_model_type}")
        
        # 检查权重文件是否存在
        if not ip_adapter_path.exists():
            raise FileNotFoundError(f"IP-Adapter权重文件不存在: {ip_adapter_path}")
        
        logger.info(f"加载IP-Adapter权重: {ip_adapter_path}")
        
        # 图像编码器目录
        image_encoder_path = model_path / "image_encoder"
        if not image_encoder_path.exists():
            raise FileNotFoundError(f"图像编码器目录不存在: {image_encoder_path}")
        
        # 使用safetensors格式的模型文件
        image_proj_path = image_encoder_path / "model.safetensors"
        if not image_proj_path.exists():
            # 尝试使用备选文件名
            image_proj_path = image_encoder_path / "pytorch_model.bin"
            if not image_proj_path.exists():
                raise FileNotFoundError(f"图像投影权重文件不存在: {image_proj_path}")
        
        logger.info(f"加载图像投影权重: {image_proj_path}")
        
        # 加载权重，根据文件扩展名选择不同的加载方法
        if str(ip_adapter_path).endswith('.safetensors'):
            ip_adapter_state_dict = load_file(ip_adapter_path)
        else:
            ip_adapter_state_dict = torch.load(ip_adapter_path, map_location=self.device)
            
        if str(image_proj_path).endswith('.safetensors'):
            image_proj_state_dict = load_file(image_proj_path)
        else:
            image_proj_state_dict = torch.load(image_proj_path, map_location=self.device)
        
        # 将图像投影权重添加到U-Net中
        self.pipeline.unet.load_state_dict(
            ip_adapter_state_dict, strict=False
        )
        
        # 创建图像投影层
        if self.ip_model_type == "base":
            self.image_proj_model = self._create_base_image_proj(
                image_proj_state_dict, cross_attention_dim=768
            )
        else:  # plus和plus_face使用相同的方式
            self.image_proj_model = self._create_plus_image_proj(
                image_proj_state_dict, cross_attention_dim=768
            )
        
        logger.info("IP-Adapter权重加载完成")
    
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
        
        # 打印状态字典的键以进行调试
        logger.info(f"图像投影状态字典包含以下键: {list(state_dict.keys())}")
        
        # 用于存储映射后的键
        weight_keys = {}
        
        # 检测键名格式并创建映射
        for key in state_dict.keys():
            # 常见的键名格式:
            # "0.weight", "layers.0.weight", "model.0.weight", "projection.0.weight" 等
            if key.endswith('.weight') or key.endswith('.bias'):
                parts = key.split('.')
                # 提取层索引
                for part in parts:
                    if part.isdigit():
                        layer_idx = int(part)
                        suffix = parts[-1]  # weight 或 bias
                        if suffix in ['weight', 'bias']:
                            mapped_key = f"{layer_idx}.{suffix}"
                            weight_keys[mapped_key] = key
                            logger.debug(f"映射键: '{mapped_key}' -> '{key}'")
        
        # 如果没有找到映射键，使用可能的默认键
        if not weight_keys:
            logger.warning("无法识别权重键格式，尝试使用完整键名")
            # 尝试找出可能的权重和偏置键
            weight_keys = {k: k for k in state_dict.keys() if '.weight' in k or '.bias' in k}
        
        # 创建模型
        image_proj_model = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.LayerNorm(1280),
            nn.GELU(),
            nn.Linear(1280, cross_attention_dim),
        )
        
        # 检查键是否存在
        expected_keys = ["0.weight", "0.bias", "1.weight", "1.bias", "3.weight", "3.bias"]
        missing_keys = [k for k in expected_keys if k not in weight_keys]
        
        if missing_keys:
            # 尝试更一般的加载方法
            logger.warning(f"找不到以下键: {missing_keys}，尝试使用Torch加载方法")
            try:
                # 尝试预处理状态字典以匹配Sequential模型预期的格式
                processed_state_dict = {}
                if all('projection.' in k for k in state_dict.keys() if 'weight' in k or 'bias' in k):
                    # 处理projection.X.weight格式
                    for k, v in state_dict.items():
                        if 'projection.' in k:
                            new_key = k.replace('projection.', '')
                            processed_state_dict[new_key] = v
                elif all('model.' in k for k in state_dict.keys() if 'weight' in k or 'bias' in k):
                    # 处理model.X.weight格式
                    for k, v in state_dict.items():
                        if 'model.' in k:
                            new_key = k.replace('model.', '')
                            processed_state_dict[new_key] = v
                else:
                    # 检查是否可以直接使用状态字典
                    try:
                        image_proj_model.load_state_dict(state_dict)
                        logger.info("成功直接加载状态字典")
                        return image_proj_model.to(self.device, dtype=self.torch_dtype).eval()
                    except Exception as e:
                        logger.warning(f"直接加载失败: {str(e)}")
                        processed_state_dict = state_dict
                
                # 尝试加载处理后的状态字典
                image_proj_model.load_state_dict(processed_state_dict, strict=False)
                logger.info("成功加载处理后的状态字典")
            except Exception as e:
                logger.error(f"尝试加载处理后的状态字典失败: {str(e)}")
                # 最后尝试手动加载可用的键
                logger.warning("尝试手动加载可用的权重")
                for layer_idx, layer in enumerate(image_proj_model):
                    if hasattr(layer, 'weight'):
                        weight_key = f"{layer_idx}.weight"
                        bias_key = f"{layer_idx}.bias"
                        # 寻找任何包含layer_idx和weight/bias的键
                        for k in state_dict.keys():
                            if str(layer_idx) in k and 'weight' in k:
                                logger.info(f"手动加载: {k} -> {layer_idx}.weight")
                                layer.weight.data = state_dict[k].to(device=self.device, dtype=self.torch_dtype)
                            if str(layer_idx) in k and 'bias' in k:
                                logger.info(f"手动加载: {k} -> {layer_idx}.bias")
                                layer.bias.data = state_dict[k].to(device=self.device, dtype=self.torch_dtype) 
        else:
            # 使用映射的键加载权重
            try:
                # 加载权重
                image_proj_model[0].weight.data = state_dict[weight_keys.get("0.weight", "0.weight")].to(
                    device=self.device, dtype=self.torch_dtype)
                image_proj_model[0].bias.data = state_dict[weight_keys.get("0.bias", "0.bias")].to(
                    device=self.device, dtype=self.torch_dtype)
                image_proj_model[1].weight.data = state_dict[weight_keys.get("1.weight", "1.weight")].to(
                    device=self.device, dtype=self.torch_dtype)
                image_proj_model[1].bias.data = state_dict[weight_keys.get("1.bias", "1.bias")].to(
                    device=self.device, dtype=self.torch_dtype)
                image_proj_model[3].weight.data = state_dict[weight_keys.get("3.weight", "3.weight")].to(
                    device=self.device, dtype=self.torch_dtype)
                image_proj_model[3].bias.data = state_dict[weight_keys.get("3.bias", "3.bias")].to(
                    device=self.device, dtype=self.torch_dtype)
                
                logger.info("成功使用映射键加载权重")
            except KeyError as e:
                logger.error(f"加载权重时发生KeyError: {str(e)}")
                logger.error(f"可用键: {list(state_dict.keys())}")
                logger.error(f"映射键: {weight_keys}")
                raise KeyError(f"加载图像投影权重失败: {str(e)}. 请检查模型文件格式。")
        
        image_proj_model.to(self.device, dtype=self.torch_dtype)
        image_proj_model.eval()
        
        return image_proj_model
    
    def _prepare_ip_adapter_features(self, face_image):
        """
        准备IP-Adapter特征
        
        Args:
            face_image: 人脸图像
            
        Returns:
            IP-Adapter特征
        """
        # 获取图像编码
        image_embeds = self.encode_image(face_image)
        
        # 通过投影层处理
        if self.ip_model_type == "base":
            image_prompt_embeds = self.image_proj_model(image_embeds)
        else:  # plus或plus_face
            # 扩展维度模拟标记
            embeds = image_embeds.unsqueeze(1)
            # 投影并返回
            image_prompt_embeds = self.image_proj_model(embeds)
        
        # 返回特征
        uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)
        
        return image_prompt_embeds, uncond_image_prompt_embeds, image_embeds
    
    def _prepare_unet_features(self, image_prompt_embeds, uncond_image_prompt_embeds):
        """
        准备U-Net特征
        
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
            ip_mask = kwargs.get("ip_adapter_mask")
            
            # 返回带有IP-Adapter特征的结果
            return out
        
        # 替换U-Net的forward方法
        unet.forward = forward_with_ip_adapter
        
        return unet.forward
    
    def generate(
        self,
        face_image,
        template_image,
        mask=None,
        strength=config.INPAINTING_STRENGTH,
        guidance_scale=config.GUIDANCE_SCALE,
        num_images=config.NUM_IMAGES_PER_PROMPT,
        seed=None,
        output_path=None,
        positive_prompt="masterpiece, best quality, high quality",
        negative_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
    ):
        """
        使用IP-Adapter Inpainting生成图像
        
        Args:
            face_image: 人脸图像
            template_image: 无脸模板图像
            mask: 掩码图像，如果为None则自动生成
            strength: 改变的强度
            guidance_scale: 分类器自由指导比例
            num_images: 生成的图像数量
            seed: 随机种子
            output_path: 输出路径
            positive_prompt: 正面提示词
            negative_prompt: 负面提示词
            
        Returns:
            生成的图像列表
        """
        # 预处理图像
        if isinstance(face_image, str):
            face_image = Image.open(face_image).convert("RGB")
        if isinstance(template_image, str):
            template_image = Image.open(template_image).convert("RGB")
        
        # 调整图像大小
        face_resized = resize_image(face_image, config.FACE_RESOLUTION)
        template_resized = resize_image(template_image, config.OUTPUT_RESOLUTION)
        
        # 如果没有提供掩码，则自动生成
        if mask is None:
            logger.info("自动生成模板掩码")
            mask = self.mask_generator.generate_template_mask(
                template_resized, 
                target_size=template_resized.size
            )
        elif isinstance(mask, str):
            mask = Image.open(mask).convert("L")
            mask = resize_image(mask, template_resized.size, keep_aspect_ratio=False)
        
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # 准备IP-Adapter特征
        image_prompt_embeds, uncond_image_prompt_embeds, _ = self._prepare_ip_adapter_features(face_resized)
        
        # 修改U-Net前向传播以使用IP-Adapter特征
        original_forward = self.pipeline.unet.forward
        self.pipeline.unet.forward = self._prepare_unet_features(image_prompt_embeds, uncond_image_prompt_embeds)
        
        # 生成图像
        try:
            output = self.pipeline(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image=template_resized,
                mask_image=mask,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                num_inference_steps=self.steps,
                strength=strength,
                cross_attention_kwargs={"ip_adapter_scale": self.scale},
            )
            
            # 恢复原始U-Net前向传播
            self.pipeline.unet.forward = original_forward
            
            # 处理输出
            images = output.images
            
            # 保存图像（如果需要）
            if output_path:
                if num_images == 1:
                    save_output_image(images[0], output_path)
                else:
                    for i, image in enumerate(images):
                        path, ext = os.path.splitext(output_path)
                        numbered_path = f"{path}_{i}{ext}"
                        save_output_image(image, numbered_path)
            
            return images
            
        except Exception as e:
            # 确保恢复原始U-Net前向传播
            self.pipeline.unet.forward = original_forward
            logger.error(f"生成过程中发生错误: {str(e)}")
            raise
    
    def get_sample_templates(self, n=3):
        """
        获取示例无脸模板
        
        Args:
            n (int): 返回的模板数量
            
        Returns:
            模板路径列表
        """
        from utils.image_utils import get_sample_templates
        return get_sample_templates("inpainting", n) 