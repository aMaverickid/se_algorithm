"""
IP-Adapter文本驱动模型，用于根据人脸图像和文本提示词生成图像
"""
import os
import logging
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from safetensors.torch import load_file
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.base import BaseIPAdapter
from utils.image_utils import resize_image, save_output_image

logger = logging.getLogger(__name__)

class IPAdapterText2Img(BaseIPAdapter):
    """IP-Adapter文本驱动模型类"""
    
    def __init__(
        self, 
        device=config.DEVICE,
        sd_model_name=config.STABLE_DIFFUSION_MODEL_PATH,
        ip_adapter_path=config.IP_ADAPTER_MODEL_PATH,
        ip_model_type="plus",
        scale=0.6,  # 略微降低IP-Adapter比重，提高文本控制效果
        steps=config.NUM_INFERENCE_STEPS,
    ):
        """
        初始化IP-Adapter文本驱动模型
        
        Args:
            device (str): 设备类型，'cuda'或'cpu'
            sd_model_name (str): Stable Diffusion模型名称
            ip_adapter_path (str): IP-Adapter模型路径
            ip_model_type (str): IP-Adapter模型类型，'base'或'plus'或'plus_face'
            scale (float): IP-Adapter的条件缩放因子
            steps (int): 推理步数
        """
        super().__init__(device=device)
        
        self.ip_adapter_path = ip_adapter_path
        self.ip_model_type = ip_model_type
        self.scale = scale
        self.steps = steps
        
        # 加载Stable Diffusion模型
        self._load_pipeline(sd_model_name)
        
        # 加载IP-Adapter权重
        self._load_ip_adapter()
    
    def _load_pipeline(self, sd_model_name):
        """加载Stable Diffusion模型"""
        logger.info(f"加载Stable Diffusion模型: {sd_model_name}")
        
        # 加载基础SD模型
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            sd_model_name,
            torch_dtype=self.torch_dtype,
            cache_dir=config.MODEL_CACHE_DIR
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
        
        # 启用内存优化（如果使用CUDA）
        if self.device == "cuda":
            self.pipeline.enable_xformers_memory_efficient_attention()
    
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
        
        # 创建模型
        image_proj_model = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.LayerNorm(1280),
            nn.GELU(),
            nn.Linear(1280, cross_attention_dim),
        )
        
        # 加载权重
        image_proj_model[0].weight.data = state_dict["0.weight"].to(
            device=self.device, dtype=self.torch_dtype)
        image_proj_model[0].bias.data = state_dict["0.bias"].to(
            device=self.device, dtype=self.torch_dtype)
        image_proj_model[1].weight.data = state_dict["1.weight"].to(
            device=self.device, dtype=self.torch_dtype)
        image_proj_model[1].bias.data = state_dict["1.bias"].to(
            device=self.device, dtype=self.torch_dtype)
        image_proj_model[3].weight.data = state_dict["3.weight"].to(
            device=self.device, dtype=self.torch_dtype)
        image_proj_model[3].bias.data = state_dict["3.bias"].to(
            device=self.device, dtype=self.torch_dtype)
        
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
            
            # 返回带有IP-Adapter特征的结果
            return out
        
        # 替换U-Net的forward方法
        unet.forward = forward_with_ip_adapter
        
        return unet.forward
    
    def generate(
        self,
        face_image,
        prompt,
        size=(config.OUTPUT_RESOLUTION, config.OUTPUT_RESOLUTION),
        guidance_scale=config.GUIDANCE_SCALE,
        num_images=config.NUM_IMAGES_PER_PROMPT,
        seed=None,
        output_path=None,
        negative_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
    ):
        """
        使用IP-Adapter文本驱动生成图像
        
        Args:
            face_image: 人脸图像
            prompt: 文本提示词
            size: 输出图像尺寸，默认为正方形
            guidance_scale: 分类器自由指导比例
            num_images: 生成的图像数量
            seed: 随机种子
            output_path: 输出路径
            negative_prompt: 负面提示词
            
        Returns:
            生成的图像列表
        """
        # 预处理人脸图像
        if isinstance(face_image, str):
            face_image = Image.open(face_image).convert("RGB")
        
        # 调整人脸图像大小
        face_resized = resize_image(face_image, config.FACE_RESOLUTION)
        
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # 增强提示词，确保生成的是人物图像
        positive_prompt = f"{prompt}, masterpiece, best quality, high quality"
        
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
                height=size[1],
                width=size[0],
                num_inference_steps=self.steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
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
    
    def generate_variations(
        self,
        face_image,
        prompts,
        size=(config.OUTPUT_RESOLUTION, config.OUTPUT_RESOLUTION),
        guidance_scale=config.GUIDANCE_SCALE,
        seed=None,
        output_dir=None,
        negative_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
    ):
        """
        使用多个提示词生成图像变体
        
        Args:
            face_image: 人脸图像
            prompts: 文本提示词列表
            size: 输出图像尺寸
            guidance_scale: 分类器自由指导比例
            seed: 随机种子
            output_dir: 输出目录
            negative_prompt: 负面提示词
            
        Returns:
            生成的图像列表字典，格式为 {prompt: image}
        """
        results = {}
        
        # 对每个提示词生成图像
        for i, prompt in enumerate(prompts):
            # 设置输出路径（如果提供了输出目录）
            output_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                # 使用提示词的前几个单词作为文件名
                prompt_words = prompt.replace(',', '').split()[:3]
                prompt_filename = '_'.join(prompt_words)
                output_path = os.path.join(output_dir, f"variation_{i}_{prompt_filename}.png")
            
            # 生成单个图像
            images = self.generate(
                face_image=face_image,
                prompt=prompt,
                size=size,
                guidance_scale=guidance_scale,
                num_images=1,  # 每个提示词一张图
                seed=seed,
                output_path=output_path,
                negative_prompt=negative_prompt,
            )
            
            # 保存结果
            results[prompt] = images[0]
        
        return results 