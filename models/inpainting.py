"""
IP-Adapter Inpainting模型，用于将人脸图像与无脸模板合成
"""
import os
import logging
import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.base import BaseIPAdapter
from utils.image_utils import resize_image, save_output_image, get_sample_templates
from processors.mask_generator import MaskGenerator

logger = logging.getLogger(__name__)

class IPAdapterInpainting(BaseIPAdapter):
    """IP-Adapter Inpainting模型类"""
    
    def __init__(
        self, 
        device=config.DEVICE,
        sd_model_path=config.STABLE_DIFFUSION_MODEL_PATH,
        ip_adapter_path=config.IP_ADAPTER_MODEL_PATH,
        ip_model_type="plus",
        scale=0.7,
        steps=config.NUM_INFERENCE_STEPS,
    ):
        """
        初始化IP-Adapter Inpainting模型
        
        Args:
            device (str): 设备类型，'cuda'或'cpu'
            sd_model_path (str): Stable Diffusion 模型路径
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
        self._load_pipeline(sd_model_path)
        # 加载IP-Adapter权重
        self._load_ip_adapter()
    
    def _load_pipeline(self, sd_model_path):
        """加载Stable Diffusion Inpainting模型"""
        logger.info(f"加载Stable Diffusion Inpainting模型: {sd_model_path}")
        
        # 加载基础inpainting模型
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            sd_model_path,
            torch_dtype=self.torch_dtype,
        )
        
        # 使用DDIM调度器，用来去除噪声
        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config
        )
        
        # 设置为推理模式并移动到设备上
        self.pipeline.to(self.device)
        self.pipeline.unet.eval()
        self.pipeline.vae.eval()
        self.pipeline.text_encoder.eval()
    
    def generate(
        self,
        face_image,
        template_image,
        mask=None,
        strength=config.INPAINTING_STRENGTH,
        guidance_scale=config.GUIDANCE_SCALE,
        num_images=config.NUM_IMAGES_PER_PROMPT,
        seed=config.SEED,
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
        
        # 设置IP-Adapter缩放因子
        self.pipeline.set_ip_adapter_scale(self.scale)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
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
                ip_adapter_image=face_resized,
            )
            
            # 处理输出
            images = output.images
            
            # 保存图像
            import uuid
            inpainting_id = str(uuid.uuid4())
            for i, image in enumerate(images):
                image.save(f"results/inpainting_{inpainting_id}_{i}.png")
            
            return images
            
        except Exception as e:
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
        return get_sample_templates("inpainting", n) 