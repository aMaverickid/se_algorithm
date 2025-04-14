"""
IP-Adapter ControlNet模型，用于通过深度图控制人脸图像的生成
"""
import os
import logging
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler
from diffusers.utils import load_image
from safetensors.torch import load_file
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.base import BaseIPAdapter
from utils.image_utils import resize_image, save_output_image, get_sample_templates
from processors.depth_estimator import DepthEstimator

logger = logging.getLogger(__name__)

class IPAdapterControlNet(BaseIPAdapter):
    """IP-Adapter ControlNet深度控制模型类"""
    
    def __init__(
        self, 
        device=config.DEVICE,
        sd_model_name=config.STABLE_DIFFUSION_MODEL_PATH,
        controlnet_model_name=config.CONTROLNET_DEPTH_MODEL_PATH,
        ip_adapter_path=config.IP_ADAPTER_MODEL_PATH,
        ip_model_type="plus",
        scale=0.7,
        steps=config.NUM_INFERENCE_STEPS,
    ):
        """
        初始化IP-Adapter ControlNet模型
        
        Args:
            device (str): 设备类型，'cuda'或'cpu'
            sd_model_name (str): Stable Diffusion模型名称
            controlnet_model_name (str): ControlNet模型名称
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
        
        # 初始化深度估计器
        self.depth_estimator = DepthEstimator(device=device)
        
        # 加载ControlNet和Stable Diffusion模型
        self._load_pipeline(sd_model_name, controlnet_model_name)
        
        # 加载IP-Adapter权重
        self._load_ip_adapter()
    
    def _load_pipeline(self, sd_model_name, controlnet_model_name):
        """加载Stable Diffusion和ControlNet模型"""
        logger.info(f"加载ControlNet模型: {controlnet_model_name}")
        
        # 加载ControlNet深度模型
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_name,
            torch_dtype=self.torch_dtype,
            cache_dir=config.MODEL_CACHE_DIR
        )
        
        logger.info(f"加载Stable Diffusion模型: {sd_model_name}")
        
        # 加载基础SD模型与ControlNet
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model_name,
            controlnet=controlnet,
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
    
    def generate(
        self,
        face_image,
        depth_image=None,
        controlnet_conditioning_scale=1.0,
        guidance_scale=config.GUIDANCE_SCALE,
        num_images=config.NUM_IMAGES_PER_PROMPT,
        seed=None,
        output_path=None,
        positive_prompt="masterpiece, best quality, high quality, photorealistic",
        negative_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
    ):
        """
        使用IP-Adapter ControlNet生成图像
        
        Args:
            face_image: 人脸图像
            depth_image: 深度控制图像，如果为None，将自动从人脸生成深度图
            controlnet_conditioning_scale: ControlNet条件权重
            guidance_scale: 分类器自由指导比例
            num_images: 生成的图像数量
            seed: 随机种子
            output_path: 输出路径
            positive_prompt: 正面提示词
            negative_prompt: 负面提示词
            
        Returns:
            生成的图像列表
        """
        # 预处理人脸图像
        if isinstance(face_image, str):
            face_image = Image.open(face_image).convert("RGB")
        
        face_resized = resize_image(face_image, config.FACE_RESOLUTION)
        
        # 处理深度图
        if depth_image is None:
            logger.info("正在从人脸生成深度图")
            depth_image = self.depth_estimator.estimate_depth(face_image)
        elif isinstance(depth_image, str):
            depth_image = Image.open(depth_image).convert("RGB")
        
        # 调整深度图大小
        depth_resized = resize_image(depth_image, config.OUTPUT_RESOLUTION)
        
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # 设置IP-Adapter缩放因子
        self.pipeline.set_ip_adapter_scale(self.scale)
        
        # 生成图像
        try:
            output = self.pipeline(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image=depth_resized,
                num_inference_steps=self.steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                ip_adapter_image=face_resized,
            )
            
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
            logger.error(f"生成过程中发生错误: {str(e)}")
            raise
    
    def get_sample_depth_templates(self, n=3):
        """
        获取示例深度模板
        
        Args:
            n (int): 返回的模板数量
            
        Returns:
            模板路径列表
        """
        return get_sample_templates("depth", n) 