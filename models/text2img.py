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
from utils.image_utils import resize_image, save_output_image

logger = logging.getLogger(__name__)

class IPAdapterText2Img:
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
        self.device = device
        self.torch_dtype = torch.float16 if device == "cuda" else torch.float32
        self.pipeline = None  # 扩散模型
        
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
        
        # 设置IP-Adapter缩放因子
        self.pipeline.set_ip_adapter_scale(self.scale)
        
        # 生成图像
        try:
            output = self.pipeline(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                height=size[1],
                width=size[0],
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                num_inference_steps=self.steps,
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
        
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 