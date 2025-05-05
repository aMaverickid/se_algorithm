"""
IP-Adapter Inpainting模型，用于将人脸图像与无脸模板合成
"""
import os
import logging
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
import sys
from pathlib import Path
import uuid

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from utils.image_utils import resize_image, get_sample_templates
from processors.face_detector import FaceDetector

logger = logging.getLogger(__name__)

class IPAdapterInpainting:
    """IP-Adapter Inpainting模型类"""
    
    def __init__(
        self, 
        device=config.DEVICE,
        sd_model_path=config.STABLE_DIFFUSION_MODEL_PATH,
        ip_adapter_path=config.IP_ADAPTER_MODEL_PATH,
        ip_model_type="plus",
        scale=1.2,  # 添加scale参数，inpainting任务通常需要更高的缩放因子，从0.9调整为1.2
    ):
        """
        初始化IP-Adapter Inpainting模型
        
        Args:
            device (str): 设备类型，'cuda'或'cpu'
            sd_model_path (str): Stable Diffusion 模型路径
            ip_adapter_path (str): IP-Adapter 模型路径
            ip_model_type (str): IP-Adapter 模型类型，'base'或'plus'或'plus_face'或'full_face'
            scale (float): IP-Adapter的条件缩放因子
            steps (int): 推理步数
        """
        self.device = device
        self.torch_dtype = torch.float16 if device == "cuda" else torch.float32
        self.pipeline = None  # 扩散模型
        
        self.ip_model_type = ip_model_type
        self.ip_adapter_path = ip_adapter_path
        self.scale = scale  # 设置IP-Adapter缩放因子

        self.ip_adapter = None
        self.face_detector = FaceDetector()
        
        # 加载Stable Diffusion Inpainting模型
        self.load_pipeline(sd_model_path)
        # 加载IP-Adapter权重
        self.load_ip_adapter()
    
    def load_pipeline(self, sd_model_path):
        """加载Stable Diffusion Inpainting模型"""
        logger.info(f"加载Stable Diffusion Inpainting模型: {sd_model_path}")
        
        # 加载基础inpainting模型
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            sd_model_path,
            torch_dtype=self.torch_dtype,
            variant="fp16"
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
    
    def generate(
        self,
        face,
        template,
        mask=None,
        ip_adapter_scale=None,
        prompt="",
        strength=0.85,
        num_inference_steps=50,
        guidance_scale=7.5,
    ):
        """
        使用IP-Adapter Inpainting生成图像
        
        Args:
            face: 人脸图像 PIL格式
            template: 无脸模板图像 PIL格式
            mask: 可选 掩码图像 PIL格式
            prompt: 提示词
            ip_adapter_scale: IP-Adapter缩放因子
            strength: 范围 0--1 值越大原始图像的内容被忽略越多，生成结果更偏向模型自由发挥
            num_inference_steps: 推理步数 步数越大生成结果越精细 但是生成时间越长
            guidance_scale: 分类器自由指导比例 值越大生成结果越遵循提示词 但是生成结果越不自然
            
        Returns:
            result: 风格化后的结果 图像的列表
        """
        # 如果没有提供掩码，则自动生成
        if mask is None:
            logger.info("自动生成模板掩码")
            # 使用更大的模糊半径和边缘模糊处理
            mask = self.face_detector.generate_mask(face, blur_radius=5, remove_holes=True, detailed_edges=True)
            # 调整掩码尺寸
            mask = resize_image(mask, (512, 512))
            # 保存掩码用于调试
            mask.save("/home/lujingdian/SE_Proj/test/mask.png")
            logger.info(f"自动生成模板掩码完成")
        else:
            # 对提供的掩码增加模糊处理，改善边缘过渡
            from PIL import ImageFilter
            mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
            logger.info("对提供的掩码进行了边缘模糊处理")
            
        # 增强提示词，添加面部相关描述，提高人脸生成质量
        if not prompt:
            prompt = "realistic face, high quality, high detail"
        else:
            prompt = f"{prompt}, realistic face, high quality, high detail"
        
        logger.info(f"使用提示词: {prompt}")
        
        # 设置IP-Adapter缩放因子
        if ip_adapter_scale is None:
            # 使用默认的scale值
            self.pipeline.set_ip_adapter_scale(self.scale)
            logger.info(f"使用默认IP-Adapter缩放因子: {self.scale}")
        else:
            # 使用传入的scale值
            self.pipeline.set_ip_adapter_scale(ip_adapter_scale)
            logger.info(f"使用自定义IP-Adapter缩放因子: {ip_adapter_scale}")
            
        # 生成图像
        try:
            result = self.pipeline(
                image=template,
                mask_image=mask,
                ip_adapter_image=face,
                prompt=prompt,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            
            return result
            
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
        
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 