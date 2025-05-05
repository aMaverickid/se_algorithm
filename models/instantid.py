"""
InstantID模型，基于IP-Adapter技术的人脸保持身份生成模型
"""
import os
import logging
import torch
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Union, Tuple
from PIL import Image
import sys
from pathlib import Path

from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from insightface.app import FaceAnalysis


# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from utils.image_utils import resize_image, image_to_base64, base64_to_image, square_pad
from models.pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps


logger = logging.getLogger(__name__)

class InstantID:
    """
    基于InstantID技术的图像生成模型
    
    特点:
    - 保持人脸身份一致性
    - 支持风格化迁移
    - 结合ControlNet和人脸嵌入向量
    """
    
    def __init__(self):
        """初始化InstantID模型"""
        self.device = config.DEVICE
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model_path = config.INSTANTID_MODEL_PATH
        self.control_net_path = config.INSTANTID_CONTROLNET_MODEL_PATH
        
        # 模型组件
        self.pipeline = None
        self.face_adapter = config.INSTANTID_MODEL_PATH / "ip-adapter.bin"
        self.face_analyzer = None
        
        # 初始化人脸检测和分析模型
        self._init_face_models()
        self._init_pipeline()
    
    def _init_face_models(self):
        """初始化人脸检测和分析模型"""
        logger.info("正在加载人脸检测和分析模型...")
        try:
            # 设置InsightFace模型路径
            insightface_models_path = config.INSTANTID_INSIGHT_PATH
            
            # 初始化FaceAnalysis
            self.face_analyzer = FaceAnalysis(
                name="antelopev2", 
                root=insightface_models_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))
            logger.info("人脸检测和分析模型加载完成")
        except ImportError:
            logger.error("未找到insightface库，请安装: pip install insightface")
            raise
        except Exception as e:
            logger.error(f"加载人脸检测和分析模型失败: {str(e)}")
            raise
    
    def _init_pipeline(self):
        """加载Stable Diffusion模型和InstantID相关组件"""
        logger.info(f"加载InstantID模型和相关组件，设备: {self.device}")
        
        try:
            # 加载ControlNet模型
            controlnet_path = config.INSTANTID_CONTROLNET_MODEL_PATH
            logger.info(f"正在加载ControlNet模型: {controlnet_path}")
            controlnet = ControlNetModel.from_pretrained(
                controlnet_path,
                torch_dtype=self.torch_dtype
            ).to(self.device)
            
            # 加载StableDiffusion模型
            logger.info(f"正在加载StableDiffusion模型: {config.INSTANTID_SD_MODEL_PATH}")
            self.pipeline = StableDiffusionXLInstantIDPipeline.from_pretrained(
                config.INSTANTID_SD_MODEL_PATH,
                controlnet=controlnet,
                torch_dtype=self.torch_dtype,
            ).to(self.device)
            
            # 加载IP-Adapter Face模型
            self.pipeline.load_ip_adapter_instantid(self.face_adapter)
            logger.info("InstantID模型初始化完成")
        
        except ImportError as e:
            logger.error(f"缺少必要的库: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"初始化InstantID模型失败: {str(e)}")
            raise
            
    def generate(self, face_image, prompt, negative_prompt, num_inference_steps=50, guidance_scale=5.0, controlnet_conditioning_scale=0.8, ip_adapter_scale=0.8):
        '''
        输入人脸图像和提示词，生成新的人脸图像
        Args:
            face_image: 人脸图像 PIL.Image.Image
            prompt: 生成提示词
            negative_prompt: 负面提示词
            num_inference_steps: 推理步数
            guidance_scale: 文本提示相关性控制
            controlnet_conditioning_scale: ControlNet条件权重
            ip_adapter_scale: IP-Adapter权重
        Returns:
            image: 生成的人脸图像
        '''
        # prepare face emb
        face_info = self.face_analyzer.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
        face_emb = face_info['embedding']
        face_kps = draw_kps(face_image, face_info['kps'])

        # generate image
        image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ip_adapter_scale=ip_adapter_scale,
        ).images[0]

        return image