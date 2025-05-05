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

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from utils.image_utils import resize_image, image_to_base64, base64_to_image, square_pad

logger = logging.getLogger(__name__)

class IPAdapterInstantID:
    """
    基于InstantID技术的图像生成模型
    
    特点:
    - 保持人脸身份一致性
    - 支持风格化迁移
    - 结合ControlNet和人脸嵌入向量
    """
    
    def __init__(self, 
                device=config.DEVICE, 
                model_path=None,
                control_net_path=None,
                face_adapter_path=None):
        """初始化InstantID模型"""
        self.device = device
        self.torch_dtype = torch.float16 if device == "cuda" else torch.float32
        self.model_path = model_path or os.path.join(config.MODEL_CACHE_DIR, "InstantID")
        self.control_net_path = control_net_path or os.path.join(self.model_path, "ControlNetModel")
        self.face_adapter_path = face_adapter_path or os.path.join(self.model_path, "ip-adapter-plus-face_sd15.safetensors")
        self.ip_adapter_path = config.IP_ADAPTER_MODEL_PATH
        
        # 模型组件
        self.pipeline = None
        self.ip_adapter = None
        self.face_analyzer = None
        
        # 初始化人脸检测和分析模型
        self._init_face_models()
        
        # 标记模型是否已初始化
        self.model_initialized = False
    
    def _init_face_models(self):
        """初始化人脸检测和分析模型"""
        logger.info("正在加载人脸检测和分析模型...")
        try:
            # 使用InsightFace进行人脸检测和分析
            import insightface
            from insightface.app import FaceAnalysis
            
            # 确保模型目录存在
            os.makedirs(os.path.join(config.MODEL_CACHE_DIR, "insightface"), exist_ok=True)
            
            # 设置InsightFace模型路径
            insightface_models_path = os.path.join(config.MODEL_CACHE_DIR, "insightface")
            
            # 初始化FaceAnalysis
            self.face_analyzer = FaceAnalysis(
                name="buffalo_l", 
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
    
    def get_face_embeds(self, face_image: Union[str, Image.Image], enhance_face: bool = True) -> Dict[str, Any]:
        """
        提取人脸嵌入向量
        
        Args:
            face_image: 人脸图像路径或PIL图像
            enhance_face: 是否增强人脸（改善光照等）
            
        Returns:
            包含人脸嵌入向量和关键点的字典
        """
        # 确保face_image是PIL图像
        if isinstance(face_image, str):
            face_image = Image.open(face_image).convert("RGB")
        
        # 转换为numpy数组(RGB)
        face_image_np = np.array(face_image)
        
        # 检测和分析人脸
        faces = self.face_analyzer.get(face_image_np)
        if len(faces) == 0:
            logger.error("未检测到人脸")
            raise ValueError("未检测到人脸")
        
        # 使用置信度最高的面部
        face = sorted(faces, key=lambda x: x.det_score, reverse=True)[0]
        
        # 获取面部关键点
        face_kps = face.kps
        
        # 提取面部特征向量
        face_embed = face.normed_embedding
        
        # 准备控制图像
        face_image_tensor = self._prepare_control_image(face_image_np, face_kps, enhance_face)
        
        # 返回结果
        return {
            "face_embed": face_embed,
            "face_kps": face_kps,
            "control_image": face_image_tensor
        }
        
    def _prepare_control_image(self, face_image_np: np.ndarray, face_kps: np.ndarray, enhance_face: bool) -> torch.Tensor:
        """
        准备用于ControlNet的控制图像
        
        Args:
            face_image_np: 人脸图像的numpy数组
            face_kps: 人脸关键点
            enhance_face: 是否增强人脸
            
        Returns:
            控制图像张量
        """
        # 使用关键点生成控制图像
        h, w = face_image_np.shape[:2]
        
        # 创建一个白色背景图像
        control_image = np.zeros((h, w, 3), dtype=np.uint8) + 255
        
        # 绘制关键点连线
        for i in range(5):
            if i < 4:
                cv2.line(control_image, (int(face_kps[i][0]), int(face_kps[i][1])), 
                         (int(face_kps[i+1][0]), int(face_kps[i+1][1])), (0, 0, 255), 2)
            if i == 4:
                cv2.line(control_image, (int(face_kps[i][0]), int(face_kps[i][1])), 
                         (int(face_kps[0][0]), int(face_kps[0][1])), (0, 0, 255), 2)
        
        # 如果需要增强人脸，应用一些增强技术
        if enhance_face:
            # 简单的增强：调整亮度和对比度
            alpha = 1.2  # 对比度增强因子
            beta = 10    # 亮度增强因子
            
            control_image = cv2.convertScaleAbs(control_image, alpha=alpha, beta=beta)
            
            # 可选：应用锐化
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            control_image = cv2.filter2D(control_image, -1, kernel)
        
        # 调整控制图像大小
        control_image = cv2.resize(control_image, (512, 512))
        
        # 转换为RGB格式的PIL图像
        control_image_pil = Image.fromarray(cv2.cvtColor(control_image, cv2.COLOR_BGR2RGB))
        
        # 转换为张量
        control_image_tensor = self._pil_to_tensor(control_image_pil)
        
        return control_image_tensor
    
    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """
        将PIL图像转换为模型输入张量
        
        Args:
            image: PIL图像
            
        Returns:
            图像张量
        """
        # 调整图像大小
        image = resize_image(image, (512, 512))
        
        # 转换为numpy数组(RGB, 0-255)
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # 调整维度顺序：HWC -> CHW
        image_np = image_np.transpose(2, 0, 1)
        
        # 转换为张量
        image_tensor = torch.from_numpy(image_np).unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def _load_pipeline(self):
        """加载Stable Diffusion模型和InstantID相关组件"""
        logger.info(f"加载InstantID模型和相关组件，设备: {self.device}")
        
        try:
            from diffusers import (
                StableDiffusionPipeline,
                StableDiffusionControlNetPipeline,
                ControlNetModel,
                DDIMScheduler
            )
            
            # 加载ControlNet模型
            controlnet_path = os.path.join(self.model_path, "ControlNetModel")
            controlnet_path = config.CONTROLNET_DEPTH_MODEL_PATH
            logger.info(f"正在加载ControlNet模型: {controlnet_path}")
            controlnet = ControlNetModel.from_pretrained(
                controlnet_path,
                torch_dtype=self.torch_dtype
            ).to(self.device)
            
            # 加载StableDiffusion模型
            sd_path = "runwayml/stable-diffusion-v1-5"
            logger.info(f"正在加载StableDiffusion模型: {sd_path}")
            self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                sd_path,
                controlnet=controlnet,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
                feature_extractor=None
            ).to(self.device)
            
            # 加载IP-Adapter Face模型
            self.load_ip_adapter()
            
            # 使用DDIM调度器
            self.pipeline.scheduler = DDIMScheduler.from_config(
                self.pipeline.scheduler.config,
                timestep_spacing="trailing",
                rescale_betas_zero_snr=True
            )
            
            # 启用内存优化
            if self.device == "cuda":
                self.pipeline.enable_xformers_memory_efficient_attention()
            
            # 标记模型已初始化
            self.model_initialized = True
            
            logger.info("InstantID模型初始化完成")
        
        except ImportError as e:
            logger.error(f"缺少必要的库: {str(e)}")
            logger.error("请安装必要的依赖: pip install diffusers transformers insightface")
            raise
        except Exception as e:
            logger.error(f"初始化InstantID模型失败: {str(e)}")
            raise
    
    def load_ip_adapter(self):
        """加载IP-Adapter模型"""
        try:
            from diffusers.utils import load_image
            from ip_adapter import IPAdapterPlus
            
            # 设置IP-Adapter模型路径
            self.ip_model_type = "plus_face"  # 使用专为人脸优化的IP-Adapter
            model_path = os.path.join(self.ip_adapter_path, "models", "ip-adapter-plus-face_sd15.safetensors")
            
            logger.info(f"加载IP-Adapter模型: {model_path}")
            
            # 初始化IP-Adapter
            self.ip_adapter = IPAdapterPlus(
                self.pipeline,
                model_path,
                device=self.device
            )
            
            logger.info("IP-Adapter模型加载完成")
        
        except ImportError:
            logger.error("未找到IP-Adapter库，请安装: pip install ip-adapter")
            raise
        except Exception as e:
            logger.error(f"加载IP-Adapter模型失败: {str(e)}")
            raise
            
    def _ensure_processor(self):
        """确保处理器已初始化"""
        # 检查FaceAnalysis是否已初始化
        if self.face_analyzer is None:
            self._init_face_models()
    
    @property
    def model_initialized(self):
        """是否已初始化模型"""
        return hasattr(self, '_model_initialized') and self._model_initialized
    
    @model_initialized.setter
    def model_initialized(self, value):
        """设置模型初始化状态"""
        self._model_initialized = value
    
    def generate(self, 
                face_image: Union[str, Image.Image],
                prompt: str = "",
                negative_prompt: str = "lowres, bad anatomy, bad hands, cropped, worst quality",
                num_samples: int = 1, 
                num_inference_steps: int = 50, 
                guidance_scale: float = 7.5,
                controlnet_conditioning_scale: float = 0.8,
                ip_adapter_scale: float = 0.8,
                style_strength: float = 20.0,
                enhance_face: bool = True,
                seed: Optional[int] = None) -> List[Image.Image]:
        """
        生成保持身份的人脸图像
        
        Args:
            face_image: 人脸图像（路径或PIL图像）
            prompt: 生成提示词
            negative_prompt: 负面提示词
            num_samples: 生成的样本数量
            num_inference_steps: 推理步数
            guidance_scale: 文本指导比例
            controlnet_conditioning_scale: ControlNet条件缩放
            ip_adapter_scale: IP-Adapter缩放比例
            style_strength: 风格强度
            enhance_face: 是否增强人脸
            seed: 随机种子，None为随机
            
        Returns:
            生成的图像列表
        """
        # 确保处理器和模型已初始化
        self._ensure_processor()
        if not self.model_initialized:
            self._load_pipeline()
        
        # 确保face_image是PIL图像
        if isinstance(face_image, str):
            face_image = Image.open(face_image).convert("RGB")
        
        # 提取人脸嵌入
        face_info = self.get_face_embeds(face_image, enhance_face)
        face_embed = face_info["face_embed"]
        control_image = face_info["control_image"]
        
        # 设置随机种子
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        logger.info(f"使用随机种子: {seed}")
        torch.manual_seed(seed)
        
        # 调整生成参数
        final_prompt = prompt if prompt else "photo of a person"
        try:
            # 执行生成
            logger.info(f"开始生成图像，提示词: {final_prompt}")
            with torch.inference_mode():
                images = self.ip_adapter.generate(
                    prompt=final_prompt,
                    negative_prompt=negative_prompt,
                    image_embeds=torch.from_numpy(face_embed).unsqueeze(0).to(self.device, dtype=self.torch_dtype),
                    controlnet_conditioning_scale=float(controlnet_conditioning_scale),
                    control_guidance_start=0.0,
                    control_guidance_end=1.0,
                    guidance_scale=float(guidance_scale),
                    ip_adapter_scale=float(ip_adapter_scale),
                    num_samples=num_samples,
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator(device=self.device).manual_seed(seed),
                    control_image=control_image
                )
            
            logger.info(f"成功生成 {len(images)} 张图像")
            return images
        
        except Exception as e:
            logger.error(f"生成图像失败: {str(e)}")
            raise
    
    def stylize(self, 
               face_image: Union[str, Image.Image],
               style_image: Optional[Union[str, Image.Image]] = None,
               style_preset: Optional[str] = None,
               prompt: str = "",
               negative_prompt: str = "lowres, bad anatomy, bad hands, cropped, worst quality",
               num_samples: int = 1, 
               num_inference_steps: int = 50, 
               guidance_scale: float = 7.5,
               controlnet_conditioning_scale: float = 0.8,
               ip_adapter_scale: float = 0.8,
               style_strength: float = 20.0,
               enhance_face: bool = True,
               seed: Optional[int] = None) -> List[Image.Image]:
        """
        基于特定风格生成保持身份的人脸图像
        
        Args:
            face_image: 人脸图像（路径或PIL图像）
            style_image: 风格参考图像（可选）
            style_preset: 预设风格名称（可选）
            prompt: 生成提示词
            negative_prompt: 负面提示词
            num_samples: 生成的样本数量
            num_inference_steps: 推理步数
            guidance_scale: 文本指导比例
            controlnet_conditioning_scale: ControlNet条件缩放
            ip_adapter_scale: IP-Adapter缩放比例
            style_strength: 风格强度
            enhance_face: 是否增强人脸
            seed: 随机种子，None为随机
            
        Returns:
            生成的图像列表
        """
        # 预设风格提示词
        style_presets = {
            "anime": "anime style, anime character, detailed anime face",
            "cartoon": "cartoon style, cartoon character, detailed cartoon face",
            "digital_art": "digital art, highly detailed digital painting, vibrant colors",
            "fantasy": "fantasy character, mystical atmosphere, ethereal lighting",
            "oil_painting": "oil painting, classical portrait, detailed brushstrokes, realistic",
            "watercolor": "watercolor painting, soft colors, artistic, flowing pigments",
            "pop_art": "pop art style, bold colors, comic-like, artistic",
            "cyberpunk": "cyberpunk style, neon lights, futuristic, sci-fi elements",
            "vintage": "vintage photo, retro style, old film grain, nostalgic",
            "comic": "comic book style, bold outlines, flat colors"
        }
        
        # 如果提供了风格预设，使用预设提示词
        if style_preset and style_preset in style_presets:
            if prompt:
                prompt = f"{prompt}, {style_presets[style_preset]}"
            else:
                prompt = style_presets[style_preset]
            
            logger.info(f"使用风格预设: {style_preset}")
            logger.info(f"最终提示词: {prompt}")
        
        # 生成图像
        return self.generate(
            face_image=face_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_samples=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ip_adapter_scale=ip_adapter_scale,
            enhance_face=enhance_face,
            seed=seed
        )
    
    def __del__(self):
        """清理资源"""
        # 清除模型
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            del self.pipeline
        if hasattr(self, 'ip_adapter') and self.ip_adapter is not None:
            del self.ip_adapter
        if hasattr(self, 'face_analyzer') and self.face_analyzer is not None:
            del self.face_analyzer
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 