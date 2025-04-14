#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text2Img模型API路由模块
"""
import logging
import traceback
from fastapi import APIRouter, HTTPException, BackgroundTasks

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from models import IPAdapterText2Img
from utils.image_utils import image_to_base64, base64_to_image
from api.models import Text2ImgRequest, Text2ImgResponse

logger = logging.getLogger(__name__)

# 创建路由器，不指定前缀以便在主路由器中使用统一前缀
router = APIRouter()

# 全局模型实例（懒加载）
text2img_model = None

def get_text2img_model():
    """获取Text2Img模型实例（懒加载）"""
    global text2img_model
    if text2img_model is None:
        logger.info("初始化Text2Img模型")
        text2img_model = IPAdapterText2Img(
            device=config.DEVICE,
            sd_model_name=config.STABLE_DIFFUSION_MODEL_PATH,
            ip_adapter_path=config.IP_ADAPTER_MODEL_PATH,
            ip_model_type="plus",
            scale=0.6,
        )
        logger.info(f"Text2Img模型初始化完成: {text2img_model}")
    return text2img_model

@router.post("/text2img", response_model=Text2ImgResponse)
async def create_text2img(request: Text2ImgRequest, background_tasks: BackgroundTasks):
    """Text2Img API"""
    try:
        # 解码人脸图像
        face_image = base64_to_image(request.face_image)
        
        # 获取或初始化模型
        model = get_text2img_model()
        
        # 生成图像
        output_images = model.generate(
            face_image=face_image,
            prompt=request.prompt,
            size=(request.width, request.height),
            guidance_scale=request.guidance_scale,
            num_images=request.num_images,
            seed=request.seed,
            negative_prompt=request.negative_prompt,
        )
        
        # 将生成的图像转换为Base64字符串
        base64_images = [image_to_base64(img) for img in output_images]
        
        # 返回响应
        return {
            "images": base64_images,
            "parameters": {
                "prompt": request.prompt,
                "width": request.width,
                "height": request.height,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed,
                "negative_prompt": request.negative_prompt,
            }
        }
    
    except Exception as e:
        logger.error(f"Text2Img错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) 