#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ControlNet模型API路由模块
"""
import os
import logging
import traceback
from fastapi import APIRouter, HTTPException, BackgroundTasks

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from models import IPAdapterControlNet
from utils.image_utils import image_to_base64, base64_to_image
from api.models import ControlNetRequest, ControlNetResponse

logger = logging.getLogger(__name__)

# 创建路由器，不指定前缀以便在主路由器中使用统一前缀
router = APIRouter()

# 全局模型实例（懒加载）
controlnet_model = None

def get_controlnet_model():
    """获取ControlNet模型实例（懒加载）"""
    global controlnet_model
    if controlnet_model is None:
        logger.info("初始化ControlNet模型")
        controlnet_model = IPAdapterControlNet(
            device=config.DEVICE,
            sd_model_name=config.STABLE_DIFFUSION_MODEL_PATH,
            controlnet_model_name=config.CONTROLNET_DEPTH_MODEL_PATH,
            ip_adapter_path=config.IP_ADAPTER_MODEL_PATH,
            ip_model_type="plus",
            scale=0.7,
        )
        logger.info(f"ControlNet模型初始化完成: {controlnet_model}")
    return controlnet_model

@router.post("/controlnet", response_model=ControlNetResponse)
async def create_controlnet(request: ControlNetRequest, background_tasks: BackgroundTasks):
    """ControlNet API"""
    try:
        # 解码人脸图像
        face_image = base64_to_image(request.face_image)
        
        # 获取深度图像
        depth_image = None
        if request.depth_template_id:
            # 使用指定的模板ID
            depth_path = os.path.join(config.DEPTH_TEMPLATE_DIR, request.depth_template_id)
            if os.path.exists(depth_path):
                depth_image = Image.open(depth_path).convert("RGB")
            else:
                raise HTTPException(status_code=404, detail=f"找不到深度模板: {request.depth_template_id}")
        elif request.depth_image:
            # 使用提供的深度图像
            depth_image = base64_to_image(request.depth_image)
        
        # 获取或初始化模型
        model = get_controlnet_model()
        
        # 如果没有提供深度图，从人脸自动生成
        if depth_image is None:
            depth_image = model.depth_estimator.estimate_depth(face_image)
            depth_base64 = image_to_base64(depth_image)
        else:
            depth_base64 = image_to_base64(depth_image) if depth_image else None
        
        # 生成图像
        output_images = model.generate(
            face_image=face_image,
            depth_image=depth_image,
            controlnet_conditioning_scale=request.controlnet_conditioning_scale,
            guidance_scale=request.guidance_scale,
            num_images=request.num_images,
            seed=request.seed,
            positive_prompt=request.positive_prompt,
            negative_prompt=request.negative_prompt,
        )
        
        # 将生成的图像转换为Base64字符串
        base64_images = [image_to_base64(img) for img in output_images]
        
        # 返回响应
        return {
            "images": base64_images,
            "depth_image": depth_base64,
            "parameters": {
                "controlnet_conditioning_scale": request.controlnet_conditioning_scale,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed,
                "positive_prompt": request.positive_prompt,
                "negative_prompt": request.negative_prompt,
            }
        }
    
    except Exception as e:
        logger.error(f"ControlNet错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) 