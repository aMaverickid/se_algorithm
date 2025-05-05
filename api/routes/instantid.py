#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
InstantID人脸生成API路由模块
"""
import os
import logging
import traceback
import uuid
from typing import Dict, List, Any, Optional
from PIL import Image
from fastapi import APIRouter, HTTPException

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from models.instantid import IPAdapterInstantID
from utils.image_utils import image_to_base64, base64_to_image, save_output_image

from api.models import (
    InstantIDRequest, InstantIDStylizeRequest, InstantIDResponse
)

logger = logging.getLogger(__name__)

# 创建路由器，不指定前缀以便在主路由器中使用统一前缀
router = APIRouter()

# 全局实例
instantid_model = None

def get_instantid_model():
    """获取或创建全局InstantID模型实例"""
    global instantid_model
    if instantid_model is None:
        logger.info("初始化InstantID模型...")
        instantid_model = IPAdapterInstantID(device=config.DEVICE)
    return instantid_model

@router.post("/instantid/generate", response_model=InstantIDResponse)
async def generate_image(request: InstantIDRequest):
    """
    生成保持身份一致性的人脸图像API
    
    使用InstantID技术根据输入的人脸图像和提示词生成新的人脸图像，
    保持身份一致性。
    """
    try:
        logger.info("InstantID 生成请求开始处理")
        
        # 获取模型实例
        model = get_instantid_model()
        
        # 解析人脸图像
        face_image = base64_to_image(request.face_image)
        
        # 生成图像
        output_images = model.generate(
            face_image=face_image,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_samples=request.num_samples,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            controlnet_conditioning_scale=request.controlnet_conditioning_scale,
            ip_adapter_scale=request.ip_adapter_scale,
            enhance_face=request.enhance_face,
            seed=request.seed
        )
        
        # 转换为Base64
        output_base64_list = [image_to_base64(img) for img in output_images]
        
        # 生成参数记录
        parameters = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "num_samples": request.num_samples,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "controlnet_conditioning_scale": request.controlnet_conditioning_scale,
            "ip_adapter_scale": request.ip_adapter_scale,
            "enhance_face": request.enhance_face,
            "seed": request.seed
        }
        
        return InstantIDResponse(
            success=True,
            images=output_base64_list,
            parameters=parameters,
            message="图像生成成功"
        )
    
    except Exception as e:
        logger.error(f"InstantID生成失败: {str(e)}")
        logger.error(traceback.format_exc())
        return InstantIDResponse(
            success=False,
            images=[],
            parameters={},
            message=f"生成失败: {str(e)}"
        )

@router.post("/instantid/stylize", response_model=InstantIDResponse)
async def stylize_image(request: InstantIDStylizeRequest):
    """
    基于特定风格生成保持身份一致性的人脸图像API
    
    使用InstantID技术结合预设风格，根据输入的人脸图像生成具有特定风格的新图像，
    同时保持身份一致性。
    """
    try:
        logger.info(f"InstantID 风格化生成请求开始处理，风格预设: {request.style_preset}")
        
        # 获取模型实例
        model = get_instantid_model()
        
        # 解析人脸图像
        face_image = base64_to_image(request.face_image)
        
        # 生成风格化图像
        output_images = model.stylize(
            face_image=face_image,
            style_preset=request.style_preset,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_samples=request.num_samples,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            controlnet_conditioning_scale=request.controlnet_conditioning_scale,
            ip_adapter_scale=request.ip_adapter_scale,
            style_strength=request.style_strength,
            enhance_face=request.enhance_face,
            seed=request.seed
        )
        
        # 转换为Base64
        output_base64_list = [image_to_base64(img) for img in output_images]
        
        # 生成参数记录
        parameters = {
            "style_preset": request.style_preset,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "num_samples": request.num_samples,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "controlnet_conditioning_scale": request.controlnet_conditioning_scale,
            "ip_adapter_scale": request.ip_adapter_scale,
            "style_strength": request.style_strength,
            "enhance_face": request.enhance_face,
            "seed": request.seed
        }
        
        return InstantIDResponse(
            success=True,
            images=output_base64_list,
            parameters=parameters,
            message="风格化图像生成成功"
        )
    
    except Exception as e:
        logger.error(f"InstantID风格化生成失败: {str(e)}")
        logger.error(traceback.format_exc())
        return InstantIDResponse(
            success=False,
            images=[],
            parameters={},
            message=f"风格化生成失败: {str(e)}"
        ) 