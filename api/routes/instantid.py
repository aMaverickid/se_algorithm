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
import json
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from models.instantid import InstantID
from utils.image_utils import image_to_base64, base64_to_image, save_output_image

from api.models import (
    InstantIDRequest, InstantIDStylizeRequest, InstantIDResponse
)

logger = logging.getLogger(__name__)

# 创建路由器，不指定前缀以便在主路由器中使用统一前缀
router = APIRouter()

# 全局实例
instantid_model = None

def get_instantid_model() -> InstantID:
    """获取或创建全局InstantID模型实例"""
    global instantid_model
    if instantid_model is None:
        logger.info("初始化InstantID模型...")
        instantid_model = InstantID()
    return instantid_model

@router.post("/instantid/generate", response_model=InstantIDResponse)
async def generate_image(request: InstantIDRequest):
    """
    输入人脸图像和提示词，生成新的人脸图像
    """
    try:
        logger.info("InstantID 生成请求开始处理")
        
        # 获取模型实例
        model = get_instantid_model()
        
        # 解析人脸图像
        ## face_image = base64_to_image(request.face_image)
        face_image = Image.open("/home/lujingdian/SE_Proj/test/face.png").convert("RGB")
        
        # 生成图像
        output_image = model.generate(
            face_image=face_image,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            controlnet_conditioning_scale=request.controlnet_conditioning_scale,
            ip_adapter_scale=request.ip_adapter_scale,
        )
        
        # 保存图像
        output_image.save("/home/lujingdian/SE_Proj/test/result.png")
        # 转换为Base64
        output_base64 = image_to_base64(output_image)
        
        # 生成参数记录
        parameters = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "controlnet_conditioning_scale": request.controlnet_conditioning_scale,
            "ip_adapter_scale": request.ip_adapter_scale,
        }
        
        return InstantIDResponse(
            success=True,
            image=output_base64,
            parameters=parameters,
            message="图像生成成功"
        )
    
    except Exception as e:
        logger.error(f"InstantID生成失败: {str(e)}")
        logger.error(traceback.format_exc())
        return InstantIDResponse(
            success=False,
            message=f"生成失败: {str(e)}"
        )

@router.get("/instantid/prompt_templates")
async def get_prompt_templates():
    """
    获取InstantID的风格化提示词模板
    """
    with open("/home/lujingdian/SE_Proj/templates/style/sdxl_styles.json", "r") as f:
        return json.load(f)