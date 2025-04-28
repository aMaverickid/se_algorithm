#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inpainting模型API路由模块
"""
import os
import logging
import traceback
from fastapi import APIRouter, HTTPException, BackgroundTasks
from PIL import Image
from utils.image_utils import resize_image
import uuid

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from models import IPAdapterInpainting
from utils.image_utils import image_to_base64, base64_to_image, get_sample_templates
from api.models import InpaintingRequest, InpaintingResponse

logger = logging.getLogger(__name__)

# 创建路由器，不指定前缀以便在主路由器中使用统一前缀
router = APIRouter()

# 全局模型实例（懒加载）
inpainting_model = None

def load_inpainting_model():
    """获取Inpainting模型实例（懒加载）"""
    global inpainting_model
    if inpainting_model is None:
        logger.info("初始化Inpainting模型")
        inpainting_model = IPAdapterInpainting(
            device=config.DEVICE,
            sd_model_path=config.STABLE_DIFFUSION_MODEL_PATH,
            ip_adapter_path=config.IP_ADAPTER_MODEL_PATH,
            ip_model_type="plus",
        )
        logger.info(f"Inpainting模型初始化完成: {inpainting_model}")
    return inpainting_model

@router.post("/inpainting", response_model=InpaintingResponse)
async def create_inpainting(request: InpaintingRequest, background_tasks: BackgroundTasks):
    """Inpainting API"""
    try:
        logger.info(f"Inpainting请求开始处理: prompt={request.prompt}, strength={request.strength}")
        
        # 解码人脸图像 - 修正人脸图像读取，使用传入的base64编码
        ## face = base64_to_image(request.face)
        # 保存上传的人脸用于调试
        face = Image.open("/home/lujingdian/SE_Proj/test/face.png")
        logger.info(f"解码人脸图像: {face.size}")

        # 获取模板图像
        if request.template_id:
            template = get_sample_templates(template_type="inpainting", number=request.template_id)
            logger.info(f"使用模板ID: {request.template_id}")
        else:
            if request.template:
                template = base64_to_image(request.template)
                logger.info(f"使用自定义模板: {template.size}")
            else:
                return InpaintingResponse(
                    success=False,
                    message="必须提供template_id或template"
                )
                
        # 获取掩码图像
        if request.mask:
            mask = base64_to_image(request.mask)
            logger.info(f"使用自定义掩码: {mask.size}")
        else:
            mask = None
            logger.info("未提供掩码，将自动生成")
        
        # 获取或初始化模型
        model = load_inpainting_model()
        
        # 调整图像大小 - 确保所有图像都是512x512
        face = resize_image(face, (512, 512))
        template = resize_image(template, (512, 512))
        if mask:
            mask = resize_image(mask, (512, 512))
        
        # 设置增强后的参数
        # 如果没有指定ip_adapter_scale，使用更高的默认值1.2
        ip_adapter_scale = request.ip_adapter_scale if request.ip_adapter_scale is not None else 1.2
        # 确保strength参数至少为0.85以允许更强的修改
        strength = max(0.85, request.strength) if request.strength is not None else 0.95
        # 确保推理步数足够高，以生成高质量结果
        num_inference_steps = max(50, request.num_inference_steps) if request.num_inference_steps is not None else 75
        
        logger.info(f"优化参数: ip_adapter_scale={ip_adapter_scale}, strength={strength}, steps={num_inference_steps}")
        
        # 生成图像
        result = model.generate(
            face=face,
            template=template,
            mask=mask,
            ip_adapter_scale=ip_adapter_scale,
            prompt=request.prompt,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=request.guidance_scale,
        )

        # 获取生成的图像并保存
        result_image = result.images[0]
        result_image.save("/home/lujingdian/SE_Proj/test/result.png")
        logger.info(f"生成完成，图像大小: {result_image.size}")

        # 将生成的图像转换为Base64字符串
        result_base64 = image_to_base64(result_image)
        
        # 返回响应
        return {
            "success": True,
            "result": result_base64,
            "message": "Inpainting成功"
        }
    
    except Exception as e:
        logger.error(f"Inpainting错误: {str(e)}")
        logger.error(traceback.format_exc())
        return InpaintingResponse(
            success=False,
            message=f"Inpainting错误: {str(e)}"
        )