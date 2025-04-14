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

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from models import IPAdapterInpainting
from utils.image_utils import image_to_base64, base64_to_image, get_sample_templates
from api.models import InpaintingRequest, InpaintingResponse
from api.models import TemplateGenerationRequest, MaskGenerationRequest

logger = logging.getLogger(__name__)

# 创建路由器，不指定前缀以便在主路由器中使用统一前缀
router = APIRouter()

# 全局模型实例（懒加载）
inpainting_model = None

def get_inpainting_model():
    """获取Inpainting模型实例（懒加载）"""
    global inpainting_model
    if inpainting_model is None:
        logger.info("初始化Inpainting模型")
        inpainting_model = IPAdapterInpainting(
            device=config.DEVICE,
            sd_model_path=config.STABLE_DIFFUSION_MODEL_PATH,
            ip_adapter_path=config.IP_ADAPTER_MODEL_PATH,
            ip_model_type="plus",
            scale=0.7,
        )
        logger.info(f"Inpainting模型初始化完成: {inpainting_model}")
    return inpainting_model

@router.post("/inpainting", response_model=InpaintingResponse)
async def create_inpainting(request: InpaintingRequest, background_tasks: BackgroundTasks):
    """Inpainting API"""
    try:
        # 解码人脸图像
        face_image = None
        if request.face_path:
            face_image = Image.open(request.face_path).convert("RGB")
        elif request.face_image:
            face_image = base64_to_image(request.face_image)
        else:
            raise HTTPException(status_code=400, detail="必须提供face_path或face_image")
            
        # 获取模板图像
        template_image = None
        if request.template_path:
            template_image = Image.open(request.template_path).convert("RGB")
        elif request.template_id:
            # 使用指定的模板ID
            template_path = os.path.join(config.INPAINTING_TEMPLATE_DIR, request.template_id)
            if os.path.exists(template_path):
                template_image = Image.open(template_path).convert("RGB")
            else:
                raise HTTPException(status_code=404, detail=f"找不到模板: {request.template_id}")
        elif request.template_image:
            # 使用提供的模板图像
            template_image = base64_to_image(request.template_image)
        elif request.auto_generate_template:
            # 自动生成无脸模板
            logger.info("自动生成无脸模板")
            
            # 创建模板生成请求
            template_request = TemplateGenerationRequest(
                face_image=request.face_image if request.face_image else image_to_base64(face_image),
                method=request.template_method,
                strength=request.template_strength
            )
            
            # 调用模板生成函数 - 需要从face模块导入
            # 使用推迟导入避免循环导入问题
            from api.routes.face import generate_template
            template_response = await generate_template(template_request)
            
            if template_response.success:
                template_image = base64_to_image(template_response.template_image)
                logger.info(f"成功生成无脸模板，ID: {template_response.template_id}")
            else:
                raise HTTPException(status_code=500, detail="无法生成无脸模板")
        else:
            # 如果没有提供模板，使用第一个默认模板
            templates = get_sample_templates("inpainting", 1)
            if templates:
                template_image = Image.open(templates[0]).convert("RGB")
            else:
                raise HTTPException(status_code=500, detail="找不到默认模板")
        
        # 解码掩码图像（如果提供）
        mask_image = None
        if request.mask_image:
            mask_image = base64_to_image(request.mask_image)
        elif request.auto_generate_mask:
            # 自动生成掩码
            logger.info(f"自动生成掩码，类型: {request.mask_type}")
            
            # 确定要使用的图像
            source_image = request.face_image if request.face_image else image_to_base64(face_image)
            if request.mask_type != "face":
                source_image = image_to_base64(template_image)
            
            # 创建掩码生成请求
            mask_request = MaskGenerationRequest(
                image=source_image,
                mask_type=request.mask_type,
                padding_ratio=request.mask_padding_ratio
            )
            
            # 调用掩码生成函数 - 需要从face模块导入
            # 使用推迟导入避免循环导入问题
            from api.routes.face import generate_mask
            mask_response = await generate_mask(mask_request)
            
            if mask_response.success:
                mask_image = base64_to_image(mask_response.mask_image)
                logger.info(f"成功生成掩码，ID: {mask_response.mask_id}")
            else:
                raise HTTPException(status_code=500, detail="无法生成掩码")
        
        # 获取或初始化模型
        model = get_inpainting_model()
        
        # 生成图像
        output_images = model.generate(
            face_image=face_image,
            template_image=template_image,
            mask=mask_image,
            strength=request.strength,
            guidance_scale=request.guidance_scale,
            num_images=request.num_images,
            seed=request.seed,
            positive_prompt=request.positive_prompt,
            negative_prompt=request.negative_prompt,
        )

        # 保存图像到 results 目录
        results_dir = config.RESULTS_DIR
        for i, img in enumerate(output_images):
            img.save(os.path.join(results_dir, f"result_{i}.png"))
        
        # 将生成的图像转换为Base64字符串
        base64_images = [image_to_base64(img) for img in output_images]
        
        # 返回响应
        return {
            "images": base64_images,
            "parameters": {
                "strength": request.strength,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed,
                "positive_prompt": request.positive_prompt,
                "negative_prompt": request.negative_prompt,
            }
        }
    
    except Exception as e:
        logger.error(f"Inpainting错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) 