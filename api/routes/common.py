#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通用API路由模块，包含健康检查和模型信息等
"""
import logging
from fastapi import APIRouter
from api.models import ModelInfoResponse

logger = logging.getLogger(__name__)

# 创建路由器，不指定前缀以便在主路由器中使用统一前缀
router = APIRouter()

@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok"}

@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info():
    """获取模型信息"""
    return {
        "models": [
            {
                "name": "IP-Adapter Inpainting",
                "type": "inpainting",
                "version": "1.0",
                "description": "将人脸合成到无脸模板中"
            },
            {
                "name": "IP-Adapter ControlNet Depth",
                "type": "controlnet",
                "version": "1.0",
                "description": "使用深度图控制生成人脸图像"
            },
            {
                "name": "IP-Adapter Text2Img",
                "type": "text2img",
                "version": "1.0",
                "description": "基于文本提示词和人脸生成图像"
            }
        ]
    } 