#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
人脸处理相关API路由模块
"""
import os
import uuid
import logging
import traceback
import cv2
import numpy as np
from PIL import Image, ImageFilter
from fastapi import APIRouter, HTTPException

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from utils.image_utils import image_to_base64, base64_to_image, save_output_image
from processors.face_detector import FaceDetector

from api.models import (
    MediapipeMaskRequest, MediapipeMaskResponse,
    MediapipeTemplateRequest, MediapipeTemplateResponse
)

logger = logging.getLogger(__name__)

# 创建路由器，不指定前缀以便在主路由器中使用统一前缀
router = APIRouter()

# 全部变量实例
face_detector = None

def load_face_detector():
    """获取人脸检测器实例"""
    global face_detector
    if face_detector is None:
        logger.info(f"初始化FaceDetector")
        face_detector = FaceDetector()
        logger.info(f"FaceDetector初始化完成: {face_detector}")
    return face_detector

@router.post("/face/mediapipe_mask", response_model=MediapipeMaskResponse)
async def mediapipe_mask(request: MediapipeMaskRequest):
    """使用Mediapipe生成掩码API"""
    try:
        logger.info(f"Mediapipe 掩码生成请求开始处理")

        # face = base64_to_image(request.face)
        face = Image.open("/home/lujingdian/SE_Proj/test/face.png")
        face_detector = load_face_detector()
        
        # 使用面部网格检测
        # 生成掩码，使用更高级的掩码生成选项
        mask = face_detector.generate_mask(
            face,
            blur_radius=request.blur_radius,
            remove_holes=request.remove_holes,
            detailed_edges=request.detailed_edges
        )

        mask.save('/home/lujingdian/SE_Proj/test/mask.png')
        mask_base64 = image_to_base64(mask)
        return MediapipeMaskResponse(
            success=True,
            mask=mask_base64,
            message="掩码生成成功"
        )

    except Exception as e:
        logger.error(f"掩码生成失败: {str(e)}")
        logger.error(traceback.format_exc())
        return MediapipeMaskResponse(
            success=False,
            message=f"掩码生成失败: {str(e)}"
        )

@router.post("/face/mediapipe_template", response_model=MediapipeTemplateResponse)
async def mediapipe_template(request: MediapipeTemplateRequest):
    """使用Mediapipe生成无脸模板API"""
    try:
        logger.info(f"Mediapipe 无脸模板生成请求开始处理")

        # face = base64_to_image(request.face)
        face = Image.open("/home/lujingdian/SE_Proj/test/face.png")

        face_detector = load_face_detector()
        template = face_detector.generate_template(face, blur_radius=request.blur_radius, remove_holes=request.remove_holes, detailed_edges=request.detailed_edges)
        template.save('/home/lujingdian/SE_Proj/test/template.png')
        template_base64 = image_to_base64(template)

        return MediapipeTemplateResponse(
            success=True,
            template=template_base64,
            message="无脸模板生成成功"
        )

    except Exception as e:
        logger.error(f"无脸模板生成失败: {str(e)}")
        logger.error(traceback.format_exc())
        return MediapipeTemplateResponse(
            success=False,
            message=f"无脸模板生成失败: {str(e)}"
        )