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
from processors.mask_generator import MaskGenerator
from processors.face_detector import FaceDetector
from api.models import (
    TemplateGenerationRequest, TemplateGenerationResponse,
    MaskGenerationRequest, MaskGenerationResponse
)

logger = logging.getLogger(__name__)

# 创建路由器，不指定前缀以便在主路由器中使用统一前缀
router = APIRouter()

# 全局掩码生成器实例（懒加载）
mask_generator = None
# 全局人脸检测器实例（懒加载）
face_detector = None

def get_mask_generator():
    """获取掩码生成器实例（懒加载）"""
    global mask_generator
    if mask_generator is None:
        logger.info("初始化MaskGenerator")
        mask_generator = MaskGenerator()
        logger.info(f"MaskGenerator初始化完成: {mask_generator}")
    return mask_generator

def get_face_detector():
    """获取人脸检测器实例（懒加载）"""
    global face_detector
    if face_detector is None:
        logger.info("初始化FaceDetector")
        face_detector = FaceDetector()
        logger.info(f"FaceDetector初始化完成: {face_detector}")
    return face_detector

@router.post("/face/generate-template", response_model=TemplateGenerationResponse)
async def generate_template(request: TemplateGenerationRequest):
    """生成无脸模板图像API"""
    try:
        logger.info(f"无脸模板生成请求开始处理，方法: {request.method}")
        
        # 检查输入图像是否存在
        if not os.path.exists(request.face_path):
            return TemplateGenerationResponse(
                success=False,
                template_id="",
                template_path="",
                message=f"图像文件不存在: {request.face_path}"
            )
        
        # 打开人脸图像
        face_image = Image.open(request.face_path).convert("RGB")
        
        # 初始化检测器
        detector = get_face_detector()
        mask_gen = get_mask_generator()
        
        # 检测人脸
        face = detector.detect_face(face_image)
        if face is None:
            return TemplateGenerationResponse(
                success=False,
                template_id="",
                template_path="",
                message="未检测到人脸"
            )
        
        # 根据方法生成无脸模板
        template_image = None
        
        if request.method == "blur":
            # 模糊处理方法：复制原图并对人脸区域进行高斯模糊
            template_image = face_image.copy()
            # 获取人脸边界框，适当扩大范围
            face_box = detector.get_face_box(face_image, expand_ratio=0.1)
            # 裁剪人脸区域
            face_area = template_image.crop(face_box)
            # 应用高斯模糊，强度由request.strength控制
            blur_radius = int(max(10, 30 * request.strength))
            blurred_face = face_area.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            # 将模糊后的区域粘贴回原图
            template_image.paste(blurred_face, (int(face_box[0]), int(face_box[1])))
            
        elif request.method == "mask":
            # 掩码处理方法：使用掩码将人脸区域替换为纯色背景
            # 生成人脸掩码
            face_mask = mask_gen.generate_face_mask(
                image=face_image,
                padding_ratio=0.1
            )
            # 创建新图像，使用原图作为基础
            template_image = face_image.copy()
            # 创建填充颜色（使用肤色的平均值）
            face_box = detector.get_face_box(face_image)
            face_area = np.array(face_image.crop(face_box))
            avg_color = face_area.mean(axis=(0, 1)).astype(np.uint8)
            # 为平均颜色添加轻微变化，使其更自然
            color_variation = np.random.randint(-20, 20, size=3)
            fill_color = tuple(np.clip(avg_color + color_variation, 0, 255))
            # 绘制一个覆盖人脸的区域
            overlay = Image.new("RGB", template_image.size, fill_color)
            # 使用掩码将填充色与原图合成
            # 调整强度（1.0为完全替换，0.0为不替换）
            alpha = int(255 * min(1.0, max(0.5, request.strength)))
            # 创建透明度掩码
            alpha_mask = Image.new("L", face_mask.size, 0)
            alpha_mask.paste(alpha, mask=face_mask)
            # 合成图像
            template_image = Image.composite(overlay, template_image, alpha_mask)
        
        else:  # "auto" 或其他值
            # 自动选择最佳方法（根据图像特性）
            # 检测人脸清晰度和大小
            face_box = detector.get_face_box(face_image)
            face_area = np.array(face_image.crop(face_box))
            face_gray = cv2.cvtColor(face_area, cv2.COLOR_RGB2GRAY)
            # 使用拉普拉斯算子评估清晰度
            clarity = cv2.Laplacian(face_gray, cv2.CV_64F).var()
            
            # 根据人脸清晰度和大小决定使用哪种方法
            if clarity > 100 and (face_box[2]-face_box[0]) > face_image.width * 0.15:
                # 高清晰度大脸，使用模糊方法
                logger.info(f"自动选择blur方法，清晰度: {clarity:.2f}")
                # 创建相同的请求但使用blur方法
                blur_request = TemplateGenerationRequest(
                    face_path=request.face_path,
                    method="blur",
                    strength=request.strength
                )
                # 递归调用自身但使用blur方法
                blur_response = await generate_template(blur_request)
                if blur_response.success:
                    return blur_response
            
            # 默认使用掩码方法
            logger.info(f"自动选择mask方法，清晰度: {clarity:.2f}")
            mask_request = TemplateGenerationRequest(
                face_path=request.face_path,
                method="mask",
                strength=request.strength
            )
            # 递归调用自身但使用mask方法
            mask_response = await generate_template(mask_request)
            return mask_response
        
        # 保存生成的模板
        template_id = f"template_{uuid.uuid4().hex}"
        template_path = os.path.join(config.TEMPLATE_UPLOAD_DIR, f"{template_id}.png")
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        template_image.save(template_path)
        
        # 返回响应
        return TemplateGenerationResponse(
            success=True,
            template_id=template_id,
            template_path=template_path,
            message="无脸模板生成成功"
        )
        
    except Exception as e:
        logger.error(f"无脸模板生成失败: {str(e)}")
        logger.error(traceback.format_exc())
        return TemplateGenerationResponse(
            success=False,
            template_id="",
            template_path="",
            message=f"无脸模板生成失败: {str(e)}"
        )

@router.post("/face/generate-mask", response_model=MaskGenerationResponse)
async def generate_mask(request: MaskGenerationRequest):
    """生成掩码图像API"""
    try:
        logger.info(f"掩码生成请求开始处理，类型: {request.mask_type}")
        
        # 获取掩码生成器
        mask_gen = get_mask_generator()
        
        # 使用本地文件路径
        if not os.path.exists(request.image_path):
            return MaskGenerationResponse(
                success=False,
                message=f"图像文件不存在: {request.image_path}"
            )
        image = request.image_path
        
        # 根据掩码类型生成掩码
        if request.mask_type == "face":
            # 生成人脸区域掩码
            mask = mask_gen.generate_face_mask(
                image=image,
                padding_ratio=request.padding_ratio
            )
        elif request.mask_type == "template":
            # 生成模板区域掩码
            mask = mask_gen.generate_template_mask(
                image=image
            )
        else:
            return MaskGenerationResponse(
                success=False,
                message=f"不支持的掩码类型: {request.mask_type}"
            )
        
        # 保存掩码
        mask_id = f"mask_{uuid.uuid4().hex}"
        mask_path = mask_gen.save_mask(mask, filename=mask_id)
        
        # 返回响应
        return MaskGenerationResponse(
            success=True,
            mask_path=mask_path,
            mask_id=mask_id,
            message="掩码生成成功"
        )
        
    except Exception as e:
        logger.error(f"掩码生成失败: {str(e)}")
        logger.error(traceback.format_exc())
        return MaskGenerationResponse(
            success=False,
            message=f"掩码生成失败: {str(e)}"
        )