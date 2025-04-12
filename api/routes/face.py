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
from PIL import Image
from fastapi import APIRouter, HTTPException

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from utils.image_utils import image_to_base64, base64_to_image
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

def get_mask_generator():
    """获取掩码生成器实例（懒加载）"""
    global mask_generator
    if mask_generator is None:
        logger.info("初始化MaskGenerator")
        mask_generator = MaskGenerator()
        logger.info(f"MaskGenerator初始化完成: {mask_generator}")
    return mask_generator

@router.post("/face/generate-template", response_model=TemplateGenerationResponse)
async def generate_template(request: TemplateGenerationRequest):
    """生成无脸模板图像API"""
    try:
        # 获取人脸图像
        face_image = None
        if request.face_image:
            # 使用提供的base64图像
            face_image = base64_to_image(request.face_image)
        elif request.face_id:
            # 使用已上传的人脸图像
            face_path = os.path.join(config.FACE_UPLOAD_DIR, request.face_id)
            if os.path.exists(face_path):
                face_image = Image.open(face_path).convert("RGB")
            else:
                raise HTTPException(status_code=404, detail=f"找不到人脸图像: {request.face_id}")
        else:
            raise HTTPException(status_code=400, detail="必须提供face_image或face_id")
        
        # 根据方法生成无脸模板
        template_image = None
        
        if request.method == "blur":
            # 使用模糊方法生成无脸模板
            # 检测人脸位置
            face_detector = FaceDetector()
            faces = face_detector.detect_faces(face_image)
            
            if not faces:
                raise HTTPException(status_code=400, detail="未检测到人脸，无法生成模板")
            
            # 创建图像副本
            template_image = face_image.copy()
            
            # 将图像转换为OpenCV格式
            img_cv = cv2.cvtColor(np.array(template_image), cv2.COLOR_RGB2BGR)
            
            for face in faces:
                x, y, w, h = face[:4]
                # 扩大一点人脸区域
                padding = int(min(w, h) * 0.1)
                face_region = img_cv[
                    max(0, y - padding):min(img_cv.shape[0], y + h + padding),
                    max(0, x - padding):min(img_cv.shape[1], x + w + padding)
                ]
                
                # 根据强度参数调整模糊程度
                blur_size = int(max(5, min(w, h) * request.strength * 0.2))
                # 确保blur_size是奇数
                if blur_size % 2 == 0:
                    blur_size += 1
                
                # 应用高斯模糊
                blurred_face = cv2.GaussianBlur(face_region, (blur_size, blur_size), 0)
                
                # 将模糊的人脸放回原图
                img_cv[
                    max(0, y - padding):min(img_cv.shape[0], y + h + padding),
                    max(0, x - padding):min(img_cv.shape[1], x + w + padding)
                ] = blurred_face
            
            # 转换回PIL格式
            template_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            
        elif request.method == "mask":
            # 使用掩码方法生成无脸模板
            # 获取掩码生成器
            mask_generator = get_mask_generator()
            
            # 生成人脸掩码
            face_mask = mask_generator.generate_face_mask(face_image)
            
            # 创建图像副本
            template_image = face_image.copy()
            
            # 将图像转换为OpenCV格式
            img_cv = cv2.cvtColor(np.array(template_image), cv2.COLOR_RGB2BGR)
            mask_cv = np.array(face_mask)
            
            # 确定填充颜色（使用图像的平均颜色）
            avg_color = cv2.mean(img_cv, mask=mask_cv)[:3]
            
            # 为了实现更平滑的过渡，创建一个渐变mask
            # 应用根据strength调整的阈值
            threshold = int(255 * (1 - request.strength))
            _, thresholded_mask = cv2.threshold(mask_cv, threshold, 255, cv2.THRESH_BINARY)
            
            # 应用填充颜色到掩码区域
            img_cv[thresholded_mask > 0] = avg_color
            
            # 应用轻微的模糊使过渡更平滑
            img_cv = cv2.GaussianBlur(img_cv, (5, 5), 0)
            
            # 转换回PIL格式
            template_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            
        else:  # "auto" or other
            # 自动选择最佳方法
            # 这里选择mask方法，因为通常效果更好
            mask_generator = get_mask_generator()
            face_mask = mask_generator.generate_face_mask(face_image)
            
            # 创建图像副本
            template_image = face_image.copy()
            
            # 将图像转换为OpenCV格式
            img_cv = cv2.cvtColor(np.array(template_image), cv2.COLOR_RGB2BGR)
            mask_cv = np.array(face_mask)
            
            # 使用图像中检测到的人脸周围的颜色
            avg_color = cv2.mean(img_cv, mask=mask_cv)[:3]
            
            # 创建轻微的渐变过渡
            blurred_mask = cv2.GaussianBlur(mask_cv, (15, 15), 0)
            
            # 根据强度参数调整效果
            strength_adjusted_mask = cv2.convertScaleAbs(blurred_mask, alpha=request.strength)
            
            # 创建一个与原始图像相同尺寸的图像，填充平均颜色
            color_fill = np.full_like(img_cv, avg_color)
            
            # 使用mask进行加权混合
            for i in range(3):  # 遍历BGR通道
                img_cv[:, :, i] = (
                    img_cv[:, :, i] * (1 - strength_adjusted_mask / 255.0) +
                    color_fill[:, :, i] * (strength_adjusted_mask / 255.0)
                )
            
            # 转换回PIL格式
            template_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        # 保存模板图像
        template_id = f"template_{uuid.uuid4().hex}.png"
        template_path = os.path.join(config.TEMPLATE_UPLOAD_DIR, template_id)
        template_image.save(template_path)
        
        # 返回结果
        return {
            "success": True,
            "template_image": image_to_base64(template_image),
            "template_id": template_id,
            "message": f"成功生成无脸模板图像，使用方法: {request.method}"
        }
    
    except Exception as e:
        logger.error(f"模板生成错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/face/generate-mask", response_model=MaskGenerationResponse)
async def generate_mask(request: MaskGenerationRequest):
    """生成掩码图像API"""
    try:
        # 获取源图像
        source_image = None
        if request.image:
            # 使用提供的base64图像
            source_image = base64_to_image(request.image)
        elif request.image_id:
            # 尝试在不同目录中查找图像
            for dir_path in [config.FACE_UPLOAD_DIR, config.TEMPLATE_UPLOAD_DIR]:
                image_path = os.path.join(dir_path, request.image_id)
                if os.path.exists(image_path):
                    source_image = Image.open(image_path).convert("RGB")
                    break
            
            if source_image is None:
                raise HTTPException(status_code=404, detail=f"找不到图像: {request.image_id}")
        else:
            raise HTTPException(status_code=400, detail="必须提供image或image_id")
        
        # 初始化掩码生成器
        mask_generator = get_mask_generator()
        
        # 根据掩码类型生成掩码
        mask_image = None
        
        if request.mask_type == "face":
            # 生成人脸区域掩码
            mask_image = mask_generator.generate_face_mask(source_image, padding_ratio=request.padding_ratio)
        elif request.mask_type == "template":
            # 生成无脸模板区域掩码
            mask_image = mask_generator.generate_template_mask(source_image)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的掩码类型: {request.mask_type}")
        
        # 保存掩码图像
        mask_id = f"mask_{uuid.uuid4().hex}.png"
        mask_path = os.path.join(config.MASK_UPLOAD_DIR, mask_id)
        mask_image.save(mask_path)
        
        # 返回结果
        return {
            "success": True,
            "mask_image": image_to_base64(mask_image),
            "mask_id": mask_id,
            "message": f"成功生成{request.mask_type}类型掩码"
        }
    
    except Exception as e:
        logger.error(f"掩码生成错误: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) 