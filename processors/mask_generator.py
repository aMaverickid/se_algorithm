"""
掩码生成模块，用于生成人脸区域的掩码
"""
import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw
import sys
import os
import uuid
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from processors.face_detector import FaceDetector
from utils.image_utils import base64_to_image, image_to_base64

logger = logging.getLogger(__name__)

class MaskGenerator:
    """掩码生成器，用于生成用于图像操作的掩码"""
    
    def __init__(self):
        """初始化掩码生成器"""
        logger.info("初始化MaskGenerator")
        self.face_detector = FaceDetector()
        
    def generate_face_mask(self, image, target_size=None, padding_ratio=0.1):
        """
        生成人脸区域的掩码（人脸区域为白色，背景为黑色）
        
        Args:
            image: PIL图像、numpy数组、图像路径或Base64编码的图像
            target_size: 目标掩码尺寸 (width, height)，如果为None则使用原始图像尺寸
            padding_ratio: 边界框扩展比例，0.1表示在各方向扩展10%
            
        Returns:
            PIL格式的掩码图像，8位灰度图，人脸区域为白色(255)，背景为黑色(0)
        """
        # 处理Base64编码的图像
        if isinstance(image, str) and image.startswith(('data:image', 'iVBORw0KGgo')):
            image = base64_to_image(image)
        
        # 如果是路径，转换为PIL图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            
        # 保存原始图像尺寸
        if isinstance(image, Image.Image):
            original_size = image.size
        else:
            # 如果是numpy数组
            original_size = (image.shape[1], image.shape[0])
            
        # 目标尺寸
        if target_size is None:
            target_size = original_size
        
        # 获取人脸边界框
        face_box = self.face_detector.get_face_box(image, expand_ratio=padding_ratio)
        
        if face_box is None:
            logger.warning("未检测到人脸，返回空掩码")
            # 创建一个空的黑色掩码
            mask = Image.new('L', target_size, 0)
            return mask
        
        # 创建掩码图像
        mask = Image.new('L', original_size, 0)
        draw = ImageDraw.Draw(mask)
        
        # 在掩码上填充人脸区域
        draw.rectangle(face_box, fill=255)
        
        # 调整掩码尺寸以匹配目标尺寸
        if target_size != original_size:
            mask = mask.resize(target_size, Image.LANCZOS)
            
        return mask
    
    def generate_template_mask(self, image, target_size=None, threshold=50):
        """
        生成模板区域的掩码（无脸区域为白色，其余区域为黑色）
        
        Args:
            image: PIL图像、numpy数组、图像路径或Base64编码的图像
            target_size: 目标掩码尺寸 (width, height)，如果为None则使用原始图像尺寸
            threshold: 灰度阈值，用于检测无脸区域
            
        Returns:
            PIL格式的掩码图像，8位灰度图，无脸区域为白色(255)，其余区域为黑色(0)
        """
        # 处理Base64编码的图像
        if isinstance(image, str) and image.startswith(('data:image', 'iVBORw0KGgo')):
            image = base64_to_image(image)
        
        # 如果是路径，转换为PIL图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            
        # 保存原始图像尺寸
        if isinstance(image, Image.Image):
            original_size = image.size
            # 转换为numpy数组
            image_np = np.array(image)
        else:
            # 如果已经是numpy数组
            image_np = image
            original_size = (image_np.shape[1], image_np.shape[0])
            
        # 目标尺寸
        if target_size is None:
            target_size = original_size
        
        # 转换为灰度图像
        if len(image_np.shape) == 3:  # 彩色图像
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
            
        # 尝试多种方法检测无脸区域
        # 方法1：基于阈值二值化
        _, binary_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # 应用形态学操作清理掩码
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # 转换为PIL图像
        mask = Image.fromarray(binary_mask)
        
        # 调整掩码尺寸以匹配目标尺寸
        if target_size != (mask.width, mask.height):
            mask = mask.resize(target_size, Image.LANCZOS)
            
        return mask
    
    def save_mask(self, mask, filename=None):
        """
        保存掩码图像
        
        Args:
            mask: PIL格式的掩码图像
            filename: 保存的文件名，如果为None则自动生成
            
        Returns:
            保存的掩码文件路径
        """
        # 确保上传目录存在
        os.makedirs(config.MASK_UPLOAD_DIR, exist_ok=True)
        
        # 如果未提供文件名，则生成随机文件名
        if filename is None:
            filename = f"mask_{uuid.uuid4().hex}.png"
            
        # 确保文件名有正确的扩展名
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename += '.png'
        
        # 构建完整路径
        mask_path = os.path.join(config.MASK_UPLOAD_DIR, filename)
        
        # 保存掩码图像
        mask.save(mask_path)
        logger.info(f"掩码保存至: {mask_path}")
        
        return mask_path
    