"""
掩码生成模块，用于生成人脸区域的掩码
"""
import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from processors.face_detector import FaceDetector

logger = logging.getLogger(__name__)

class MaskGenerator:
    """人脸掩码生成器，用于inpainting任务"""
    
    def __init__(self, face_detector=None, feather_amount=10):
        """
        初始化掩码生成器
        
        Args:
            face_detector: 人脸检测器实例，如果为None将创建一个新实例
            feather_amount (int): 边缘羽化程度
        """
        self.face_detector = face_detector if face_detector else FaceDetector()
        self.feather_amount = feather_amount
    
    def generate_face_mask(self, image, padding_ratio=0.1):
        """
        生成人脸区域的掩码
        
        Args:
            image: PIL图像或图像路径
            padding_ratio (float): 人脸边界框的扩展比例
            
        Returns:
            PIL格式的掩码图像，人脸区域为白色(255)，背景为黑色(0)
        """
        # 确保图像为PIL格式
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        if not isinstance(image, Image.Image):
            raise ValueError("image必须是PIL.Image.Image类型或有效的图像路径")
        
        # 检测人脸
        faces = self.face_detector.detect_faces(image)
        
        if not faces:
            logger.warning("未检测到人脸，将返回空掩码")
            return Image.new("L", image.size, 0)
        
        # 创建掩码画布
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # 对所有检测到的人脸生成掩码
        for face in faces:
            x, y, w, h = face[:4]
            
            # 应用padding
            pad_w = int(w * padding_ratio)
            pad_h = int(h * padding_ratio)
            
            # 计算扩展后的边界框
            left = max(0, x - pad_w)
            top = max(0, y - pad_h)
            right = min(image.width, x + w + pad_w)
            bottom = min(image.height, y + h + pad_h)
            
            # 绘制填充的椭圆作为掩码
            # 椭圆比矩形更接近人脸形状
            ellipse_box = (left, top, right, bottom)
            draw.ellipse(ellipse_box, fill=255)
        
        # 羽化边缘
        if self.feather_amount > 0:
            mask = self._feather_mask(mask)
        
        return mask
    
    def generate_face_landmarks_mask(self, image, padding_ratio=0.05, include_all_landmarks=False):
        """
        基于人脸关键点生成更精确的掩码
        
        Args:
            image: PIL图像或图像路径
            padding_ratio (float): 轮廓的扩展比例
            include_all_landmarks (bool): 是否包含所有关键点（如果为False，只使用脸部轮廓点）
            
        Returns:
            PIL格式的掩码图像
        """
        # 确保图像为PIL格式
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # 获取人脸关键点
        landmarks, _ = self.face_detector.detect_face_landmarks(image)
        
        if not landmarks:
            logger.warning("未检测到人脸关键点，将返回基于边界框的掩码")
            return self.generate_face_mask(image, padding_ratio)
        
        # 创建掩码画布
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        
        # 转换为OpenCV格式以便于计算凸包
        landmarks_np = np.array(landmarks)
        
        # 如果只使用脸部轮廓点，则筛选相关的点
        if not include_all_landmarks:
            # 这是一个近似值，MediaPipe的脸部轮廓在不同版本可能有所不同
            # 一般来说，前面的点通常对应脸部轮廓
            contour_indices = list(range(0, 150))
            landmarks_np = landmarks_np[contour_indices]
        
        # 计算凸包
        hull = cv2.convexHull(landmarks_np)
        
        # 扩展凸包
        center = np.mean(hull, axis=0)[0]
        hull_expanded = []
        for point in hull:
            x, y = point[0]
            # 从中心向外扩展
            vector = np.array([x - center[0], y - center[1]])
            norm = np.linalg.norm(vector)
            if norm > 0:  # 避免除零错误
                vector = vector / norm
                expand_amount = max(image.width, image.height) * padding_ratio
                x_new = int(x + vector[0] * expand_amount)
                y_new = int(y + vector[1] * expand_amount)
                hull_expanded.append((x_new, y_new))
            else:
                hull_expanded.append((x, y))
        
        # 绘制填充的多边形
        draw.polygon(hull_expanded, fill=255)
        
        # 羽化边缘
        if self.feather_amount > 0:
            mask = self._feather_mask(mask)
        
        return mask
    
    def _feather_mask(self, mask):
        """
        羽化掩码边缘
        
        Args:
            mask: PIL掩码图像
            
        Returns:
            羽化后的PIL掩码图像
        """
        # 转换为OpenCV格式
        mask_cv = np.array(mask)
        
        # 应用高斯模糊
        mask_cv = cv2.GaussianBlur(mask_cv, (self.feather_amount * 2 + 1, self.feather_amount * 2 + 1), 0)
        
        # 转回PIL格式
        return Image.fromarray(mask_cv)
    
    def generate_template_mask(self, template_image, target_size=None):
        """
        为无脸模板图像生成对应的掩码
        
        Args:
            template_image: 无脸模板图像
            target_size: 目标尺寸，如果为None则使用原始尺寸
            
        Returns:
            无脸区域的掩码
        """
        # 确保图像为PIL格式
        if isinstance(template_image, str):
            template_image = Image.open(template_image).convert("RGB")
        
        # 调整尺寸（如需要）
        if target_size and template_image.size != target_size:
            template_image = template_image.resize(target_size, Image.LANCZOS)
        
        # 转换为OpenCV格式
        template_cv = cv2.cvtColor(np.array(template_image), cv2.COLOR_RGB2BGR)
        
        # 转换为灰度图
        gray = cv2.cvtColor(template_cv, cv2.COLOR_BGR2GRAY)
        
        # 应用自适应阈值，检测无脸区域
        # 调整参数可能需要根据模板图像特性进行微调
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 应用形态学操作来优化掩码
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建掩码
        mask = np.zeros_like(gray)
        
        # 如果找到轮廓，选择最大的作为掩码
        if contours:
            # 按轮廓面积排序
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # 绘制最大轮廓
            cv2.drawContours(mask, [contours[0]], -1, 255, -1)
            
            # 羽化边缘
            mask = cv2.GaussianBlur(mask, (self.feather_amount * 2 + 1, self.feather_amount * 2 + 1), 0)
        
        # 转回PIL格式
        return Image.fromarray(mask) 