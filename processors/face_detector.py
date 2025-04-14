"""
人脸检测模块，用于定位图像中的人脸
"""
import logging
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

class FaceDetector:
    """人脸检测器，使用mediapipe库定位图像中的人脸"""

    def __init__(self, min_detection_confidence=0.5):
        """
        初始化人脸检测器
        
        Args:
            min_detection_confidence: 检测阈值，越高越严格，范围0-1
        """
        logger.info(f"初始化FaceDetector，检测阈值: {min_detection_confidence}")
        
        # 初始化mediapipe人脸检测模块
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 创建人脸检测器实例
        self.face_detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
    
    def detect_faces(self, image):
        """
        检测图像中的所有人脸
        
        Args:
            image: PIL图像或numpy数组或图像路径
            
        Returns:
            检测到的人脸列表，每个人脸包含边界框和关键点信息
        """
        # 预处理图像
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        if isinstance(image, Image.Image):
            # 将PIL图像转换为numpy数组
            image_np = np.array(image)
        else:
            # 已经是numpy数组
            image_np = image
            
        # 确保图像是RGB格式
        if len(image_np.shape) == 2:  # 灰度图像
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:  # RGBA图像
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
        # 图像需要转换为RGB格式供mediapipe处理
        image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
        # 执行人脸检测
        results = self.face_detector.process(image_np_rgb)
        
        # 解析结果
        faces = []
        if results.detections:
            for detection in results.detections:
                face_data = {
                    'score': detection.score[0],
                    'box': self._get_box_from_detection(detection, image_np.shape[1], image_np.shape[0]),
                    'keypoints': self._get_keypoints_from_detection(detection, image_np.shape[1], image_np.shape[0])
                }
                faces.append(face_data)
                
        return faces
    
    def detect_face(self, image):
        """
        检测图像中的主要人脸（得分最高或面积最大的一个）
        
        Args:
            image: PIL图像或numpy数组或图像路径
            
        Returns:
            主要人脸信息，如果未检测到则返回None
        """
        faces = self.detect_faces(image)
        
        if not faces:
            logger.warning("未检测到人脸")
            return None
            
        # 选择得分最高的人脸
        main_face = max(faces, key=lambda face: face['score'])
        logger.debug(f"检测到主要人脸，得分: {main_face['score']:.2f}")
        
        return main_face
    
    def get_face_box(self, image, expand_ratio=0.0):
        """
        获取图像中主要人脸的边界框
        
        Args:
            image: PIL图像或numpy数组或图像路径
            expand_ratio: 边界框扩展比例，0.1表示在各方向扩展10%
            
        Returns:
            (x_min, y_min, x_max, y_max)格式的边界框，如果未检测到则返回None
        """
        face = self.detect_face(image)
        
        if not face:
            return None
            
        box = face['box']
        
        # 应用扩展比例
        if expand_ratio > 0:
            width = box[2] - box[0]
            height = box[3] - box[1]
            
            # 计算扩展量
            x_expand = width * expand_ratio
            y_expand = height * expand_ratio
            
            # 扩展边界框
            expanded_box = (
                max(0, box[0] - x_expand),
                max(0, box[1] - y_expand),
                box[2] + x_expand,
                box[3] + y_expand
            )
            
            return expanded_box
            
        return box
    
    def _get_box_from_detection(self, detection, image_width, image_height):
        """从mediapipe检测结果提取边界框坐标"""
        bbox = detection.location_data.relative_bounding_box
        x_min = max(0, bbox.xmin * image_width)
        y_min = max(0, bbox.ymin * image_height)
        width = bbox.width * image_width
        height = bbox.height * image_height
        
        # 确保边界框不超出图像范围
        x_max = min(image_width, x_min + width)
        y_max = min(image_height, y_min + height)
        
        return (int(x_min), int(y_min), int(x_max), int(y_max))
    
    def _get_keypoints_from_detection(self, detection, image_width, image_height):
        """从mediapipe检测结果提取关键点坐标"""
        keypoints = {}
        
        # 提取关键点（右眼、左眼、鼻尖、嘴巴中心、右耳、左耳）
        keypoint_names = ['right_eye', 'left_eye', 'nose_tip', 'mouth_center', 'right_ear', 'left_ear']
        
        for i, name in enumerate(keypoint_names):
            if i < len(detection.location_data.relative_keypoints):
                kp = detection.location_data.relative_keypoints[i]
                keypoints[name] = (int(kp.x * image_width), int(kp.y * image_height))
        
        return keypoints