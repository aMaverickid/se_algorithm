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
    """使用MediaPipe进行人脸检测和关键点识别"""
    
    def __init__(self, min_detection_confidence=0.5):
        """
        初始化人脸检测器
        
        Args:
            min_detection_confidence (float): 最小检测置信度
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.min_detection_confidence = min_detection_confidence
        
        # 初始化检测器
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
        
        # 初始化关键点检测器
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence
        )
    
    def detect_faces(self, image):
        """
        检测图像中的人脸
        
        Args:
            image: PIL图像或图像路径
            
        Returns:
            包含人脸边界框坐标的列表，格式为 [xmin, ymin, width, height, confidence]
        """
        # 确保图像为PIL格式
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        if not isinstance(image, Image.Image):
            raise ValueError("image必须是PIL.Image.Image类型或有效的图像路径")
        
        # 转换为OpenCV格式
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 执行检测
        results = self.face_detection.process(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = image_cv.shape
                
                xmin = max(0, int(bbox.xmin * iw))
                ymin = max(0, int(bbox.ymin * ih))
                width = min(int(bbox.width * iw), iw - xmin)
                height = min(int(bbox.height * ih), ih - ymin)
                
                faces.append([xmin, ymin, width, height, detection.score[0]])
        
        return faces
    
    def detect_face_landmarks(self, image):
        """
        检测图像中的人脸关键点
        
        Args:
            image: PIL图像或图像路径
            
        Returns:
            关键点列表和人脸边界框
        """
        # 确保图像为PIL格式
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        if not isinstance(image, Image.Image):
            raise ValueError("image必须是PIL.Image.Image类型或有效的图像路径")
        
        # 转换为OpenCV格式
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 执行关键点检测
        results = self.face_mesh.process(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        
        landmarks = []
        bbox = None
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # 提取关键点
            ih, iw, _ = image_cv.shape
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * iw)
                y = int(landmark.y * ih)
                landmarks.append((x, y))
            
            # 计算包围盒
            x_coords = [landmark.x * iw for landmark in face_landmarks.landmark]
            y_coords = [landmark.y * ih for landmark in face_landmarks.landmark]
            
            xmin = max(0, int(min(x_coords)))
            ymin = max(0, int(min(y_coords)))
            xmax = min(int(max(x_coords)), iw)
            ymax = min(int(max(y_coords)), ih)
            
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
        
        return landmarks, bbox
    
    def crop_face(self, image, padding=0.2):
        """
        裁剪图像中的人脸区域
        
        Args:
            image: PIL图像或图像路径
            padding (float): 边界框周围的额外填充比例
            
        Returns:
            裁剪后的PIL图像，如果没有检测到人脸则返回None
        """
        # 确保图像为PIL格式
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # 检测人脸
        faces = self.detect_faces(image)
        
        if not faces:
            logger.warning("未检测到人脸")
            return None
        
        # 找到最大的人脸
        main_face = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[0]
        x, y, w, h = main_face[:4]
        
        # 添加padding
        padding_x = int(w * padding)
        padding_y = int(h * padding)
        
        # 计算裁剪区域
        left = max(0, x - padding_x)
        top = max(0, y - padding_y)
        right = min(image.width, x + w + padding_x)
        bottom = min(image.height, y + h + padding_y)
        
        # 裁剪人脸
        face_image = image.crop((left, top, right, bottom))
        
        return face_image
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close() 