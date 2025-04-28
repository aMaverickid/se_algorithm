"""
人脸检测模块，使用MediaPipe Face Mesh定位图像中的人脸并生成无脸模板
"""
import logging
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFilter
import sys
from pathlib import Path
import time

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    人脸检测器，使用MediaPipe Face Mesh定位图像中的人脸并生成无脸模板
    
    支持多种掩码类型，适用于不同场景：
    
    1. 透明背景 (mask_type='transparent')：
       - 将人脸区域替换为完全透明背景
       - 适用于需要将人脸与背景分离的场景
       
    2. 白色填充 (mask_type='white')：
       - 将人脸区域替换为纯白色
       - 适用于大多数AI填充/修复模型，如SD-inpainting
       
    3. 黑色填充 (mask_type='black')：
       - 将人脸区域替换为纯黑色
       - 适用于某些特定的图像处理管线
    
    4. 红色填充 (mask_type='red')：
       - 将人脸区域替换为红色
       - 适用于需要明显标记人脸区域的情况
       
    5. 边缘保留 (mask_type='edge_preserved')：
       - 只移除面部内部，保留面部轮廓
       - 适用于需要保持面部结构但替换内容的情况
    
    6. Inpainting优化 (mask_type='inpainting')：
       - 专为IP-Adapter等AI图像修复模型优化的掩码
       - 默认使用白色填充，支持边缘渐变过渡
       - 当edge_blur_radius > 0时，会在边缘处创建平滑过渡，改善融合效果
       - 推荐参数：mask_type='inpainting', edge_blur_radius=5
       
    对于IP-Adapter的inpainting任务，推荐使用以下组合：
    - 基本模板: mask_type='inpainting'
    - 较好的边缘过渡: mask_type='inpainting', edge_blur_radius=3~10
    - 最佳效果可能需要实验不同的edge_blur_radius值
    """

    def __init__(self, min_detection_confidence=0.5):
        """
        初始化人脸检测器
        
        Args:
            min_detection_confidence: 检测阈值，越高越严格，范围0-1
        """
        logger.info(f"初始化FaceDetector，检测阈值: {min_detection_confidence}")
        
        # mediapipe 面部网格模块
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # 创建面部网格检测器实例
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, # 静态图像模式
            max_num_faces=1, # 最大检测人脸数
            min_detection_confidence=min_detection_confidence, # 检测阈值
            refine_landmarks=True # 细化关键点
        )
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            self.face_mesh.close()
    
    def detect_face_mesh(self, image):
        """
        使用Face Mesh检测图像中的主要人脸，获取详细的面部网格点
        
        Args:
            image: PIL图像(RGB)
        Returns:
            检测到的主要人脸，包含面部网格点信息
        """
        image_np_rgb = np.array(image)
        
        # 确保图像为RGB格式
        if len(image_np_rgb.shape) != 3:
            if len(image_np_rgb.shape) == 2:  # 灰度图像
                image_np_rgb = cv2.cvtColor(image_np_rgb, cv2.COLOR_GRAY2RGB)
            elif image_np_rgb.shape[2] == 4:  # RGBA图像
                image_np_rgb = image_np_rgb[:, :, :3]
            else:
                logger.error(f"图像格式错误: {image_np_rgb.shape}, 图像模式: {image.mode}")
                raise ValueError(f"不支持的图像格式: {image_np_rgb.shape}")
        
        # 执行面部网格检测
        results = self.face_mesh.process(image_np_rgb)
        
        # 如果没有检测到面部网格，则抛出异常
        if not results.multi_face_landmarks:
            logger.warning("未检测到人脸")
            raise ValueError("未检测到人脸")
        
        # 提取面部网格点（全部478个点）
        image_height, image_width = image_np_rgb.shape[:2]
        face_landmarks = results.multi_face_landmarks[0]
        
        # 转换为绝对坐标
        mesh_points = []
        for landmark in face_landmarks.landmark:
            x, y = int(landmark.x * image_width), int(landmark.y * image_height)
            mesh_points.append((x, y))
        
        # 构建返回数据
        face_data = {
            'image': image,
            'mesh_points': mesh_points,  # 完整的面部网格点
        }
        
        return face_data
    
    def generate_mask(self, face, blur_radius=4, remove_holes=True, detailed_edges=False):
        """
        为检测到的人脸生成掩码图像
        
        Args:
            face_data: 检测到的人脸数据，包含'image'和'mesh_points'
                image: 原始人脸图像，PIL格式
                mesh_points: 面部网格点（必要）
            blur_radius: 高斯模糊半径，用于平滑掩码边缘，默认值由2增加到4
            remove_holes: 是否移除掩码中的小孔洞
            detailed_edges: 是否生成更细致的边缘

        Returns:
            PIL掩码图像
        """
        face_data = self.detect_face_mesh(face)
        
        image = face_data['image']
        width, height = image.size

        mesh_points = face_data['mesh_points']
        
        # 定义面部轮廓线的点索引（根据MediaPipe Face Mesh的定义）
        face_oval_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
        ]
        
        contour_points = [mesh_points[idx] for idx in face_oval_indices]
        contour_array = np.array(contour_points, dtype=np.int32)
        
        # 扩大掩码区域，确保覆盖整个面部
        # 计算轮廓的中心点
        center_x = np.mean(contour_array[:, 0])
        center_y = np.mean(contour_array[:, 1])
        
        # 向外扩展轮廓点（扩大15%）
        expanded_contour = []
        for point in contour_array:
            direction_x = point[0] - center_x
            direction_y = point[1] - center_y
            
            # 计算点到中心的距离
            distance = max(np.sqrt(direction_x**2 + direction_y**2), 1e-6)
            
            # 扩展系数 - 向外扩展15%
            scale_factor = 1.15
            new_x = int(center_x + direction_x * scale_factor)
            new_y = int(center_y + direction_y * scale_factor)
            
            # 确保坐标在图像范围内
            new_x = max(0, min(width - 1, new_x))
            new_y = max(0, min(height - 1, new_y))
            
            expanded_contour.append([new_x, new_y])
        
        expanded_contour_array = np.array(expanded_contour, dtype=np.int32)
        
        # 创建掩码
        mask_np = np.zeros((height, width), dtype=np.uint8)
        
        if detailed_edges:
            # 使用扩展后的轮廓
            expanded_contour_reshaped = expanded_contour_array.reshape((-1, 1, 2))
            cv2.drawContours(mask_np, [expanded_contour_reshaped], 0, 255, -1)
        else:
            # 使用扩展后的轮廓填充多边形
            cv2.fillPoly(mask_np, [expanded_contour_array], 255)
        
        # 如果需要移除小孔洞
        if remove_holes:
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_np, contours, -1, 255, -1)
        
        mask = Image.fromarray(mask_np)
        
        # 应用高斯模糊以平滑掩码边缘
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return mask

    def generate_template(self, face, background_color=None, blur_radius=2, remove_holes=True, detailed_edges=False, mask_type='inpainting', edge_blur_radius=0):
        """
        生成无脸模板图像，将检测到的人脸区域替换为特定背景
        
        Args:
            face: PIL图像
            background_color: 背景颜色，默认为根据mask_type决定
            blur_radius: 高斯模糊半径，用于平滑掩码边缘，默认为2
            remove_holes: 是否移除掩码中的小孔洞，默认为True
            detailed_edges: 是否生成更细致的边缘，默认为False
            mask_type: 掩码类型，可选值：
                'transparent': 透明背景 (默认)
                'white': 白色填充
                'black': 黑色填充
                'red': 红色填充
                'edge_preserved': 保留边缘轮廓
                'inpainting': 专为inpainting优化的掩码
            edge_blur_radius: 边缘额外模糊半径，用于为inpainting创建更平滑的过渡，默认为0
            
        Returns:
            无脸模板PIL图像(RGBA)
        """
        face_mask = self.generate_mask(face, blur_radius, remove_holes, detailed_edges)
        
        # 转换掩码为L模式(灰度)
        mask_l = face_mask.convert("L")
        
        # 如果指定了额外的边缘模糊，创建更平滑的边缘过渡
        if edge_blur_radius > 0:
            # 创建更模糊的掩码用于边缘过渡
            edge_mask = mask_l.filter(ImageFilter.GaussianBlur(radius=edge_blur_radius))
            # 将模糊掩码转换为numpy数组以便处理
            edge_mask_np = np.array(edge_mask)
            # 创建渐变边缘效果 - 只在中间值区域创建渐变
            # 保持纯白区域为白色，纯黑区域为黑色
            gradient_mask_np = np.zeros_like(edge_mask_np)
            # 将50-200之间的值线性映射到0-255，创建更平滑的渐变
            mid_range = (edge_mask_np > 50) & (edge_mask_np < 200)
            gradient_mask_np[mid_range] = ((edge_mask_np[mid_range] - 50) * (255 / 150)).astype(np.uint8)
            gradient_mask_np[edge_mask_np >= 200] = 255
            # 创建新的渐变掩码
            mask_l = Image.fromarray(gradient_mask_np)
        
        # 创建模板图像（确保为RGBA格式）
        if face.mode != 'RGBA':
            template = face.convert('RGBA')
        else:
            template = face.copy()
        
        width, height = template.size
        
        # 根据mask_type设置适当的背景颜色
        if background_color is None:
            if mask_type == 'transparent':
                background_color = (0, 0, 0, 0)  # 完全透明
            elif mask_type == 'white':
                background_color = (255, 255, 255, 255)  # 白色
            elif mask_type == 'black':
                background_color = (0, 0, 0, 255)  # 黑色
            elif mask_type == 'red':
                background_color = (255, 0, 0, 255)  # 红色
            elif mask_type == 'edge_preserved':
                # 对于edge_preserved，我们稍后会特殊处理
                background_color = (0, 0, 0, 0)
            elif mask_type == 'inpainting':
                # 对于inpainting优化的掩码，使用白色
                background_color = (255, 255, 255, 255)
            else:
                # 默认为透明
                background_color = (0, 0, 0, 0)
        
        transparent_layer = Image.new('RGBA', (width, height), background_color)
        
        # 对于edge_preserved类型，我们需要特殊处理以保留边缘
        if mask_type == 'edge_preserved':
            # 创建一个略小的掩码，用于保留边缘
            face_data = self.detect_face_mesh(face)
            mesh_points = face_data['mesh_points']
            
            # 使用轮廓点创建掩码
            face_oval_indices = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
            ]
            
            contour_points = [mesh_points[idx] for idx in face_oval_indices]
            contour_array = np.array(contour_points, dtype=np.int32)
            
            # 创建两个掩码：外部轮廓和内部轮廓
            outer_mask = np.zeros((height, width), dtype=np.uint8)
            inner_mask = np.zeros((height, width), dtype=np.uint8)
            
            # 填充外部掩码
            cv2.fillPoly(outer_mask, [contour_array], 255)
            
            # 创建稍小的内部轮廓（向内收缩5像素）
            center_x = np.mean(contour_array[:, 0])
            center_y = np.mean(contour_array[:, 1])
            inner_contour = []
            for point in contour_array:
                direction_x = point[0] - center_x
                direction_y = point[1] - center_y
                
                # 避免除以零
                distance = max(np.sqrt(direction_x**2 + direction_y**2), 1e-6)
                
                # 向内收缩5像素
                scale_factor = (distance - 5) / distance
                new_x = int(center_x + direction_x * scale_factor)
                new_y = int(center_y + direction_y * scale_factor)
                
                inner_contour.append([new_x, new_y])
            
            inner_contour_array = np.array(inner_contour, dtype=np.int32)
            cv2.fillPoly(inner_mask, [inner_contour_array], 255)
            
            # 边缘掩码 = 外部掩码 - 内部掩码
            edge_mask = outer_mask - inner_mask
            
            # 创建边缘保留图像
            edge_template = template.copy()
            edge_mask_pil = Image.fromarray(edge_mask)
            inner_mask_pil = Image.fromarray(inner_mask)
            
            # 应用内部掩码（填充内部）
            inner_template = Image.composite(transparent_layer, edge_template, inner_mask_pil.convert("L"))
            
            # 保留原始模板（保留边缘）
            template = inner_template
        elif mask_type == 'inpainting':
            # 为inpainting创建特殊优化的掩码
            # 这种掩码会为边缘创建更平滑的过渡，以帮助模型更好地融合生成内容
            
            # 首先应用普通掩码替换
            base_template = Image.composite(transparent_layer, template, mask_l)
            
            # 如果指定了边缘模糊，创建额外的混合层
            if edge_blur_radius > 0:
                # 创建一个混合掩码，使边缘区域保留部分原始图像信息
                original_template = template.copy()
                # 在边缘区域使用透明度渐变
                edge_blend = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                # 使用黑白掩码生成透明度渐变
                mask_array = np.array(mask_l)
                # 只在边缘区域（非纯黑非纯白）创建透明度渐变
                edge_region = (mask_array > 30) & (mask_array < 225)
                
                # 提取原始图像的RGB
                original_np = np.array(original_template)
                # 创建新的RGBA图像，边缘区域使用原始RGB但有渐变透明度
                blend_np = np.zeros((height, width, 4), dtype=np.uint8)
                blend_np[edge_region, 0:3] = original_np[edge_region, 0:3]
                # 根据掩码值设置透明度 - 接近黑色区域更不透明，接近白色区域更透明
                blend_np[edge_region, 3] = 255 - mask_array[edge_region]
                
                edge_blend = Image.fromarray(blend_np)
                
                # 将基础模板和边缘混合层合并
                template = Image.alpha_composite(base_template, edge_blend)
            else:
                template = base_template
        else:
            # 使用掩码合并原图和透明层
            # 掩码中白色区域(255)是人脸区域，将被透明层替换
            # 掩码中黑色区域(0)是非人脸区域，将保留原图
            template = Image.composite(transparent_layer, template, mask_l)
            
        return template