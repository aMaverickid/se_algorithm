"""
图像处理工具函数
"""
import os
import logging
import uuid
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import base64
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

def resize_image(image, target_size, keep_aspect_ratio=True):
    """
    调整图像尺寸
    
    Args:
        image: PIL图像或图像路径
        target_size: 目标尺寸，可以是(width, height)元组或单个整数
        keep_aspect_ratio: 是否保持宽高比
        
    Returns:
        调整后的PIL图像
    """
    # 确保图像为PIL格式
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    if not isinstance(image, Image.Image):
        raise ValueError("image必须是PIL.Image.Image类型或有效的图像路径")
    
    # 处理目标尺寸
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    if keep_aspect_ratio:
        # 计算调整后的尺寸，保持宽高比
        width, height = image.size
        aspect_ratio = width / height
        
        if width > height:
            new_width = target_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size[1]
            new_width = int(new_height * aspect_ratio)
            
        # 确保不超过目标尺寸
        if new_width > target_size[0]:
            new_width = target_size[0]
            new_height = int(new_width / aspect_ratio)
        if new_height > target_size[1]:
            new_height = target_size[1]
            new_width = int(new_height * aspect_ratio)
            
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # 创建目标尺寸的画布并居中粘贴
        result = Image.new("RGB", target_size, (0, 0, 0))
        paste_x = (target_size[0] - new_width) // 2
        paste_y = (target_size[1] - new_height) // 2
        result.paste(resized_image, (paste_x, paste_y))
        
        return result
    else:
        # 直接调整到目标尺寸
        return image.resize(target_size, Image.LANCZOS)

def square_pad(image, target_size=None, background_color=(0, 0, 0)):
    """
    将图像填充为正方形
    
    Args:
        image: PIL图像或图像路径
        target_size: 目标尺寸，如果为None则使用图像的最大边长
        background_color: 背景颜色
        
    Returns:
        填充后的PIL图像
    """
    # 确保图像为PIL格式
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    
    if not isinstance(image, Image.Image):
        raise ValueError("image必须是PIL.Image.Image类型或有效的图像路径")
    
    width, height = image.size
    
    # 确定目标尺寸
    if target_size is None:
        target_size = max(width, height)
    
    # 创建正方形画布
    result = Image.new("RGB", (target_size, target_size), background_color)
    
    # 计算粘贴位置
    paste_x = (target_size - width) // 2
    paste_y = (target_size - height) // 2
    
    # 粘贴原始图像
    result.paste(image, (paste_x, paste_y))
    
    return result

def image_to_base64(image):
    """
    将PIL图像转换为Base64编码的字符串
    
    Args:
        image: PIL图像
        
    Returns:
        Base64编码的字符串
    """
    if not isinstance(image, Image.Image):
        raise ValueError("image必须是PIL.Image.Image类型")
    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_image(base64_string):
    """
    将Base64编码的字符串转换为PIL图像
    
    Args:
        base64_string: Base64编码的字符串
        
    Returns:
        PIL图像
    """
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

def save_temp_image(image, prefix="temp_", suffix=".png"):
    """
    将图像保存为临时文件
    
    Args:
        image: PIL图像
        prefix: 文件名前缀
        suffix: 文件扩展名
        
    Returns:
        临时文件路径
    """
    if not isinstance(image, Image.Image):
        raise ValueError("image必须是PIL.Image.Image类型")
    
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, prefix=prefix, suffix=suffix)
    temp_path = temp_file.name
    temp_file.close()
    
    # 保存图像
    image.save(temp_path)
    
    return temp_path

def save_output_image(image, output_dir=None, filename=None, prefix="output_", suffix=".png"):
    """
    保存输出图像到指定目录
    
    Args:
        image: PIL图像
        output_dir: 输出目录
        filename: 文件名，如果为None则自动生成
        prefix: 文件名前缀（当filename为None时使用）
        suffix: 文件扩展名（当filename为None时使用）
        
    Returns:
        保存的文件路径
    """
    if not isinstance(image, Image.Image):
        raise ValueError("image必须是PIL.Image.Image类型")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / "ip_adapter_outputs"
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定文件名
    if filename is None:
        filename = f"{prefix}{uuid.uuid4().hex}{suffix}"
    
    # 构建完整路径
    output_path = os.path.join(output_dir, filename)
    
    # 保存图像
    image.save(output_path)
    logger.info(f"图像已保存到: {output_path}")
    
    return output_path

def get_sample_templates(template_type="inpainting", n=3):
    """
    获取示例模板图像
    
    Args:
        template_type: 模板类型，"inpainting"或"depth"
        n: 返回的模板数量
        
    Returns:
        模板图像路径列表
    """
    # 确定模板目录
    if template_type == "inpainting":
        template_dir = config.INPAINTING_TEMPLATE_DIR
    elif template_type == "depth":
        template_dir = config.DEPTH_TEMPLATE_DIR
    else:
        raise ValueError(f"不支持的模板类型: {template_type}")
    
    # 获取目录中的所有图像文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    template_files = [
        f for f in os.listdir(template_dir)
        if f.lower().endswith(image_extensions) # 判断文件名小写是否以image_extensions结尾
    ]
    
    # 如果没有足够的模板，返回警告
    if len(template_files) < n:
        logger.warning(f"请求{n}个模板，但目录中只有{len(template_files)}个")
        
    # 返回前n个模板的路径
    return [os.path.join(template_dir, f) for f in template_files[:n]] 