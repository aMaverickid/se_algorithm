#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成示例模板图像，包括无脸模板和深度模板
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def create_faceless_template(output_path, width=512, height=768, style="simple"):
    """
    创建无脸模板图像
    
    Args:
        output_path: 输出路径
        width: 图像宽度
        height: 图像高度
        style: 模板样式，可选值："simple", "artistic", "portrait"
    
    Returns:
        保存的图像路径
    """
    # 创建画布
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 根据样式生成不同的无脸模板
    if style == "simple":
        # 简单人形轮廓
        # 绘制身体
        body_color = (200, 200, 200)
        draw.ellipse((width//2-100, height//4-100, width//2+100, height//4+100), fill=body_color)  # 头部
        draw.rectangle((width//2-75, height//4+50, width//2+75, height//2+150), fill=body_color)  # 躯干
        
        # 模糊面部区域
        face_mask = Image.new("L", (width, height), color=0)
        face_draw = ImageDraw.Draw(face_mask)
        face_draw.ellipse((width//2-60, height//4-60, width//2+60, height//4+40), fill=255)
        img = img.filter(ImageFilter.GaussianBlur(radius=10))
        
        # 添加背景细节
        for _ in range(50):
            x, y = random.randint(0, width), random.randint(0, height)
            size = random.randint(5, 20)
            color = (random.randint(200, 240), random.randint(200, 240), random.randint(200, 240))
            draw = ImageDraw.Draw(img)
            draw.ellipse((x, y, x+size, y+size), fill=color)
    
    elif style == "artistic":
        # 艺术风格
        # 绘制彩色背景
        for i in range(height):
            for j in range(width):
                r = int(200 + 55 * np.sin(i/50))
                g = int(200 + 55 * np.sin(j/50))
                b = int(200 + 55 * np.sin((i+j)/100))
                img.putpixel((j, i), (r, g, b))
        
        # 绘制人形轮廓
        draw.ellipse((width//2-80, height//5-80, width//2+80, height//5+80), fill=(150, 150, 150))  # 头部
        draw.rectangle((width//2-60, height//5+50, width//2+60, height//2+100), fill=(130, 130, 130))  # 躯干
        
        # 模糊面部
        face_area = img.crop((width//2-70, height//5-70, width//2+70, height//5+70))
        face_area = face_area.filter(ImageFilter.GaussianBlur(radius=15))
        img.paste(face_area, (width//2-70, height//5-70))
    
    elif style == "portrait":
        # 肖像风格
        # 绘制渐变背景
        for i in range(height):
            for j in range(width):
                r = int(100 + (i / height) * 100)
                g = int(100 + (i / height) * 100)
                b = int(150 + (i / height) * 100)
                img.putpixel((j, i), (r, g, b))
        
        # 绘制衣服区域
        draw.rectangle((0, height//3, width, height), fill=(50, 50, 120))
        
        # 绘制头部区域
        draw.ellipse((width//2-100, height//6-100, width//2+100, height//6+100), fill=(220, 180, 150))
        
        # 模糊面部
        face_area = img.crop((width//2-90, height//6-90, width//2+90, height//6+70))
        face_area = face_area.filter(ImageFilter.GaussianBlur(radius=20))
        img.paste(face_area, (width//2-90, height//6-90))
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    
    return output_path

def create_depth_template(output_path, width=512, height=768, pose_type="front"):
    """
    创建深度模板图像
    
    Args:
        output_path: 输出路径
        width: 图像宽度
        height: 图像高度
        pose_type: 姿势类型，可选值："front", "side", "angle"
    
    Returns:
        保存的图像路径
    """
    # 创建画布
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    
    # 根据姿势类型生成不同的深度图
    if pose_type == "front":
        # 前视图深度模板
        for i in range(height):
            for j in range(width):
                # 计算到图像中心的距离
                dx = (j - width//2) / (width//2)
                dy = (i - height//3) / (height//3)
                dist = np.sqrt(dx**2 + dy**2)
                
                # 头部
                if dist < 0.3:
                    depth = int(200 - dist * 200)
                # 身体
                elif i > height//3 and abs(dx) < 0.4:
                    depth = int(150 - abs(dx) * 100)
                else:
                    depth = 20
                
                img.putpixel((j, i), (depth, depth, depth))
    
    elif pose_type == "side":
        # 侧视图深度模板
        for i in range(height):
            for j in range(width):
                # 计算到参考点的距离
                dy = (i - height//3) / (height//3)
                dx = (j - width//3) / (width//3)
                
                # 头部
                if dx**2 + dy**2 < 0.3:
                    depth = int(220 - (dx**2 + dy**2) * 200)
                # 身体
                elif i > height//3 and j < width//2 and j > width//5:
                    depth = int(180 - abs(j - width//3) / (width//6) * 100)
                else:
                    depth = 30
                
                img.putpixel((j, i), (depth, depth, depth))
    
    elif pose_type == "angle":
        # 45度角视图深度模板
        for i in range(height):
            for j in range(width):
                # 计算到参考点的距离
                dy = (i - height//3) / (height//3)
                dx = (j - width*0.45) / (width//3)
                
                # 头部
                head_dist = np.sqrt(dx**2 + dy**2)
                if head_dist < 0.25:
                    depth = int(230 - head_dist * 200)
                # 身体
                elif i > height//3 and abs(dx) < 0.4:
                    body_dist = abs(dx) + (i - height//3) / height
                    depth = int(170 - body_dist * 120)
                else:
                    depth = 40
                
                # 为了让深度图更明显，增加一些颜色
                r = depth
                g = depth
                b = int(depth * 1.2) if depth < 200 else 255
                
                img.putpixel((j, i), (r, g, b))
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    
    return output_path

def generate_example_templates():
    """生成示例模板图像"""
    # 生成无脸模板
    inpainting_dir = config.INPAINTING_TEMPLATE_DIR
    os.makedirs(inpainting_dir, exist_ok=True)
    
    create_faceless_template(os.path.join(inpainting_dir, "template_simple.png"), style="simple")
    create_faceless_template(os.path.join(inpainting_dir, "template_artistic.png"), style="artistic")
    create_faceless_template(os.path.join(inpainting_dir, "template_portrait.png"), style="portrait")
    
    # 生成深度模板
    depth_dir = config.DEPTH_TEMPLATE_DIR
    os.makedirs(depth_dir, exist_ok=True)
    
    create_depth_template(os.path.join(depth_dir, "depth_front.png"), pose_type="front")
    create_depth_template(os.path.join(depth_dir, "depth_side.png"), pose_type="side")
    create_depth_template(os.path.join(depth_dir, "depth_angle.png"), pose_type="angle")
    
    print(f"生成了3个无脸模板到 {inpainting_dir}")
    print(f"生成了3个深度模板到 {depth_dir}")

if __name__ == "__main__":
    generate_example_templates() 