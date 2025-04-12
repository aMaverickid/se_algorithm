#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模板相关API路由模块
"""
import os
import logging
from fastapi import APIRouter, HTTPException

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from utils.image_utils import get_sample_templates
from api.models import TemplatesResponse

logger = logging.getLogger(__name__)

# 创建路由器，不指定前缀以便在主路由器中使用统一前缀
router = APIRouter()

@router.get("/templates/{template_type}", response_model=TemplatesResponse)
async def get_templates(template_type: str):
    """获取模板列表"""
    if template_type not in ["inpainting", "depth"]:
        raise HTTPException(status_code=400, detail=f"不支持的模板类型: {template_type}")
    
    # 获取模板路径列表
    template_paths = get_sample_templates(template_type)
    templates = []
    
    for i, path in enumerate(template_paths):
        template_id = os.path.basename(path) # 得到路径中的最后一个组件 即文件名
        name = os.path.splitext(template_id)[0] # 去掉扩展名 splitext 返回 (filename, extension)
        
        # 获取预览URL（这里简单地使用模板ID作为静态资源URL）
        preview_url = f"/static/templates/{template_type}/{template_id}"
        
        templates.append({
            "id": template_id,
            "name": name,
            "preview_url": preview_url,
            "description": f"{template_type.capitalize()} 模板 {i+1}" # capitalize 首字母大写
        })
    
    return {
        "templates": templates,
        "type": template_type
    } 