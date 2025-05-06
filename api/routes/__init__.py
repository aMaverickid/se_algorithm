#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API路由模块，将所有子路由整合到一个主路由器
"""
from fastapi import APIRouter
from api.routes.common import router as common_router
# from api.routes.inpainting import router as inpainting_router
# from api.routes.controlnet import router as controlnet_router
from api.routes.text2img import router as text2img_router
# from api.routes.face import router as face_router
# from api.routes.instantid import router as instantid_router

# 创建主路由器
router = APIRouter(prefix="/api", tags=["ip-adapter"])

# 包含所有子路由器
router.include_router(common_router)
# router.include_router(inpainting_router)
# router.include_router(controlnet_router)
router.include_router(text2img_router)
# router.include_router(face_router)
# router.include_router(instantid_router)

# 导出主路由器
__all__ = ["router"] 