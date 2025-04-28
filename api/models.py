"""
API数据模型定义
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ImageData(BaseModel):
    """图像数据，支持Base64编码"""
    data: str = Field(..., description="Base64编码的图像数据")

class TemplateInfo(BaseModel):
    """模板信息"""
    id: str = Field(..., description="模板ID")
    name: str = Field(..., description="模板名称")
    preview_url: str = Field(..., description="预览图URL")
    description: Optional[str] = Field(None, description="模板描述")

class ModelInfo(BaseModel):
    """模型信息"""
    name: str = Field(..., description="模型名称")
    type: str = Field(..., description="模型类型")
    version: str = Field(..., description="模型版本")
    description: Optional[str] = Field(None, description="模型描述")

# Inpainting请求/响应模型
class InpaintingRequest(BaseModel):
    """Inpainting请求数据模型"""
    face: str = Field(..., description="Base64编码的人脸图像")

    template_id: Optional[int] = Field(1, description="使用指定的模板ID")
    template: Optional[str] = Field(None, description="Base64编码的模板图像")
    mask: Optional[str] = Field(None, description="Base64编码的掩码图像")

    # 生成控制参数
    ip_adapter_scale: Optional[float] = Field(None, description="IP-Adapter缩放因子")
    strength: Optional[float] = Field(0.85, description="人脸与模板的融合强度，范围0-1，值越大融合越强")
    prompt: Optional[str] = Field("", description="提示词")
    num_inference_steps: Optional[int] = Field(50, description="推理步数，范围20-100，越高越精细，但生成时间越长")
    guidance_scale: Optional[float] = Field(7.5, description="文本提示相关性控制，范围1-20，越高越遵循文本提示，低则更自由")

class InpaintingResponse(BaseModel):
    """Inpainting响应数据模型"""
    success: bool = Field(..., description="是否成功")
    result: Optional[str] = Field(None, description="Base64编码的生成图像")
    message: Optional[str] = Field(None, description="附加信息")

# ControlNet请求/响应模型
class ControlNetRequest(BaseModel):
    """ControlNet请求数据模型"""
    face_image: str = Field(..., description="Base64编码的人脸图像")
    depth_template_id: Optional[str] = Field(None, description="深度模板ID，如果为None则使用depth_image")
    depth_image: Optional[str] = Field(None, description="Base64编码的深度图像，如果depth_template_id不为None或都为None则自动生成")
    controlnet_conditioning_scale: float = Field(1.0, description="ControlNet条件权重")
    guidance_scale: float = Field(7.5, description="分类器自由指导比例")
    num_images: int = Field(1, description="生成的图像数量", ge=1, le=4)
    seed: Optional[int] = Field(None, description="随机种子")
    positive_prompt: Optional[str] = Field("masterpiece, best quality, high quality, photorealistic", description="正面提示词")
    negative_prompt: Optional[str] = Field("lowres, bad anatomy, bad hands, cropped, worst quality", description="负面提示词")

class ControlNetResponse(BaseModel):
    """ControlNet响应数据模型"""
    images: List[str] = Field(..., description="Base64编码的生成图像列表")
    depth_image: Optional[str] = Field(None, description="Base64编码的深度图像")
    parameters: Dict[str, Any] = Field(..., description="生成参数")

# Text2Img请求/响应模型
class Text2ImgRequest(BaseModel):
    """Text2Img请求数据模型"""
    face_image: str = Field(..., description="Base64编码的人脸图像")
    prompt: str = Field(..., description="文本提示词")
    width: int = Field(768, description="输出宽度")
    height: int = Field(768, description="输出高度")
    guidance_scale: float = Field(7.5, description="分类器自由指导比例")
    num_images: int = Field(1, description="生成的图像数量", ge=1, le=4)
    seed: Optional[int] = Field(None, description="随机种子")
    negative_prompt: Optional[str] = Field("lowres, bad anatomy, bad hands, cropped, worst quality", description="负面提示词")

class Text2ImgResponse(BaseModel):
    """Text2Img响应数据模型"""
    images: List[str] = Field(..., description="Base64编码的生成图像列表")
    parameters: Dict[str, Any] = Field(..., description="生成参数")

# 其他API响应模型
class ModelInfoResponse(BaseModel):
    """模型信息响应"""
    models: List[ModelInfo] = Field(..., description="可用模型列表")

class TemplatesResponse(BaseModel):
    """模板信息响应"""
    templates: List[TemplateInfo] = Field(..., description="可用模板列表")
    type: str = Field(..., description="模板类型")

class MediapipeMaskRequest(BaseModel):
    """使用Mediapipe生成掩码请求数据模型"""
    face: str = Field(..., description="Base64编码的人脸")
    blur_radius: Optional[int] = Field(2, description="掩码边缘模糊半径，值越大边缘越平滑")
    remove_holes: Optional[bool] = Field(True, description="是否移除掩码中的小孔洞")
    detailed_edges: Optional[bool] = Field(False, description="是否生成更细致的边缘")

class MediapipeMaskResponse(BaseModel):
    """使用Mediapipe生成掩码响应数据模型"""
    success: bool = Field(..., description="是否成功生成")
    mask: Optional[str] = Field(None, description="Base64编码的掩码图像")
    message: Optional[str] = Field(None, description="附加信息")


class MediapipeTemplateRequest(BaseModel):
    """使用Mediapipe生成无脸模板请求数据模型"""
    face: str = Field(..., description="Base64编码的人脸")
    blur_radius: Optional[int] = Field(2, description="模板边缘模糊半径，值越大边缘越平滑")
    remove_holes: Optional[bool] = Field(True, description="是否移除模板中的小孔洞")
    detailed_edges: Optional[bool] = Field(False, description="是否生成更细致的边缘")
    mask_type: Optional[str] = Field("inpainting", description="模板类型，可选值：'transparent'(透明背景), 'white'(白色背景), 'black'(黑色背景), 'red'(红色背景), 'edge_preserved'(保留边缘轮廓), 'inpainting'(专为inpainting优化的掩码)")
    edge_blur_radius: Optional[int] = Field(0, description="边缘额外模糊半径，用于为inpainting创建更平滑的过渡")

class MediapipeTemplateResponse(BaseModel):
    """使用Mediapipe生成无脸模板响应数据模型"""
    success: bool = Field(..., description="是否成功生成")
    template: Optional[str] = Field(None, description="Base64编码的无脸模板图像")
    message: Optional[str] = Field(None, description="附加信息")