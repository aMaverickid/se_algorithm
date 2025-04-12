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
    face_image: str = Field(..., description="Base64编码的人脸图像")
    template_id: Optional[str] = Field(None, description="模板ID，如果为None则使用template_image")
    template_image: Optional[str] = Field(None, description="Base64编码的模板图像，如果template_id不为None则忽略")
    mask_image: Optional[str] = Field(None, description="Base64编码的掩码图像，如果为None则自动生成")
    strength: float = Field(0.85, description="改变强度，0.0-1.0")
    guidance_scale: float = Field(7.5, description="分类器自由指导比例")
    num_images: int = Field(1, description="生成的图像数量", ge=1, le=4) # range 1-4
    seed: Optional[int] = Field(None, description="随机种子")
    positive_prompt: Optional[str] = Field("masterpiece, best quality, high quality", description="正面提示词")
    negative_prompt: Optional[str] = Field("lowres, bad anatomy, bad hands, cropped, worst quality", description="负面提示词")
    auto_generate_template: bool = Field(False, description="如果为True且未提供template_id/template_image，则自动生成无脸模板")
    template_method: str = Field("auto", description="自动生成无脸模板的方法，可选值：'auto', 'blur', 'mask'")
    template_strength: float = Field(0.8, description="自动生成无脸模板的强度，0.0-1.0")
    auto_generate_mask: bool = Field(False, description="如果为True且未提供mask_image，则自动生成掩码")
    mask_type: str = Field("face", description="自动生成掩码的类型，可选值：'face'(人脸区域), 'template'(无脸区域)")
    mask_padding_ratio: float = Field(0.1, description="自动生成人脸掩码时的边界框扩展比例")

class InpaintingResponse(BaseModel):
    """Inpainting响应数据模型"""
    images: List[str] = Field(..., description="Base64编码的生成图像列表")
    parameters: Dict[str, Any] = Field(..., description="生成参数")

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


# 无脸模板生成请求/响应模型
class TemplateGenerationRequest(BaseModel):
    """无脸模板生成请求数据模型"""
    face_image: Optional[str] = Field(None, description="Base64编码的人脸图像，如不提供则使用face_id")
    face_id: Optional[str] = Field(None, description="已上传的人脸图像ID")
    method: str = Field("auto", description="生成方法，可选值：'auto', 'blur', 'mask'")
    strength: float = Field(0.8, description="处理强度，0.0-1.0")

class TemplateGenerationResponse(BaseModel):
    """无脸模板生成响应数据模型"""
    success: bool = Field(..., description="是否成功生成")
    template_image: str = Field(..., description="Base64编码的无脸模板图像")
    template_id: Optional[str] = Field(None, description="生成的模板ID，如果已保存")
    message: Optional[str] = Field(None, description="附加信息")

# 掩码生成请求/响应模型
class MaskGenerationRequest(BaseModel):
    """掩码生成请求数据模型"""
    image: Optional[str] = Field(None, description="Base64编码的图像，如不提供则使用image_id")
    image_id: Optional[str] = Field(None, description="已上传的图像ID")
    mask_type: str = Field("face", description="掩码类型，可选值：'face'(人脸区域), 'template'(无脸区域)")
    padding_ratio: float = Field(0.1, description="人脸边界框的扩展比例，仅对mask_type='face'有效")
    feather_amount: int = Field(10, description="边缘羽化程度")

class MaskGenerationResponse(BaseModel):
    """掩码生成响应数据模型"""
    success: bool = Field(..., description="是否成功生成")
    mask_image: str = Field(..., description="Base64编码的掩码图像")
    mask_id: Optional[str] = Field(None, description="生成的掩码ID，如果已保存")
    message: Optional[str] = Field(None, description="附加信息") 