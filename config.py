import os
from pathlib import Path

# 随机种子
SEED = 42

# 设备
DEVICE = "cuda"


# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent

# 模型缓存目录
MODEL_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", str(ROOT_DIR / "models" / "cache")))

# 结果目录
RESULTS_DIR = ROOT_DIR / "results"
TEST_DIR = ROOT_DIR / "test"

# 模板目录
TEMPLATE_DIR = ROOT_DIR / "templates"
INPAINTING_TEMPLATE_DIR = TEMPLATE_DIR / "inpainting"
DEPTH_TEMPLATE_DIR = TEMPLATE_DIR / "depth"

# 确保目录存在
INPAINTING_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
DEPTH_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

# 模型配置
CLIP_MODEL_PATH = MODEL_CACHE_DIR / "clip-vit-large-patch14"
STABLE_DIFFUSION_MODEL_PATH = MODEL_CACHE_DIR / "stable-diffusion-inpainting"
IP_ADAPTER_MODEL_PATH = MODEL_CACHE_DIR / "IP-Adapter"
CONTROLNET_DEPTH_MODEL_PATH = MODEL_CACHE_DIR / "control_v11f1p_sd15_depth"
INSTANTID_MODEL_PATH = MODEL_CACHE_DIR / "InstantID"
INSTANTID_CONTROLNET_MODEL_PATH = MODEL_CACHE_DIR / "InstantID" / "ControlNetModel"
INSTANTID_INSIGHT_PATH = MODEL_CACHE_DIR / "insightface"
INSTANTID_SD_MODEL_PATH = MODEL_CACHE_DIR / "stable-diffusion-xl-base-1.0"

TEXT2IMG_STABLE_DIFFUSION_MODEL_PATH = MODEL_CACHE_DIR / "stable-diffusion-v1-5"
VAE_MODEL_PATH = MODEL_CACHE_DIR / "sd-vae-ft-mse"
FACE_RESOLUTION = (256, 256)

# 大语言模型配置
DEEP_SEEK_API_KEY = "sk-6716e5b5b210471d9e3f037e7afde64e"
QWEN_API_KEY = "sk-5f135400cc0e4d9fab8f5e6d8b03a285"

# API配置
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "IP-Adapter API"
API_DESCRIPTION = "基于IP-Adapter模型的图像处理API"
API_VERSION = "0.1.0"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB

# 日志配置
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"