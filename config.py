import os
from pathlib import Path

# 随机种子
SEED = 42

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
INSTANTID_MODEL_PATH = MODEL_CACHE_DIR / "stable-diffusion-xl-base-1.0"
INSIGHTFACE_MODEL_PATH = MODEL_CACHE_DIR / "insightface"

# 大语言模型配置
LLM_MODEL_PATH = MODEL_CACHE_DIR / "llm"
DEEPSEEK_MODEL_PATH = LLM_MODEL_PATH / "deepseek-coder-6.7b-instruct"
QWEN_MODEL_PATH = LLM_MODEL_PATH / "Qwen-7B-Chat"

DEEP_SEEK_API_KEY = "sk-5f135400cc0e4d9fab8f5e6d8b03a285"
QWEN_API_KEY = "sk-5f135400cc0e4d9fab8f5e6d8b03a285"

# 确保LLM模型目录存在
LLM_MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 推理参数
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 50
NUM_IMAGES_PER_PROMPT = 1
FACE_RESOLUTION = 512
OUTPUT_RESOLUTION = 768
INPAINTING_STRENGTH = 1.0
DEVICE = "cuda"

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

# 上传目录配置
UPLOAD_DIR = ROOT_DIR / "uploads"
FACE_UPLOAD_DIR = UPLOAD_DIR / "faces"
TEMPLATE_UPLOAD_DIR = UPLOAD_DIR / "templates"
MASK_UPLOAD_DIR = UPLOAD_DIR / "masks"

# 确保上传目录存在
UPLOAD_DIR.mkdir(exist_ok=True)
FACE_UPLOAD_DIR.mkdir(exist_ok=True)
TEMPLATE_UPLOAD_DIR.mkdir(exist_ok=True)
MASK_UPLOAD_DIR.mkdir(exist_ok=True)

# InstantID配置
INSTANTID_STYLE_PRESETS = [
    "anime", "cartoon", "digital_art", "fantasy", 
    "oil_painting", "watercolor", "pop_art", 
    "cyberpunk", "vintage", "comic"
]

# 大语言模型配置
LLM_DEFAULT_MODEL = "deepseek"
LLM_MAX_TOKENS = 2048
LLM_DEFAULT_TEMPERATURE = 0.7

# 文档目录配置
DOCS_DIR = ROOT_DIR / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# RAG配置
RAG_VECTOR_DB_DIR = DOCS_DIR / "vectordb"
RAG_VECTOR_DB_DIR.mkdir(exist_ok=True)
RAG_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 200
RAG_DEFAULT_TOP_K = 5 