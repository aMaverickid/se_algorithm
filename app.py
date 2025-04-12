"""
IP-Adapter API服务主应用
"""
import os
import logging
import uuid
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent))
import config
from api import router

# 设置日志
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


# 初始化回调
async def lifespan(app: FastAPI):
    """应用启动时执行的操作"""
    logger.info(f"服务启动: {config.API_TITLE} v{config.API_VERSION}")
    
    # 确保模板目录存在
    os.makedirs(config.INPAINTING_TEMPLATE_DIR, exist_ok=True)
    os.makedirs(config.DEPTH_TEMPLATE_DIR, exist_ok=True)
    
    # 记录设备信息
    import torch
    logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA内存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")

    # 上面是 程序启动时执行的逻辑
    yield  
    # 下面是 程序关闭时执行的逻辑

    logger.info("服务关闭")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 创建应用
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置为特定域名
    allow_credentials=True, # 允许跨域请求携带凭证
    allow_methods=["*"], # 允许所有请求方法
    allow_headers=["*"], # 允许所有请求头
)

# 设置请求ID中间件 处理每个HTTP请求前都会执行 add_request_id 函数
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """为每个请求添加唯一ID，方便跟踪"""
    request_id = str(uuid.uuid4()) # 生成唯一ID
    logger.info(f"Request {request_id}: {request.method} {request.url.path}")
    
    # 调用下一个中间件或路由处理函数
    response = await call_next(request)
    
    # 在响应头中添加请求ID
    response.headers["X-Request-ID"] = request_id
    return response

# 添加API路由
app.include_router(router)

# 挂载静态文件目录
# 相当于建立一个虚拟路径/static/templates，指向实际的templates目录
templates_dir = os.path.join(config.ROOT_DIR, "templates")
if os.path.exists(templates_dir):
    app.mount("/static/templates", StaticFiles(directory=templates_dir), name="templates")

@app.get("/")
async def root():
    """根路径处理"""
    return {
        "service": config.API_TITLE,
        "version": config.API_VERSION,
        "description": config.API_DESCRIPTION,
        "docs_url": "/docs",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", # 模块名称:定义的app实例
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True, # 修改代码后自动重启
    ) 