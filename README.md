# IP-Adapter 图像处理应用

这是一个基于IP-Adapter技术的图像处理Web应用，支持多种图像生成和编辑功能，包括人脸嵌入、图像修复和深度控制等。

## 功能特性

- **人脸嵌入**: 将用户提供的人脸图像嵌入到模板图像中
- **图像修复(Inpainting)**: 智能填充图像的缺失部分或替换指定区域
- **深度控制(ControlNet)**: 通过深度图控制图像的3D结构和姿态
- **文本到图像(Text2Img)**: 根据文本描述生成图像
- **模板管理**: 支持自定义模板图像的上传和管理
- **RESTful API**: 提供完整的API接口，方便集成到其他应用

## 安装指南

### 环境要求

- Python 3.10+
- CUDA兼容的GPU (推荐 8GB+ 显存)
- 足够的磁盘空间用于存储模型 (约10GB)

### 使用Conda安装

```bash
# 克隆仓库
git clone <repository-url>
cd <repository-dir>

# 创建并激活conda环境
conda env create -f environment.yml
conda activate ip-adapter

# 下载必要的模型
bash hfd.sh stable-diffusion-v1-5/stable-diffusion-inpainting
bash hfd.sh h94/IP-Adapter
bash hfd.sh lllyasviel/control_v11f1p_sd15_depth
bash hfd.sh stable-diffusion-v1-5/stable-diffusion-v1-5
bash hfd.sh openai/clip-vit-large-patch14
bash hfd.sh stabilityai/sd-vae-ft-mse
```

## 使用说明

### 启动服务

```bash
# 使用默认GPU
uvicorn app:app --host 0.0.0.0 --port 8000

# 指定GPU
CUDA_VISIBLE_DEVICES=1 uvicorn app:app --host 0.0.0.0 --port 8000
```

### 关闭服务

```bash
# 关闭指定端口的服务
fuser -k 8000/tcp
```

### 访问Web界面和API文档

- Web界面: http://localhost:8000
- API文档: http://localhost:8000/docs

## API说明

本项目提供以下API端点:

- `/api/v1/inpainting`: 图像修复API
- `/api/v1/controlnet`: 基于深度图的图像生成API
- `/api/v1/text2img`: 文本到图像生成API
- `/api/v1/templates`: 模板管理API
- `/api/v1/face`: 人脸处理API

请参考API文档获取详细使用方法。

## 项目结构

```
├── api/                # API接口定义
│   ├── models.py       # API模型定义
│   └── routes/         # API路由
├── models/             # 模型实现
│   ├── base.py         # 基础模型类
│   ├── inpainting.py   # 图像修复模型
│   ├── controlnet.py   # ControlNet模型
│   └── text2img.py     # 文本到图像模型
├── processors/         # 图像处理器
│   ├── face_detector.py # 人脸检测
│   └── depth_estimator.py # 深度估计
├── utils/              # 工具函数
├── templates/          # 模板目录
├── uploads/            # 上传文件存储
├── results/            # 生成结果存储
├── config.py           # 配置文件
└── app.py              # 主应用入口
```

## 使用的模型

- [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
- [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)
- [stable-diffusion-v1-5/stable-diffusion-inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)
- [lllyasviel/control_v11f1p_sd15_depth](https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth)
- [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)

## IP-Adapter 工作原理

IP-Adapter 模型输入为图片，输出为图片的特征，从而嵌入到扩散模型中，实现对图片的修改和生成。可以将用户上传的人脸图像特征嵌入到模板图像中，生成保持人脸特征的新图像。


## InstantID 
- For higher similarity, increase the weight of controlnet_conditioning_scale (IdentityNet) and ip_adapter_scale (Adapter).若要提高相似度，请增加 controlnet_conditioning_scale (IdentityNet) 和 ip_adapter_scale (Adapter) 的权重。
- For over-saturation, decrease the ip_adapter_scale. If not work, decrease controlnet_conditioning_scale.
- For higher text control ability, decrease ip_adapter_scale.要提高文本控制能力，请减小 ip_adapter_scale。
- For specific styles, choose corresponding base model makes differences.如需了解具体款式，请选择相应的基本型号。
- We have not supported multi-person yet, only use the largest face as reference facial landmarks.我们还不支持多人，只使用最大的人脸作为参考面部地标。