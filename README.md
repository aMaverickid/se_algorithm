## models/cache 使用的模型

- [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
- [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)
- [stable-diffusion-v1-5/stable-diffusion-inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)

## 运行项目

```bash
CUDA_VISIBLE_DEVICES=1 uvicorn app:app --host 0.0.0.0 --port 8000
uvicorn app:app --host 0.0.0.0 --port 8000
```

关闭8000端口对应的程序
```
fuser -k 8000/tcp
```