#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
大语言模型聊天API路由模块
"""
import os
import logging
import traceback
import json
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from models.llm import LLMModel

from api.models import (
    ChatRequest, ChatResponse, ChatMessage
)

logger = logging.getLogger(__name__)

# 创建路由器，不指定前缀以便在主路由器中使用统一前缀
router = APIRouter()

# 全局LLM模型实例池
llm_models = {}

def get_llm_model(model_type: str = "deepseek") -> LLMModel:
    """
    获取或创建LLM模型实例
    
    Args:
        model_type: 模型类型，可选值为'deepseek'或'qwen'
        
    Returns:
        LLMModel实例
    """
    
    # 如果模型已存在，直接返回
    model_key = f"{model_type}"
    if model_key in llm_models:
        return llm_models[model_key]

    # 创建新的模型实例
    logger.info(f"初始化LLM模型: {model_type}")
    
    # 读取配置信息
    api_key = config.DEEP_SEEK_API_KEY if model_type == "deepseek" else config.QWEN_API_KEY

    model = LLMModel(
        model_type=model_type,
        api_key=api_key,
        device=config.DEVICE
    )
    
    llm_models[model_key] = model
    return model

@router.post("/chat/completion", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """
    聊天完成API
    
    处理用户对话请求，生成模型响应
    """
    try:
        logger.info(f"聊天请求开始处理，模型: {request.model}")
        
        # 获取模型实例
        model = get_llm_model(request.model)
        
        # 如果是流式输出，返回流式响应
        if request.stream:
            return await stream_chat_completion(request, model)
        
        # 生成响应
        response = await model.generate(
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            documents=request.documents,
            use_rag=request.use_rag,
            rag_query=request.rag_query,
            rag_top_k=request.rag_top_k,
            rag_filter=request.rag_filter,
            stream=False
        )
        
        return ChatResponse(
            message=response["message"],
            model=response["model"],
            usage=response["usage"],
            references=response.get("references")
        )
    
    except Exception as e:
        logger.error(f"聊天请求处理失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 返回错误响应
        error_message = ChatMessage(
            role="assistant",
            content=f"处理请求时发生错误: {str(e)}"
        )
        
        return ChatResponse(
            message=error_message,
            model=request.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )

async def stream_chat_completion(request: ChatRequest, model: LLMModel):
    """
    流式聊天完成
    
    Args:
        request: 聊天请求
        model: LLM模型实例
        
    Returns:
        流式事件响应
    """
    async def event_generator():
        try:
            # 生成流式响应
            response_queue = await model.generate(
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                documents=request.documents,
                use_rag=request.use_rag,
                rag_query=request.rag_query,
                rag_top_k=request.rag_top_k,
                rag_filter=request.rag_filter,
                stream=True
            )
            
            # 累积内容
            accumulated_content = ""
            
            # 处理队列中的响应
            while True:
                chunk = response_queue.get()
                
                # 检查是否为结束标记
                if chunk is None:
                    # 发送最终完成事件
                    completion = {
                        "message": {
                            "role": "assistant",
                            "content": accumulated_content
                        },
                        "model": request.model,
                        "usage": {
                            "prompt_tokens": 0,  # 无法在流式响应中获取确切的令牌使用情况
                            "completion_tokens": 0,
                            "total_tokens": 0
                        },
                        "finished": True
                    }
                    yield json.dumps(completion)
                    break
                
                # 累积内容
                accumulated_content += chunk
                
                # 发送事件
                event_data = {
                    "message": {
                        "role": "assistant",
                        "content": chunk
                    },
                    "model": request.model,
                    "finished": False
                }
                yield json.dumps(event_data)
                
        except Exception as e:
            logger.error(f"流式生成失败: {str(e)}")
            error_data = {
                "message": {
                    "role": "assistant",
                    "content": f"生成失败: {str(e)}"
                },
                "model": request.model,
                "error": str(e),
                "finished": True
            }
            yield json.dumps(error_data)
    
    return EventSourceResponse(event_generator())

@router.get("/chat/models")
async def list_models():
    """获取可用的聊天模型列表"""
    # 支持的模型列表
    models = [
        {
            "id": "deepseek",
            "name": "DeepSeek",
            "description": "DeepSeek 大语言模型",
            "capabilities": ["chat", "code", "reasoning"]
        },
        {
            "id": "qwen",
            "name": "通义千问",
            "description": "阿里云通义千问大语言模型",
            "capabilities": ["chat", "chinese", "reasoning"]
        }
    ]
    
    return {"models": models} 