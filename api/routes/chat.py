"""
聊天API路由处理
"""
import logging
import json
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Request, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
import config
from models.deepseek import DeepSeekModel

logger = logging.getLogger(__name__)

# 创建路由
router = APIRouter(prefix="/chat", tags=["chat"])

# 全局DeepSeek模型实例
deepseek_model = DeepSeekModel()

@router.post("/completions")
async def chat_completions(request: Request):
    """
    处理聊天请求并获得AI响应
    
    请求格式：
    {
        "messages": "JSON序列化的消息数组，如 [{\"role\": \"user\", \"content\": \"你好\"}]",
        "temperature": 0.7,
        "max_tokens": 1024,
        "documents": ["文档内容1", "文档内容2"],
        "use_rag": false,
        "rag_query": "用于检索的查询",
        "rag_top_k": 5,
        "rag_filter": "过滤条件"
    }
    """
    try:
        # 解析请求体
        body = await request.json()
        messages = body.get("messages", "[]")
        temperature = float(body.get("temperature", 0.7))
        max_tokens = int(body.get("max_tokens", 1024))
        documents = body.get("documents", None)
        use_rag = body.get("use_rag", False)
        rag_query = body.get("rag_query", None)
        rag_top_k = int(body.get("rag_top_k", 5))
        rag_filter = body.get("rag_filter", None)
        
        # 检查messages是否是有效的JSON字符串
        if isinstance(messages, list):
            messages = json.dumps(messages)
            
        # 调用DeepSeek模型生成响应
        response = await deepseek_model.generate_response(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            documents=documents,
            use_rag=use_rag,
            rag_query=rag_query,
            rag_top_k=rag_top_k,
            rag_filter=rag_filter
        )
        
        # 检查是否为流式响应
        if response.get("message") == "__STREAM__":
            stream = response.get("stream")
            
            async def response_stream():
                """返回流式响应"""
                try:
                    async for line in stream.content:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            try:
                                data = json.loads(line_str[6:])
                                if data.get('choices') and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                pass
                finally:
                    # 关闭流
                    if not stream.closed:
                        stream.close()
                        
            return StreamingResponse(
                response_stream(),
                media_type="text/plain"
            )
            
        # 非流式响应
        return response
    except Exception as e:
        logger.exception(f"聊天接口错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")

@router.post("/document/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(None),
    source: str = Form(None),
    description: str = Form(None),
    tags: str = Form(None)
):
    """
    上传文档到RAG系统
    """
    try:
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # 准备元数据
        metadata = {
            "title": title or file.filename,
            "source": source or "用户上传",
            "description": description or "",
            "tags": tags.split(",") if tags else [],
            "filename": file.filename,
            "content_type": file.content_type
        }
        
        # 添加文档到RAG系统
        result = await deepseek_model.rag_manager.add_document(
            content=text_content,
            metadata=metadata
        )
        
        return result
    except Exception as e:
        logger.exception(f"上传文档错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传文档失败: {str(e)}")
        
@router.get("/document/list")
async def list_documents(
    limit: int = 100,
    offset: int = 0
):
    """
    获取文档列表
    """
    try:
        result = await deepseek_model.rag_manager.get_documents(
            limit=limit,
            offset=offset
        )
        return result
    except Exception as e:
        logger.exception(f"获取文档列表错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")
        
@router.get("/document/{doc_id}")
async def get_document(doc_id: str):
    """
    获取特定文档
    """
    try:
        result = await deepseek_model.rag_manager.get_document(doc_id)
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", {}).get("message", "文档不存在"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"获取文档错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取文档失败: {str(e)}")
        
@router.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    """
    删除特定文档
    """
    try:
        result = await deepseek_model.rag_manager.delete_document(doc_id)
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", {}).get("message", "删除文档失败"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"删除文档错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")
        
@router.post("/document/search")
async def search_documents(request: Request):
    """
    搜索文档
    
    请求格式：
    {
        "query": "搜索关键词",
        "top_k": 5,
        "filter": "过滤条件"
    }
    """
    try:
        body = await request.json()
        query = body.get("query", "")
        top_k = int(body.get("top_k", 5))
        filter_condition = body.get("filter", None)
        
        if not query:
            raise HTTPException(status_code=400, detail="搜索查询不能为空")
            
        result = await deepseek_model.rag_manager.search(
            query=query,
            top_k=top_k,
            filter_condition=filter_condition
        )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"搜索文档错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索文档失败: {str(e)}") 