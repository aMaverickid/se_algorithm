"""
文档管理API路由模块，负责文档的上传、查询和删除
"""
import os
import logging
import json
import shutil
import tempfile
import time
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, BackgroundTasks

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from utils.rag_utils import get_rag_processor
from api.models import (
    DocumentResponse, DocumentListResponse, DocumentUploadResponse,
    DocumentError
)

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter()

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    overwrite: bool = Form(False)
):
    """
    上传文档并添加到向量数据库
    
    文档将被处理并添加到RAG系统中
    """
    try:
        # 检查文件扩展名是否支持
        supported_extensions = [".txt", ".pdf", ".docx", ".doc", ".md", ".html", ".htm"]
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in supported_extensions:
            return DocumentUploadResponse(
                success=False,
                error=DocumentError(
                    message=f"不支持的文件类型: {file_ext}，支持的类型: {', '.join(supported_extensions)}"
                )
            )
        
        # 创建临时文件
        temp_file = Path(tempfile.gettempdir()) / f"upload_{int(time.time())}_{file.filename}"
        
        try:
            # 保存上传的文件
            with open(temp_file, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            # 准备元数据
            metadata = {
                "title": title or Path(file.filename).stem,
                "description": description or "",
                "tags": tags.split(",") if tags else [],
                "original_filename": file.filename,
                "upload_time": time.time()
            }
            
            # 获取RAG处理器
            rag_processor = get_rag_processor()
            
            # 添加文档到向量数据库
            # 注意: 为避免阻塞API响应，我们使用background_tasks进行处理
            doc_id = await process_document(rag_processor, temp_file, metadata)
            
            return DocumentUploadResponse(
                success=True,
                document_id=doc_id,
                message="文档上传成功并添加到向量数据库"
            )
            
        finally:
            # 确保临时文件被删除
            if temp_file.exists():
                temp_file.unlink()
    
    except Exception as e:
        logger.error(f"文档上传失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        return DocumentUploadResponse(
            success=False,
            error=DocumentError(
                message=f"处理文档时出错: {str(e)}"
            )
        )

async def process_document(rag_processor, file_path, metadata):
    """在后台处理文档"""
    try:
        # 添加文档到向量数据库
        doc_id = rag_processor.add_document(file_path, metadata)
        logger.info(f"文档处理完成: {doc_id}")
        return doc_id
    except Exception as e:
        logger.error(f"文档处理失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    tag: Optional[str] = Query(None, description="按标签筛选文档"),
    limit: int = Query(100, description="返回的最大文档数量")
):
    """
    获取文档列表
    
    返回所有已添加到RAG系统的文档
    """
    try:
        # 获取RAG处理器
        rag_processor = get_rag_processor()
        
        # 获取所有文档
        documents = rag_processor.list_documents()
        
        # 按标签过滤
        if tag:
            documents = [
                doc for doc in documents 
                if "tags" in doc.get("metadata", {}) and tag in doc.get("metadata", {}).get("tags", [])
            ]
        
        # 限制返回数量
        documents = documents[:limit]
        
        return DocumentListResponse(
            success=True,
            documents=documents,
            total=len(documents)
        )
    
    except Exception as e:
        logger.error(f"获取文档列表失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail=f"获取文档列表失败: {str(e)}"
        )

@router.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str):
    """
    获取文档详情
    
    返回指定文档的详细信息和所有文本块
    """
    try:
        # 获取RAG处理器
        rag_processor = get_rag_processor()
        
        # 获取文档的所有块
        chunks = rag_processor.get_document_chunks(doc_id)
        
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=f"文档不存在: {doc_id}"
            )
        
        # 从第一个块的元数据中提取文档信息
        metadata = chunks[0].get("metadata", {})
        
        return DocumentResponse(
            success=True,
            document_id=doc_id,
            metadata=metadata,
            chunks=chunks
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"获取文档详情失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail=f"获取文档详情失败: {str(e)}"
        )

@router.delete("/documents/{doc_id}", response_model=DocumentResponse)
async def delete_document(doc_id: str):
    """
    删除文档
    
    从向量数据库中删除指定文档及其所有块
    """
    try:
        # 获取RAG处理器
        rag_processor = get_rag_processor()
        
        # 删除文档
        success = rag_processor.delete_document(doc_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"文档不存在或删除失败: {doc_id}"
            )
        
        return DocumentResponse(
            success=True,
            document_id=doc_id,
            message=f"文档已成功删除: {doc_id}"
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail=f"删除文档失败: {str(e)}"
        )

@router.post("/documents/search", response_model=DocumentResponse)
async def search_documents(
    query: str = Form(...),
    limit: int = Form(5, description="返回的最大结果数量"),
    filter_tag: Optional[str] = Form(None, description="按标签筛选结果")
):
    """
    搜索文档
    
    在向量数据库中搜索与查询相关的文档块
    """
    try:
        # 获取RAG处理器
        rag_processor = get_rag_processor()
        
        # 构建过滤条件
        filter_expr = None
        if filter_tag:
            filter_expr = f"'metadata.tags' CONTAINS '{filter_tag}'"
        
        # 执行搜索
        results = rag_processor.search(
            query=query,
            limit=limit,
            filter_expr=filter_expr
        )
        
        return DocumentResponse(
            success=True,
            query=query,
            results=results,
            total=len(results)
        )
    
    except Exception as e:
        logger.error(f"搜索文档失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail=f"搜索文档失败: {str(e)}"
        ) 