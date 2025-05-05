"""
检索增强生成(RAG)相关工具
"""
import logging
import os
import json
import uuid
import shutil
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import aiofiles
from pathlib import Path
import config

# 尝试导入向量数据库相关库
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    
logger = logging.getLogger(__name__)

# 文档存储目录
DOCUMENTS_DIR = Path(config.ROOT_DIR) / "documents"
DOCUMENTS_DIR.mkdir(exist_ok=True)

# 向量索引目录 
VECTOR_INDEX_DIR = DOCUMENTS_DIR / "vector_index"
VECTOR_INDEX_DIR.mkdir(exist_ok=True)

# 文档元数据文件
METADATA_FILE = DOCUMENTS_DIR / "metadata.json"

class RAGManager:
    """检索增强生成管理器"""
    
    def __init__(self):
        """初始化RAG管理器"""
        self.documents_dir = DOCUMENTS_DIR
        self.metadata_file = METADATA_FILE
        self.vector_index_dir = VECTOR_INDEX_DIR
        
        # 加载文档元数据
        self.metadata = self._load_metadata()
        
        # 初始化向量模型和索引
        self.model = None
        self.index = None
        self.doc_ids = []
        
        if VECTOR_SEARCH_AVAILABLE:
            asyncio.create_task(self._initialize_vector_search())
            logger.info("正在异步初始化向量搜索...")
        else:
            logger.warning("向量搜索不可用，请安装faiss-cpu和sentence-transformers库")
            
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """加载文档元数据"""
        if not self.metadata_file.exists():
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
            return {}
            
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载元数据文件出错: {str(e)}")
            return {}
            
    async def _save_metadata(self):
        """保存文档元数据"""
        try:
            async with aiofiles.open(self.metadata_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.metadata, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error(f"保存元数据文件出错: {str(e)}")
            
    async def _initialize_vector_search(self):
        """初始化向量搜索"""
        if not VECTOR_SEARCH_AVAILABLE:
            return
            
        try:
            # 加载sentence transformer模型
            self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
            logger.info("已加载文本嵌入模型")
            
            # 检查是否存在索引文件
            index_file = self.vector_index_dir / "faiss_index.bin"
            doc_ids_file = self.vector_index_dir / "doc_ids.json"
            
            if index_file.exists() and doc_ids_file.exists():
                # 加载现有索引
                self.index = faiss.read_index(str(index_file))
                
                async with aiofiles.open(doc_ids_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    self.doc_ids = json.loads(content)
                    
                logger.info(f"已加载向量索引，包含 {len(self.doc_ids)} 个文档块")
            else:
                # 创建新索引
                await self._rebuild_index()
                
        except Exception as e:
            logger.exception(f"初始化向量搜索出错: {str(e)}")
            
    async def _rebuild_index(self):
        """重建向量索引"""
        if not VECTOR_SEARCH_AVAILABLE or not self.model:
            return
            
        try:
            all_chunks = []
            all_ids = []
            
            # 收集所有文档块
            for doc_id, info in self.metadata.items():
                doc_file = self.documents_dir / f"{doc_id}.json"
                if not doc_file.exists():
                    continue
                    
                async with aiofiles.open(doc_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    doc_data = json.loads(content)
                    
                for chunk in doc_data.get("chunks", []):
                    all_chunks.append(chunk["content"])
                    all_ids.append(f"{doc_id}:{chunk['chunk_id']}")
                    
            if not all_chunks:
                logger.info("没有文档块可索引")
                return
                
            # 计算嵌入
            embeddings = self.model.encode(all_chunks)
            vector_dim = embeddings.shape[1]
            
            # 创建和填充FAISS索引
            self.index = faiss.IndexFlatIP(vector_dim)  # 内积相似度(点积)
            self.index = faiss.IndexIDMap(self.index)   # 映射到文档ID
            
            # 添加向量到索引
            faiss.normalize_L2(embeddings)  # 归一化向量
            self.index.add_with_ids(embeddings, np.arange(len(all_ids)))
            self.doc_ids = all_ids
            
            # 保存索引和ID映射
            faiss.write_index(self.index, str(self.vector_index_dir / "faiss_index.bin"))
            
            async with aiofiles.open(self.vector_index_dir / "doc_ids.json", 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.doc_ids))
                
            logger.info(f"已重建向量索引，包含 {len(self.doc_ids)} 个文档块")
            
        except Exception as e:
            logger.exception(f"重建索引出错: {str(e)}")
            
    async def add_document(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        添加新文档
        
        Args:
            content: 文档内容
            metadata: 文档元数据
            
        Returns:
            包含文档ID和状态的字典
        """
        try:
            doc_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # 准备元数据
            metadata = metadata or {}
            metadata.update({
                "created_at": timestamp,
                "updated_at": timestamp,
                "size": len(content),
            })
            
            # 分块处理文档
            chunks = self._chunk_document(content)
            
            # 创建文档数据
            doc_data = {
                "doc_id": doc_id,
                "metadata": metadata,
                "chunks": [
                    {"chunk_id": i, "content": chunk}
                    for i, chunk in enumerate(chunks)
                ]
            }
            
            # 保存文档
            doc_file = self.documents_dir / f"{doc_id}.json"
            async with aiofiles.open(doc_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(doc_data, ensure_ascii=False, indent=2))
                
            # 更新元数据
            self.metadata[doc_id] = {
                "doc_id": doc_id,
                "metadata": metadata,
                "chunk_count": len(chunks)
            }
            await self._save_metadata()
            
            # 异步更新索引
            if VECTOR_SEARCH_AVAILABLE and self.model and self.index:
                asyncio.create_task(self._update_document_vectors(doc_data))
                
            return {
                "success": True,
                "document_id": doc_id,
                "message": "文档添加成功"
            }
            
        except Exception as e:
            logger.exception(f"添加文档出错: {str(e)}")
            return {
                "success": False,
                "error": {"message": f"添加文档失败: {str(e)}"}
            }
            
    async def _update_document_vectors(self, doc_data: Dict[str, Any]):
        """更新文档向量"""
        if not VECTOR_SEARCH_AVAILABLE or not self.model or not self.index:
            return
            
        try:
            doc_id = doc_data["doc_id"]
            chunks = [chunk["content"] for chunk in doc_data["chunks"]]
            chunk_ids = [f"{doc_id}:{chunk['chunk_id']}" for chunk in doc_data["chunks"]]
            
            # 计算嵌入
            embeddings = self.model.encode(chunks)
            faiss.normalize_L2(embeddings)  # 归一化向量
            
            # 添加到索引
            self.index.add_with_ids(embeddings, np.arange(len(self.doc_ids), len(self.doc_ids) + len(chunks)))
            self.doc_ids.extend(chunk_ids)
            
            # 保存更新后的索引和ID映射
            faiss.write_index(self.index, str(self.vector_index_dir / "faiss_index.bin"))
            
            async with aiofiles.open(self.vector_index_dir / "doc_ids.json", 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.doc_ids))
                
            logger.info(f"已更新文档 {doc_id} 的向量索引")
            
        except Exception as e:
            logger.exception(f"更新文档向量出错: {str(e)}")
            
    def _chunk_document(self, content: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        将文档分成多个块
        
        Args:
            content: 文档内容
            chunk_size: 每个块的目标大小(字符数)
            overlap: 块之间的重叠大小
            
        Returns:
            文档块列表
        """
        if not content:
            return []
            
        # 按段落拆分
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            # 如果段落本身超过块大小，则分割
            if len(para) > chunk_size:
                # 如果当前块不为空，先完成当前块
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    # 保留最后一段用于重叠
                    if overlap > 0 and len(current_chunk) > 0:
                        current_chunk = [current_chunk[-1]]
                        current_size = len(current_chunk[-1])
                    else:
                        current_chunk = []
                        current_size = 0
                        
                # 分割大段落
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i:i + chunk_size])
            else:
                # 添加段落到当前块
                if current_size + len(para) > chunk_size:
                    # 当前块已满，保存并开始新块
                    chunks.append('\n'.join(current_chunk))
                    
                    # 重叠处理
                    if overlap > 0 and len(current_chunk) > 0:
                        # 保留最后一段用于重叠
                        current_chunk = [current_chunk[-1]]
                        current_size = len(current_chunk[-1])
                    else:
                        current_chunk = []
                        current_size = 0
                        
                current_chunk.append(para)
                current_size += len(para)
                
        # 处理最后一个块
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks
        
    async def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """
        删除文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            操作结果字典
        """
        try:
            if doc_id not in self.metadata:
                return {
                    "success": False,
                    "error": {"message": f"文档不存在: {doc_id}"}
                }
                
            # 删除文档文件
            doc_file = self.documents_dir / f"{doc_id}.json"
            if doc_file.exists():
                os.remove(doc_file)
                
            # 从元数据中删除
            del self.metadata[doc_id]
            await self._save_metadata()
            
            # 标记需要重建索引
            if VECTOR_SEARCH_AVAILABLE and self.model and self.index:
                asyncio.create_task(self._rebuild_index())
                
            return {
                "success": True,
                "message": f"文档 {doc_id} 已删除"
            }
            
        except Exception as e:
            logger.exception(f"删除文档出错: {str(e)}")
            return {
                "success": False,
                "error": {"message": f"删除文档失败: {str(e)}"}
            }
            
    async def search(self, 
                   query: str, 
                   top_k: int = 5, 
                   filter_condition: Optional[str] = None) -> Dict[str, Any]:
        """
        检索文档
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            filter_condition: 过滤条件
            
        Returns:
            检索结果字典
        """
        if not query:
            return {"success": True, "results": [], "query": query}
            
        try:
            results = []
            
            # 使用向量搜索
            if VECTOR_SEARCH_AVAILABLE and self.model and self.index and len(self.doc_ids) > 0:
                # 对查询进行编码
                query_embedding = self.model.encode([query])[0]
                query_embedding = query_embedding / np.linalg.norm(query_embedding)  # 归一化
                query_embedding = query_embedding.reshape(1, -1)
                
                # 搜索最相似的向量
                k = min(top_k * 2, len(self.doc_ids))  # 获取更多候选，以便后面过滤
                scores, indices = self.index.search(query_embedding, k)
                
                # 处理结果
                scored_results = []
                seen_docs = set()  # 用于去重
                
                for i, idx in enumerate(indices[0]):
                    if idx < 0 or idx >= len(self.doc_ids):
                        continue
                        
                    chunk_ref = self.doc_ids[idx]
                    doc_id, chunk_id = chunk_ref.split(":")
                    
                    # 如果文档ID不在元数据中，则跳过
                    if doc_id not in self.metadata:
                        continue
                        
                    # 应用过滤条件
                    if filter_condition:
                        doc_metadata = self.metadata[doc_id]["metadata"]
                        # 简单字符串匹配过滤，实际实现可能需要更复杂的逻辑
                        matched = False
                        for k, v in doc_metadata.items():
                            if filter_condition in str(v):
                                matched = True
                                break
                        if not matched:
                            continue
                            
                    # 加载文档
                    doc_file = self.documents_dir / f"{doc_id}.json"
                    if not doc_file.exists():
                        continue
                        
                    async with aiofiles.open(doc_file, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        doc_data = json.loads(content)
                        
                    # 查找块
                    chunk = None
                    for c in doc_data["chunks"]:
                        if str(c["chunk_id"]) == chunk_id:
                            chunk = c
                            break
                            
                    if not chunk:
                        continue
                        
                    score = float(scores[0][i])
                    
                    # 添加到结果
                    scored_results.append({
                        "score": score,
                        "content": chunk["content"],
                        "metadata": {
                            "doc_id": doc_id,
                            "chunk_id": chunk_id,
                            "source": doc_data["metadata"].get("source", "未知来源"),
                            **doc_data["metadata"]
                        }
                    })
                    
                # 按相关性排序并限制结果数量
                scored_results.sort(key=lambda x: x["score"], reverse=True)
                
                # 去重：每个文档只保留最相关的块
                unique_results = []
                seen_docs = set()
                
                for result in scored_results:
                    doc_id = result["metadata"]["doc_id"]
                    if doc_id not in seen_docs:
                        seen_docs.add(doc_id)
                        unique_results.append(result)
                        
                        if len(unique_results) >= top_k:
                            break
                            
                results = unique_results
            else:
                # 回退到简单的文本搜索
                logger.info("向量搜索不可用，使用简单文本搜索")
                
                query_lower = query.lower()
                scored_results = []
                
                # 遍历所有文档
                for doc_id, info in self.metadata.items():
                    # 应用过滤条件
                    if filter_condition:
                        doc_metadata = info["metadata"]
                        matched = False
                        for k, v in doc_metadata.items():
                            if filter_condition in str(v):
                                matched = True
                                break
                        if not matched:
                            continue
                            
                    # 加载文档
                    doc_file = self.documents_dir / f"{doc_id}.json"
                    if not doc_file.exists():
                        continue
                        
                    async with aiofiles.open(doc_file, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        doc_data = json.loads(content)
                        
                    # 检查每个块
                    for chunk in doc_data["chunks"]:
                        chunk_content = chunk["content"].lower()
                        if query_lower in chunk_content:
                            # 简单的匹配分数
                            score = chunk_content.count(query_lower) / len(chunk_content)
                            
                            scored_results.append({
                                "score": score,
                                "content": chunk["content"],
                                "metadata": {
                                    "doc_id": doc_id,
                                    "chunk_id": chunk["chunk_id"],
                                    "source": doc_data["metadata"].get("source", "未知来源"),
                                    **doc_data["metadata"]
                                }
                            })
                            
                # 按相关性排序
                scored_results.sort(key=lambda x: x["score"], reverse=True)
                results = scored_results[:top_k]
                
            return {
                "success": True,
                "query": query,
                "results": results
            }
            
        except Exception as e:
            logger.exception(f"搜索文档出错: {str(e)}")
            return {
                "success": False,
                "query": query,
                "error": {"message": f"搜索失败: {str(e)}"},
                "results": []
            }
            
    async def get_documents(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """
        获取文档列表
        
        Args:
            limit: 返回的文档数量
            offset: 开始位置
            
        Returns:
            文档列表字典
        """
        try:
            # 按更新时间排序
            sorted_docs = sorted(
                self.metadata.values(),
                key=lambda x: x["metadata"].get("updated_at", ""),
                reverse=True
            )
            
            # 应用分页
            paginated = sorted_docs[offset:offset + limit]
            
            return {
                "success": True,
                "documents": paginated,
                "total": len(sorted_docs)
            }
            
        except Exception as e:
            logger.exception(f"获取文档列表出错: {str(e)}")
            return {
                "success": False,
                "error": {"message": f"获取文档列表失败: {str(e)}"},
                "documents": [],
                "total": 0
            }
            
    async def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        获取单个文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档详情字典
        """
        try:
            if doc_id not in self.metadata:
                return {
                    "success": False,
                    "error": {"message": f"文档不存在: {doc_id}"}
                }
                
            # 加载文档
            doc_file = self.documents_dir / f"{doc_id}.json"
            if not doc_file.exists():
                return {
                    "success": False,
                    "error": {"message": f"文档文件不存在: {doc_id}"}
                }
                
            async with aiofiles.open(doc_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                doc_data = json.loads(content)
                
            return {
                "success": True,
                "document_id": doc_id,
                "metadata": doc_data["metadata"],
                "chunks": doc_data["chunks"]
            }
            
        except Exception as e:
            logger.exception(f"获取文档出错: {str(e)}")
            return {
                "success": False,
                "error": {"message": f"获取文档失败: {str(e)}"}
            } 