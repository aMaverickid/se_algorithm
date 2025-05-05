"""
RAG（检索增强生成）工具模块，提供文档处理、向量化和检索功能
"""
import os
import sys
import logging
import json
import uuid
import shutil
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import time

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

try:
    import lancedb
    from lancedb.embeddings import EmbeddingFunctionRegistry
    from lancedb.pydantic import LanceModel, Vector
    from sentence_transformers import SentenceTransformer
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # 文件处理库
    import pypdf
    import docx
    import markdown
    from unstructured.partition.auto import partition
except ImportError:
    logger.warning("缺少RAG相关库，请安装: pip install lancedb sentence-transformers langchain-text-splitters unstructured pypdf markdown python-docx")

class DocumentChunk(LanceModel):
    """文档块模型，用于向量数据库存储"""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Vector

class RAGProcessor:
    """
    RAG处理器，提供文档的加载、处理、向量化和检索功能
    """
    
    def __init__(self, 
                 db_path: Optional[str] = None,
                 embedding_model: Optional[str] = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 device: str = config.DEVICE,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        初始化RAG处理器
        
        Args:
            db_path: 向量数据库路径，默认为config.DOCS_DIR/"vectordb"
            embedding_model: 嵌入模型名称或路径
            device: 运行设备，'cuda'或'cpu'
            chunk_size: 文档分块大小
            chunk_overlap: 文档分块重叠大小
        """
        self.db_path = db_path or (config.DOCS_DIR / "vectordb")
        self.embedding_model_name = embedding_model
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 创建向量数据库目录
        os.makedirs(self.db_path, exist_ok=True)
        
        # 初始化嵌入模型
        self._init_embedding_model()
        
        # 初始化数据库连接
        self._init_db()
        
        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
    
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        try:
            logger.info(f"加载嵌入模型: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name, 
                device=self.device
            )
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"嵌入模型加载完成，向量维度: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {str(e)}")
            raise
    
    def _init_db(self):
        """初始化向量数据库"""
        try:
            logger.info(f"连接向量数据库: {self.db_path}")
            self.db = lancedb.connect(self.db_path)
            
            # 检查并创建文档表
            if "documents" not in self.db.table_names():
                logger.info("创建文档表")
                schema = DocumentChunk.schema()
                self.document_table = self.db.create_table(
                    "documents", 
                    schema=schema,
                    mode="create"
                )
            else:
                self.document_table = self.db.open_table("documents")
            
            logger.info("向量数据库初始化完成")
        except Exception as e:
            logger.error(f"初始化向量数据库失败: {str(e)}")
            raise
    
    def load_document(self, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        加载文档并提取文本内容
        
        Args:
            file_path: 文档路径
            metadata: 文档元数据
            
        Returns:
            文档文本内容
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 基本元数据
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "file_type": file_path.suffix.lower(),
            "created_at": time.time()
        })
        
        # 根据文件类型选择处理方法
        file_type = file_path.suffix.lower()
        
        try:
            if file_type == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            
            elif file_type == ".pdf":
                text = self._load_pdf(file_path)
            
            elif file_type in [".docx", ".doc"]:
                text = self._load_docx(file_path)
            
            elif file_type in [".md", ".markdown"]:
                text = self._load_markdown(file_path)
            
            elif file_type in [".html", ".htm"]:
                text = self._load_html(file_path)
            
            else:
                # 尝试使用unstructured库处理其他格式
                text = self._load_with_unstructured(file_path)
            
            return text
            
        except Exception as e:
            logger.error(f"加载文档失败: {str(e)}")
            raise
    
    def _load_pdf(self, file_path: Path) -> str:
        """加载PDF文档"""
        text = ""
        try:
            with open(file_path, "rb") as f:
                pdf = pypdf.PdfReader(f)
                for page in pdf.pages:
                    text += page.extract_text() + "\n\n"
        except Exception as e:
            logger.error(f"处理PDF文档失败: {str(e)}")
            # 尝试使用unstructured作为备选方案
            text = self._load_with_unstructured(file_path)
        
        return text
    
    def _load_docx(self, file_path: Path) -> str:
        """加载Word文档"""
        try:
            doc = docx.Document(file_path)
            return "\n\n".join([para.text for para in doc.paragraphs if para.text])
        except Exception as e:
            logger.error(f"处理Word文档失败: {str(e)}")
            # 尝试使用unstructured作为备选方案
            return self._load_with_unstructured(file_path)
    
    def _load_markdown(self, file_path: Path) -> str:
        """加载Markdown文档"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                md_text = f.read()
            
            # 移除markdown语法
            html = markdown.markdown(md_text)
            # 简单地去除HTML标签
            import re
            text = re.sub(r'<[^>]+>', '', html)
            return text
        except Exception as e:
            logger.error(f"处理Markdown文档失败: {str(e)}")
            # 直接返回原文本作为备选方案
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
    
    def _load_html(self, file_path: Path) -> str:
        """加载HTML文档"""
        try:
            from bs4 import BeautifulSoup
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            
            # 获取文本，忽略脚本和样式
            for script in soup(["script", "style"]):
                script.extract()
            
            text = soup.get_text()
            # 整理文本格式
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
        except Exception as e:
            logger.error(f"处理HTML文档失败: {str(e)}")
            # 尝试使用unstructured作为备选方案
            return self._load_with_unstructured(file_path)
    
    def _load_with_unstructured(self, file_path: Path) -> str:
        """使用unstructured库加载文档"""
        try:
            elements = partition(str(file_path))
            return "\n\n".join([str(el) for el in elements])
        except Exception as e:
            logger.error(f"使用unstructured处理文档失败: {str(e)}")
            # 对于二进制文件，可能无法提取文本
            return f"无法提取文件内容: {file_path.name}"
    
    def add_document(self, 
                    file_path: Union[str, Path], 
                    metadata: Optional[Dict[str, Any]] = None, 
                    doc_id: Optional[str] = None) -> str:
        """
        添加文档到向量数据库
        
        Args:
            file_path: 文档路径
            metadata: 文档元数据
            doc_id: 文档ID，如不提供则自动生成
            
        Returns:
            文档ID
        """
        # 加载文档
        text = self.load_document(file_path, metadata)
        
        # 为文档生成唯一ID
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        # 文档元数据
        if metadata is None:
            metadata = {}
        
        # 添加文档ID到元数据
        metadata["doc_id"] = doc_id
        
        # 分割文档
        chunks = self.text_splitter.split_text(text)
        
        # 向量化并存储
        for i, chunk in enumerate(chunks):
            # 生成块ID
            chunk_id = f"{doc_id}_{i}"
            
            # 更新块级元数据
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": chunk_id,
                "chunk_index": i,
                "chunk_count": len(chunks)
            })
            
            # 向量化
            embedding = self.embed_text(chunk)
            
            # 创建文档块
            document_chunk = DocumentChunk(
                id=chunk_id,
                text=chunk,
                metadata=chunk_metadata,
                embedding=embedding.tolist()
            )
            
            # 存储到数据库
            self.document_table.add([document_chunk])
        
        logger.info(f"文档添加成功: {doc_id}, 共{len(chunks)}个块")
        return doc_id
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        向量化文本
        
        Args:
            text: 待向量化的文本
            
        Returns:
            文本向量
        """
        try:
            return self.embedding_model.encode(text)
        except Exception as e:
            logger.error(f"文本向量化失败: {str(e)}")
            raise
    
    def search(self, 
              query: str, 
              limit: int = 5,
              filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        搜索相关文档
        
        Args:
            query: 搜索查询
            limit: 返回结果数量
            filter_expr: 过滤表达式，如"metadata.source == 'manual'"
            
        Returns:
            相关文档列表
        """
        try:
            # 向量化查询
            query_embedding = self.embed_text(query)
            
            # 构建搜索
            search_query = self.document_table.search(query_embedding)
            
            # 添加过滤器
            if filter_expr:
                search_query = search_query.where(filter_expr)
            
            # 执行搜索
            results = search_query.limit(limit).to_list()
            
            # 处理结果
            search_results = []
            for result in results:
                search_results.append({
                    "id": result.id,
                    "text": result.text,
                    "metadata": result.metadata,
                    "score": result.score
                })
            
            return search_results
        
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            是否成功删除
        """
        try:
            # 删除所有属于该文档的块
            self.document_table.delete(f"metadata.doc_id = '{doc_id}'")
            logger.info(f"文档删除成功: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            return False
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        列出所有文档(不包含块)
        
        Returns:
            文档列表
        """
        try:
            # 获取所有唯一的doc_id
            query = "SELECT DISTINCT metadata.doc_id, metadata.file_name, metadata.file_type, metadata.created_at FROM documents"
            results = self.document_table.query(query).to_list()
            
            # 处理结果
            documents = []
            for result in results:
                # 提取元数据
                doc_id = result["metadata.doc_id"]
                file_name = result["metadata.file_name"]
                file_type = result["metadata.file_type"]
                created_at = result["metadata.created_at"]
                
                documents.append({
                    "id": doc_id,
                    "file_name": file_name,
                    "file_type": file_type,
                    "created_at": created_at
                })
            
            return documents
        
        except Exception as e:
            logger.error(f"获取文档列表失败: {str(e)}")
            return []
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        获取文档的所有块
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档块列表
        """
        try:
            # 获取所有属于该文档的块
            chunks = self.document_table.to_pandas(
                where=f"metadata.doc_id = '{doc_id}'",
                columns=["id", "text", "metadata"]
            )
            
            # 转换为字典列表
            chunk_list = []
            for _, row in chunks.iterrows():
                chunk_list.append({
                    "id": row["id"],
                    "text": row["text"],
                    "metadata": row["metadata"]
                })
            
            # 按块索引排序
            chunk_list.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
            
            return chunk_list
        
        except Exception as e:
            logger.error(f"获取文档块失败: {str(e)}")
            return []
    
    def clear_database(self) -> bool:
        """
        清空向量数据库
        
        Returns:
            是否成功清空
        """
        try:
            # 删除并重新创建表
            if "documents" in self.db.table_names():
                self.db.drop_table("documents")
            
            schema = DocumentChunk.schema()
            self.document_table = self.db.create_table(
                "documents", 
                schema=schema,
                mode="create"
            )
            
            logger.info("向量数据库已清空")
            return True
        
        except Exception as e:
            logger.error(f"清空向量数据库失败: {str(e)}")
            return False

# 全局RAG处理器实例
_rag_processor = None

def get_rag_processor() -> RAGProcessor:
    """
    获取或创建RAG处理器实例
    
    Returns:
        RAGProcessor实例
    """
    global _rag_processor
    
    if _rag_processor is None:
        try:
            _rag_processor = RAGProcessor(
                db_path=os.path.join(config.DOCS_DIR, "vectordb"),
                device=config.DEVICE
            )
        except ImportError:
            logger.error("缺少RAG相关库，请安装所需依赖")
            raise
    
    return _rag_processor 