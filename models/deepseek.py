"""
DeepSeek 大语言模型接口
"""
import logging
import os
import json
from typing import List, Dict, Any, Optional
import aiohttp
import config
from utils.rag_utils import RAGManager

logger = logging.getLogger(__name__)

class DeepSeekModel:
    """DeepSeek 大语言模型接口"""
    
    def __init__(self):
        """初始化DeepSeek模型接口"""
        self.api_key = config.DEEP_SEEK_API_KEY
        self.api_base = "https://api.deepseek.com"
        self.model = "deepseek-chat"  # 或使用其他可用模型
        self.rag_manager = RAGManager()
        logger.info(f"初始化DeepSeek模型: {self.model}")

    async def generate_response(self, 
                              messages: str,
                              temperature: float = 0.7,
                              max_tokens: int = 1024,
                              documents: Optional[List[str]] = None,
                              use_rag: bool = False,
                              rag_query: Optional[str] = None,
                              rag_top_k: int = 5,
                              rag_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        生成聊天响应
        
        Args:
            messages: 用户输入的消息数组JSON字符串
            temperature: 生成温度，默认0.7
            max_tokens: 最大生成令牌数，默认1024
            documents: 附加文档内容，用于指导模型
            use_rag: 是否使用RAG功能
            rag_query: RAG检索查询，若为None则使用最后一条用户消息
            rag_top_k: 检索返回文档数量
            rag_filter: 检索过滤条件
            
        Returns:
            包含生成响应的字典
        """
        try:
            # 解析消息列表
            messages_list = json.loads(messages)
            if not isinstance(messages_list, list):
                raise ValueError("Messages should be a JSON string of a list")
            
            # 如果启用RAG且没有提供文档，进行检索
            references = None
            if use_rag:
                if not rag_query and messages_list:
                    # 默认使用最后一条用户消息作为查询
                    for msg in reversed(messages_list):
                        if msg.get('role') == 'user':
                            rag_query = msg.get('content', '')
                            break
                
                if rag_query:
                    # 执行文档检索
                    search_results = await self.rag_manager.search(
                        query=rag_query,
                        top_k=rag_top_k,
                        filter_condition=rag_filter
                    )
                    
                    # 准备引用文档
                    if search_results.get("results"):
                        references = search_results["results"]
                        # 将检索到的文档作为系统消息添加到对话
                        context_docs = self._format_context_from_results(search_results["results"])
                        if context_docs:
                            # 在对话开头添加系统消息，包含检索到的文档内容
                            messages_list.insert(0, {
                                "role": "system",
                                "content": f"请使用以下参考文档来回答用户问题。如果参考文档中没有相关信息，请如实告知。\n\n{context_docs}"
                            })
            
            # 如果提供了文档，添加到对话中
            if documents and not use_rag:
                context = "\n\n".join(documents)
                messages_list.insert(0, {
                    "role": "system", 
                    "content": f"请使用以下参考文档来回答用户问题。如果参考文档中没有相关信息，请如实告知。\n\n{context}"
                })
            
            # 请求DeepSeek API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages_list,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True  # 启用流式输出
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base}/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"DeepSeek API错误: {response.status} - {error_text}")
                        return {
                            "message": f"API错误: {response.status}",
                            "model": self.model,
                            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                            "references": references
                        }
                    
                    # 流式输出
                    return {
                        "message": "__STREAM__",  # 特殊标记表示这是一个流
                        "model": self.model,
                        "stream": response,  # 返回响应对象以便调用方处理流
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},  # 最终会更新
                        "references": references
                    }
                    
        except Exception as e:
            logger.exception(f"DeepSeek模型生成错误: {str(e)}")
            return {
                "message": f"生成错误: {str(e)}",
                "model": self.model,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "references": None
            }
    
    def _format_context_from_results(self, results: List[Dict[str, Any]]) -> str:
        """
        从检索结果中格式化上下文信息
        
        Args:
            results: 检索结果列表
            
        Returns:
            格式化的上下文字符串
        """
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            source = result.get("metadata", {}).get("source", "未知来源")
            context_parts.append(f"文档{i} (来源: {source}):\n{content}\n")
        
        return "\n".join(context_parts) 