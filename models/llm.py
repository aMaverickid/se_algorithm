"""
大语言模型，支持DeepSeek、通义千问等多个大模型
"""
import os
import logging
import time
import json
import requests
import threading
import queue
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from utils.rag_utils import get_rag_processor

logger = logging.getLogger(__name__)

class LLMModel:
    """
    大语言模型，支持多种大模型的统一接口
    
    可支持的模型:
    - deepseek: DeepSeek大模型
    - qwen: 通义千问大模型
    - custom_api: 自定义API
    
    实现两种使用方式:
    1. 本地部署模式：直接在本地加载轻量级大语言模型(需要较多资源)
    2. API调用模式：调用远程API接口(需要网络和API密钥)
    """
    
    def __init__(self, 
                 model_type: str = "deepseek", 
                 api_key: Optional[str] = None,
                 api_url: Optional[str] = None,
                 device: str = config.DEVICE):
        """
        初始化大语言模型
        
        Args:
            model_type: 模型类型，支持'deepseek', 'qwen'
            api_key: API密钥(API调用模式需要)
            api_url: API地址(API调用模式需要)
            device: 运行设备，'cuda'或'cpu'
        """
        self.model_type = model_type
        self.api_key = api_key
        self.api_url = api_url
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # API设置
        self.api_config = {
            "deepseek": {
                "url": "https://api.deepseek.com/v1/chat/completions",
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}" if api_key else ""
                }
            },
            "qwen": {
                "url": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}" if api_key else ""
                }
            }
        }
        
        logger.info(f"初始化LLMModel，模型: {model_type}")
    
    async def generate(self, 
                     messages: List[Dict[str, str]], 
                     temperature: float = 0.7,
                     max_tokens: int = 1024,
                     documents: Optional[List[str]] = None,
                     use_rag: bool = False,
                     rag_query: Optional[str] = None,
                     rag_top_k: int = 5,
                     rag_filter: Optional[str] = None,
                     stream: bool = False) -> Dict[str, Any]:
        """
        生成对话响应
        
        Args:
            messages: 对话历史消息列表，格式[{"role": "user", "content": "..."}, ...]
            temperature: 生成温度，值越低越确定性
            max_tokens: 最大生成长度
            documents: 相关文档内容列表，用于指导模型思考
            use_rag: 是否使用检索增强生成
            rag_query: RAG检索查询，如果为None则使用最后一条用户消息
            rag_top_k: 检索返回的文档数量
            rag_filter: 检索过滤条件
            stream: 是否流式输出
            
        Returns:
            生成的响应和元数据
        """
        # 准备检索结果
        retrieval_results = []
        references = None
        
        # 如果启用RAG
        if use_rag:
            # 确定检索查询
            if not rag_query and messages:
                # 获取最后一条用户消息作为查询
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        rag_query = msg.get("content", "")
                        break
            
            if rag_query:
                # 执行检索
                retrieval_results, references = await self._retrieve_context(
                    query=rag_query,
                    top_k=rag_top_k,
                    filter_expr=rag_filter
                )
                
                # 如果检索到文档，添加到上下文
                if retrieval_results:
                    # 如果提供了documents，合并结果
                    if documents:
                        documents = retrieval_results + documents
                    else:
                        documents = retrieval_results
        
        # 生成响应
        response = await self._generate_via_api(messages, temperature, max_tokens, documents, stream)
        
        # 添加引用信息
        if references and not stream:
            response["references"] = references
        
        return response
    
    async def _retrieve_context(self, 
                              query: str, 
                              top_k: int = 5,
                              filter_expr: Optional[str] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        从向量数据库检索相关上下文
        
        Args:
            query: 检索查询
            top_k: 返回结果数量
            filter_expr: 过滤表达式
            
        Returns:
            (文本列表, 引用信息列表)
        """
        try:
            # 获取RAG处理器
            rag_processor = get_rag_processor()
            
            # 执行检索
            search_results = rag_processor.search(
                query=query,
                limit=top_k,
                filter_expr=filter_expr
            )
            
            # 提取文本和引用信息
            context_texts = []
            references = []
            
            for result in search_results:
                # 添加文本到上下文
                context_texts.append(result["text"])
                
                # 添加引用信息
                references.append({
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "score": result["score"]
                })
            
            return context_texts, references
        
        except Exception as e:
            logger.error(f"检索上下文失败: {str(e)}")
            return [], []
        
    async def _generate_via_api(self, 
                              messages: List[Dict[str, str]], 
                              temperature: float = 0.7,
                              max_tokens: int = 1024,
                              documents: Optional[List[str]] = None,
                              stream: bool = False) -> Dict[str, Any]:
        """使用API生成对话响应"""
        # 获取API配置
        api_config = self.api_config.get(self.model_type)
        
        # 如果提供了文档，将其合并到系统消息中
        if documents and len(documents) > 0:
            # 检查是否已有系统消息
            has_system_message = False
            for msg in messages:
                if msg.get("role") == "system":
                    # 在现有系统消息后添加文档内容
                    msg["content"] += "\n\n参考文档:\n" + "\n\n".join(documents)
                    has_system_message = True
                    break
            
            # 如果没有系统消息，创建一个
            if not has_system_message:
                system_message = {
                    "role": "system",
                    "content": "参考以下文档回答用户问题:\n" + "\n\n".join(documents)
                }
                # 在消息列表开头添加系统消息
                messages = [system_message] + messages
        
        try:
            # 准备请求参数
            model_name = self._get_api_model_name()
            
            # 不同模型的API请求格式不同
            if self.model_type == "deepseek":
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": stream
                }
                
            elif self.model_type == "qwen":
                payload = {
                    "model": model_name,
                    "input": {
                        "messages": messages
                    },
                    "parameters": {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "result_format": "message"
                    }
                }
                
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
            
            # 处理流式或非流式响应
            if stream:
                # 流式响应处理
                logger.info(f"开始流式生成，模型: {model_name}")
                return await self._handle_streaming_response(api_config["url"], api_config["headers"], payload)
            else:
                # 非流式响应处理
                logger.info(f"开始生成，模型: {model_name}")
                start_time = time.time()
                
                response = requests.post(
                    api_config["url"],
                    headers=api_config["headers"],
                    json=payload,
                    timeout=60
                )
                
                # 检查响应状态
                if response.status_code != 200:
                    logger.error(f"API请求失败: {response.status_code} {response.text}")
                    error_message = response.text
                    try:
                        error_json = response.json()
                        if "error" in error_json:
                            error_message = error_json["error"].get("message", error_message)
                    except:
                        pass
                    
                    return {
                        "message": {
                            "role": "assistant",
                            "content": f"生成失败: {error_message}"
                        },
                        "model": model_name,
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    }
                
                # 处理响应
                response_json = response.json()
                generation_time = time.time() - start_time
                
                if self.model_type == "deepseek":
                    content = response_json["choices"][0]["message"]["content"]
                    total_tokens = response_json["usage"]["total_tokens"]
                    prompt_tokens = response_json["usage"]["prompt_tokens"]
                    completion_tokens = response_json["usage"]["completion_tokens"]
                    
                elif self.model_type == "qwen":
                    content = response_json["output"]["choices"][0]["message"]["content"]
                    total_tokens = response_json["usage"]["total_tokens"]
                    prompt_tokens = response_json["usage"]["input_tokens"]
                    completion_tokens = response_json["usage"]["output_tokens"]
                
                logger.info(f"生成完成，耗时: {generation_time:.2f}秒，共计: {total_tokens}个token")
                
                # 构建响应
                return {
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "model": model_name,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                }
        
        except Exception as e:
            logger.error(f"生成过程中发生错误: {str(e)}")
            
            return {
                "message": {
                    "role": "assistant",
                    "content": f"生成失败: {str(e)}"
                },
                "model": self.model_type,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
    
    async def _handle_streaming_response(self, api_url, headers, payload):
        """处理流式响应"""
        # 创建响应队列
        response_queue = queue.Queue()
        
        # 创建处理线程
        def process_stream():
            try:
                # 发送流式请求
                with requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=60
                ) as response:
                    # 检查响应状态
                    if response.status_code != 200:
                        logger.error(f"流式API请求失败: {response.status_code}")
                        error_message = f"流式生成失败，状态码: {response.status_code}"
                        
                        try:
                            error_text = next(response.iter_lines()).decode('utf-8')
                            error_json = json.loads(error_text)
                            if "error" in error_json:
                                error_message = error_json["error"].get("message", error_message)
                        except:
                            pass
                        
                        # 将错误信息放入队列
                        response_queue.put(error_message)
                        response_queue.put(None)  # 结束标记
                        return
                    
                    # 处理流式响应
                    if self.model_type == "deepseek":
                        for line in response.iter_lines():
                            if not line:
                                continue
                            
                            # 移除SSE前缀
                            line = line.decode('utf-8')
                            if line.startswith("data: "):
                                line = line[6:]
                            
                            if line == "[DONE]":
                                break
                            
                            try:
                                chunk_data = json.loads(line)
                                if "choices" in chunk_data:
                                    chunk = chunk_data["choices"][0]
                                    if "delta" in chunk and "content" in chunk["delta"]:
                                        content = chunk["delta"]["content"]
                                        response_queue.put(content)
                            except json.JSONDecodeError:
                                continue
                            except Exception as e:
                                logger.error(f"处理流式响应块时出错: {str(e)}")
                    
                    elif self.model_type == "qwen":
                        # 阿里云的流式响应格式不同
                        buffer = b""
                        for chunk in response.iter_content(chunk_size=8192):
                            if not chunk:
                                continue
                            
                            buffer += chunk
                            lines = buffer.split(b'\n')
                            buffer = lines.pop()
                            
                            for line in lines:
                                line = line.decode('utf-8').strip()
                                if not line or line.startswith(":"):  # SSE注释
                                    continue
                                
                                if line.startswith("data: "):
                                    line = line[6:]
                                
                                try:
                                    chunk_data = json.loads(line)
                                    if "output" in chunk_data:
                                        chunk = chunk_data["output"]["choices"][0]
                                        if "message" in chunk and "content" in chunk["message"]:
                                            content = chunk["message"]["content"]
                                            response_queue.put(content)
                                except json.JSONDecodeError:
                                    continue
                                except Exception as e:
                                    logger.error(f"处理流式响应块时出错: {str(e)}")
                
                # 添加结束标记
                response_queue.put(None)
                
            except Exception as e:
                logger.error(f"流式请求过程中发生错误: {str(e)}")
                # 将错误信息放入队列
                response_queue.put(f"流式生成过程中出错: {str(e)}")
                response_queue.put(None)  # 结束标记
        
        # 启动处理线程
        threading.Thread(target=process_stream).start()
        
        # 返回响应队列
        return response_queue
    
    def _convert_messages_to_prompt(self, 
                                  messages: List[Dict[str, str]], 
                                  documents: Optional[List[str]] = None) -> str:
        """
        将消息列表转换为单一提示词字符串
        
        Args:
            messages: 消息列表
            documents: 文档列表
            
        Returns:
            格式化的提示词
        """
        prompt = ""
        
        # 添加文档
        if documents:
            prompt += "参考以下文档回答问题:\n\n"
            for i, doc in enumerate(documents):
                prompt += f"[文档 {i+1}]\n{doc}\n\n"
            prompt += "基于上述文档，"
        
        # 添加对话历史
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                prompt += f"系统指令: {content}\n\n"
            elif role == "user":
                prompt += f"用户问题: {content}\n\n"
            elif role == "assistant":
                prompt += f"助手回答: {content}\n\n"
        
        # 添加最后的提示
        prompt += "助手回答: "
        
        return prompt
    
    def _get_api_model_name(self) -> str:
        """获取API模型名称"""
        model_names = {
            "deepseek": "deepseek-chat",
            "qwen": "qwen-max"
        }
        return model_names.get(self.model_type, self.model_type)
    
    def __del__(self):
        """清理资源"""
        # 如果是本地模型，需要释放资源
        if self.model_loaded and self.model is not None:
            del self.model
            del self.tokenizer
            
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class TextIteratorStreamer:
    """用于流式输出文本的迭代器"""
    
    def __init__(self, tokenizer, skip_prompt=False):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.text_queue = queue.Queue()
        self.stop_signal = None
        self.text_iterator = None
    
    def put(self, token_ids):
        """将token放入队列"""
        if token_ids is not None:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            self.text_queue.put(text)
    
    def end(self):
        """发送结束信号"""
        self.text_queue.put(self.stop_signal)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        value = self.text_queue.get()
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value 