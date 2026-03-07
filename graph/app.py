# app.py - FastAPI主应用文件
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import json
from datetime import datetime
import asyncio

# 导入已有的RAG工作流
from langchain_core.messages import HumanMessage
from graph2 import create_rag_workflow

# 初始化FastAPI应用
app = FastAPI(
    title="RAG工作流API",
    description="基于LangGraph的智能问答系统API",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化工作流
rag_app = create_rag_workflow()


# 定义请求/响应模型
class ChatRequest(BaseModel):
    """聊天请求模型"""
    query: str
    conversation_id: Optional[str] = None  # 可选会话ID
    user_id: Optional[str] = None  # 可选用户ID
    stream: bool = False  # 是否流式输出


class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str
    conversation_id: Optional[str] = None
    timestamp: str
    processing_time: float
    source: str  # 回答来源：direct或rag


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: str
    version: str


class WorkflowStats(BaseModel):
    """工作流统计"""
    total_requests: int
    avg_processing_time: float
    rag_usage: float  # RAG使用比例


# 全局统计
stats = {
    "total_requests": 0,
    "total_processing_time": 0,
    "rag_responses": 0,
    "direct_responses": 0
}

# 会话历史存储（简单内存存储，生产环境请用Redis或数据库）
conversation_history = {}


@app.get("/", tags=["根路径"])
async def root():
    """根路径，返回API信息"""
    return {
        "name": "RAG工作流API",
        "version": "1.0.0",
        "description": "基于LangGraph的智能问答系统",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "stats": "/stats",
            "history": "/history/{conversation_id}"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["健康检查"])
async def health_check():
    """健康检查端点"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.get("/stats", response_model=WorkflowStats, tags=["统计信息"])
async def get_stats():
    """获取工作流统计信息"""
    avg_time = stats["total_processing_time"] / max(stats["total_requests"], 1)
    rag_usage = (stats["rag_responses"] / max(stats["total_requests"], 1)) * 100

    return WorkflowStats(
        total_requests=stats["total_requests"],
        avg_processing_time=round(avg_time, 3),
        rag_usage=round(rag_usage, 2)
    )


@app.post("/chat", response_model=ChatResponse, tags=["聊天接口"])
async def chat(request: ChatRequest):
    """
    处理用户查询并返回回答

    - **query**: 用户的问题
    - **conversation_id**: 可选，用于连续对话
    - **user_id**: 可选，用户标识
    - **stream**: 是否流式输出（当前版本暂不支持）
    """
    start_time = datetime.now()

    # 更新统计
    stats["total_requests"] += 1

    try:
        # 初始化状态
        initial_state = {
            "messages": [HumanMessage(content=request.query)],
            "should_retrieve": False,
            "is_relevant": False,
            "final_answer": ""
        }

        # 运行工作流
        result = rag_app.invoke(initial_state)

        # 获取最终答案
        final_answer = result.get("final_answer", "")

        # 判断回答来源
        source = "rag" if result.get("should_retrieve", False) else "direct"

        if source == "rag":
            stats["rag_responses"] += 1
        else:
            stats["direct_responses"] += 1

        # 如果没有生成答案，返回默认响应
        if not final_answer:
            final_answer = "抱歉，我无法回答这个问题。请尝试换一种方式提问。"

        # 生成或使用提供的conversation_id
        conversation_id = request.conversation_id or f"conv_{datetime.now().timestamp()}"

        # 保存历史记录
        if conversation_id not in conversation_history:
            conversation_history[conversation_id] = []

        conversation_history[conversation_id].append({
            "query": request.query,
            "response": final_answer,
            "timestamp": datetime.now().isoformat(),
            "source": source
        })

        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds()
        stats["total_processing_time"] += processing_time

        return ChatResponse(
            response=final_answer,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat(),
            processing_time=round(processing_time, 3),
            source=source
        )

    except Exception as e:
        # 记录错误
        print(f"处理查询时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理查询时出错: {str(e)}")


@app.get("/history/{conversation_id}", tags=["历史记录"])
async def get_conversation_history(conversation_id: str, limit: int = 10):
    """获取指定会话的历史记录"""
    if conversation_id not in conversation_history:
        raise HTTPException(status_code=404, detail="会话不存在")

    history = conversation_history[conversation_id][-limit:]
    return {
        "conversation_id": conversation_id,
        "total_messages": len(conversation_history[conversation_id]),
        "history": history
    }


@app.post("/clear_history/{conversation_id}", tags=["历史记录"])
async def clear_conversation_history(conversation_id: str):
    """清空指定会话的历史记录"""
    if conversation_id in conversation_history:
        del conversation_history[conversation_id]
        return {"message": "历史记录已清空", "conversation_id": conversation_id}
    else:
        raise HTTPException(status_code=404, detail="会话不存在")


@app.get("/workflow/nodes", tags=["工作流信息"])
async def get_workflow_nodes():
    """获取工作流节点信息"""
    try:
        # 获取工作流图结构
        graph = rag_app.get_graph()

        nodes = []
        for node in graph.nodes:
            nodes.append({
                "name": node,
                "type": "node"
            })

        edges = []
        for edge in graph.edges:
            edges.append({
                "from": edge[0],
                "to": edge[1]
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges)
        }

    except Exception as e:
        return {"error": str(e)}


# 流式响应端点（可选，需要额外实现）
@app.post("/chat/stream", tags=["聊天接口"])
async def chat_stream(request: ChatRequest):
    """流式聊天接口（暂为占位符）"""
    return {"message": "流式接口暂未实现，请使用普通接口"}


# 批量处理端点
@app.post("/chat/batch", tags=["批量处理"])
async def chat_batch(requests: List[ChatRequest]):
    """批量处理多个查询"""
    results = []

    for req in requests:
        try:
            response = await chat(req)
            results.append(response.dict())
        except Exception as e:
            results.append({
                "error": str(e),
                "query": req.query,
                "success": False
            })

    return {
        "total": len(requests),
        "successful": len([r for r in results if "error" not in r]),
        "results": results
    }


# 中间件示例：记录请求日志
@app.middleware("http")
async def log_requests(request, call_next):
    """记录HTTP请求日志"""
    start_time = datetime.now()

    response = await call_next(request)

    process_time = (datetime.now() - start_time).total_seconds()

    print(
        f"{start_time.strftime('%Y-%m-%d %H:%M:%S')} - {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")

    return response


if __name__ == "__main__":
    # 启动服务器
    print("=" * 60)
    print("RAG工作流API服务器启动")
    print("=" * 60)
    print("API文档: http://localhost:8000/docs")
    print("健康检查: http://localhost:8000/health")
    print("服务地址: http://localhost:8000")
    print("=" * 60)

    uvicorn.run(
        app,
        port=8890
    )