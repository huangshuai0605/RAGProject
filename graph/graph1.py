from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from llm_models.all_llm import llm
from agent.rag_agent import agent as rag_agent
from draw_png import draw_graph

import os
from typing import TypedDict, Annotated, Literal, List

class SupportState(TypedDict):
    query: str
    category: str
    messages: Annotated[list, add_messages]
    response: str

def classifier(state: SupportState) -> dict:
    """分类器: 识别问题类型"""
    print("分类器: 识别问题类型")
    messages = [
        SystemMessage(content="""分析用户问题，返回分类：
    - semiconductor：半导体相关问题
    - general：其他一般性问题
    只返回分类名称，不要其他内容。"""),
        HumanMessage(content=state["query"])
    ]
    response = llm.invoke(messages)
    category = response.content.strip().lower()

    #确保返回有效分类
    if category not in ["semiconductor", "general"]:
        category = "general"

    print(f"  [分类器] 问题分类为：{category}")
    return {"category": category}

def route_to_specialist(state: SupportState) -> Literal["semiconductor", "general"]:
    return state["category"]

def semiconductor(state: SupportState) -> dict:
    """半导体问答专家"""
    print("[半导体问答专家] 处理半导体问答问题...")
    # messages = [
    #     SystemMessage(content="""你是一个问答专家，擅长回答与半导体相关的问题。
    #     可以调用工具回答用户问题"""),
    #     HumanMessage(content=state["query"])
    # ]
    # import pdb
    # pdb.set_trace()
    # 正确的调用方式
    response = rag_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {"response": f"[半导体问答专家] {response['messages'][-1].content}"}

def general_agent(state: SupportState) -> dict:
    """通用客服"""
    print("  [通用客服] 处理一般问题...")

    messages = [
        SystemMessage(content="你是友好的客服代表，请热情地回复用户的问题。用中文回复。"),
        HumanMessage(content=state["query"])
    ]
    response = llm.invoke(messages)

    return {"response": f"😊 [客服] {response.content}"}

graph = StateGraph(SupportState)
#将各个agent加入节点
graph.add_node("classifier", classifier)
graph.add_node("semiconductor", semiconductor)
graph.add_node("general", general_agent)

#START到分类节点
graph.add_edge(START, "classifier")

#分类节点路由
graph.add_conditional_edges(
    "classifier",
    route_to_specialist,
    {
        "semiconductor": "semiconductor",
        "general": "general"
    }
)

graph.add_edge("semiconductor", END)
graph.add_edge("general", END)

app = graph.compile()

if __name__ == "__main__":
    #query = "如何使用langchain?"
    # query = "半导体封装是什么"
    # state = {
    #     "query": query,
    #     "messages": []
    # }
    # result = app.invoke(state)
    # print(result['response'])
    draw_graph(app, "graph.png")

