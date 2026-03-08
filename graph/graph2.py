from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from llm_models.all_llm import llm
from tools.retriever_tools import retriever_tool
from typing import TypedDict, Annotated, Literal, List
from pydantic import BaseModel
import operator


# ==================== 1. 定义状态 ====================

class AgentState(TypedDict):
    """Agent工作流状态"""
    messages: Annotated[List[HumanMessage | AIMessage], operator.add]
    should_retrieve: bool
    is_relevant: bool
    final_answer: str


# ==================== 2. 定义节点函数 ====================

def agent_node(state: AgentState) -> dict:
    """Agent节点：分析用户问题，决定是否需要检索"""
    print("\n" + "=" * 60)
    print("[agent节点] 分析用户问题，决定是否需要检索...")
    print("=" * 60)

    # 获取最新的用户消息
    messages = state.get("messages", [])
    if not messages:
        return {
            "should_retrieve": False,
            "messages": []
        }

    # 查找最新的人类消息
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_message = msg
            break
    else:
        # 如果没有找到人类消息，使用最后一条消息
        last_message = messages[-1]

    question = last_message.content

    # 系统提示：让LLM决定是否需要检索
    system_prompt = """判断用户问题是否需要检索半导体知识库。

    半导体相关：包括半导体设计、制造、封装、测试、材料、设备、工艺等。

    如果问题是半导体相关，回答"yes"，否则回答"no"。

    只回答"yes"或"no"，不要其他内容。

    问题：{question}"""
    # import pdb
    # pdb.set_trace()
    messages_to_send = [
        SystemMessage(content=system_prompt.format(question=question)),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages_to_send)
    answer = response.content.strip().lower()

    # 决定是否需要检索
    should_retrieve = answer == "yes"
    print(f"用户问题: {question}")
    print(f"是否需要检索: {should_retrieve}")

    return {
        "should_retrieve": should_retrieve,
        "messages": []  # 不添加新消息，只是更新状态
    }

def general_node(state: AgentState) -> dict:
    """普通问答节点：直接回答不需要检索的问题"""
    print("\n" + "=" * 60)
    print("[general节点] 直接回答问题...")
    print("=" * 60)

    # 获取最新的用户消息
    messages = state.get("messages", [])
    if not messages:
        return {
            "final_answer": "抱歉，我没有收到您的问题。",
            "messages": []
        }

    # 查找最新的人类消息
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_message = msg
            break
    else:
        # 如果没有找到人类消息，使用最后一条消息
        last_message = messages[-1]

    question = last_message.content

    # 直接回答问题（不使用检索）
    answer_prompt = """你是一个有用的AI助手。请根据你的知识直接回答用户的问题。

    用户问题：{question}

    请提供准确、有帮助的回答。如果你不确定，请如实说明。"""

    messages_to_send = [
        SystemMessage(content=answer_prompt.format(question=question)),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages_to_send)
    final_answer = response.content

    print(f"用户问题: {question}")
    print(f"直接回答长度: {len(final_answer)} 字符")
    print(f"回答预览: {final_answer[:200]}...")

    return {
        "final_answer": final_answer,
        "messages": []  # 不添加新消息
    }

def retrieve_node(state: AgentState) -> dict:
    """检索节点：执行检索工具"""
    print("\n" + "=" * 60)
    print("[retrieve节点] 执行检索...")
    print("=" * 60)

    # 获取用户问题
    last_message = state["messages"][-1]
    #last_message = state["query"]
    question = last_message.content

    # 调用检索工具
    result = retriever_tool.invoke({"query": question})

    # 创建工具消息
    tool_message = AIMessage(
        content=result,
        tool_calls=[{
            "name": "rag_retriever",
            "args": {"query": question},
            "id": "retrieve_call"
        }]
    )

    print(f"检索查询: {question}")
    print(f"检索结果长度: {len(result)} 字符")

    return {"messages": [tool_message]}


def document_relevance_evaluation(state: AgentState) -> dict:
    """文档相关性评估节点：判断检索到的文档是否相关"""
    print("\n" + "=" * 60)
    print("[文档相关性评估] 评估检索文档的相关性...")
    print("=" * 60)
    # 获取检索结果（最后一条消息应该是工具返回的消息）
    last_message = state["messages"][-1]
    retrieved_docs = last_message.content

    # 获取原始用户问题
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    question = user_messages[-1].content if user_messages else ""

    # 评估相关性
    evaluation_prompt = """评估以下文档与用户问题的相关性：

    用户问题：{question}

    检索到的文档：
    {docs}

    判断标准：
    1. 如果文档直接回答了用户问题或提供了相关信息，回答 "relevant"
    2. 如果文档不相关或没有提供有用信息，回答 "no relation"

    只回答 "relevant" 或 "no relation"，不要其他内容。"""

    messages = [
        SystemMessage(content=evaluation_prompt.format(
            question=question,
            docs=retrieved_docs[:1000]  # 限制长度
        ))
    ]

    response = llm.invoke(messages)
    relevance = response.content.strip().lower()
    # import pdb
    # pdb.set_trace()
    is_relevant = "relevant" in relevance

    print(f"用户问题: {question}")
    print(f"文档相关性: {'相关' if is_relevant else '不相关'}")

    return {
        "is_relevant": is_relevant,
        "messages": []  # 不添加新消息
    }


def rewrite_node(state: AgentState) -> dict:
    """重写查询节点：优化查询词"""
    print("\n" + "=" * 60)
    print("[rewrite节点] 重写查询以获取更好结果...")
    print("=" * 60)

    # 获取用户原始问题和检索结果
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    original_question = user_messages[-1].content if user_messages else ""

    last_message = state["messages"][-1]
    retrieved_docs = last_message.content

    # 重写提示
    rewrite_prompt = """之前的查询没有得到相关文档，请改进查询词：

    原始查询：{question}

    之前检索到的不相关文档：
    {docs}

    请基于以下原则改进查询：
    1. 更具体的关键词
    2. 添加相关技术术语
    3. 澄清问题意图

    返回改进后的查询语句："""

    messages = [
        SystemMessage(content=rewrite_prompt.format(
            question=original_question,
            docs=retrieved_docs[:500]
        ))
    ]

    response = llm.invoke(messages)
    rewritten_query = response.content.strip()

    # 创建重写后的用户消息
    rewritten_message = HumanMessage(
        content=f"原问题: {original_question}\n改进后: {rewritten_query}"
    )

    print(f"原查询: {original_question}")
    print(f"改进后: {rewritten_query}")

    return {"messages": [rewritten_message]}


def generate_node(state: AgentState) -> dict:
    """生成答案节点：基于相关文档生成最终答案"""
    print("\n" + "=" * 60)
    print("[generate节点] 基于相关文档生成答案...")
    print("=" * 60)

    # 获取用户问题和检索结果
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    question = user_messages[-1].content if user_messages else ""

    # 获取检索结果（最后一条工具消息）
    tool_messages = [msg for msg in state["messages"]
                     if hasattr(msg, 'tool_calls') and msg.tool_calls]
    context = tool_messages[-1].content if tool_messages else ""

    # 生成答案
    generation_prompt = """基于以下文档回答用户问题：

    用户问题：{question}

    相关文档：
    {context}

    请提供准确、全面的回答。如果文档中没有相关信息，请如实说明。"""

    messages = [
        SystemMessage(content=generation_prompt.format(
            question=question,
            context=context[:2000]  # 限制上下文长度
        ))
    ]

    response = llm.invoke(messages)
    final_answer = response.content

    print(f"用户问题: {question}")
    print(f"最终答案长度: {len(final_answer)} 字符")
    print(f"答案预览: {final_answer[:200]}...")

    return {
        "final_answer": final_answer,
        "messages": []  # 不添加新消息
    }


# ==================== 3. 条件判断函数 ====================

def tools_condition(state: AgentState) -> Literal["retrieve", END]:
    """判断是否需要检索"""
    if state.get("should_retrieve", False):
        return "retrieve"
    return "general"


def relevance_condition(state: AgentState) -> Literal["generate", "rewrite"]:
    """判断文档是否相关"""
    if state.get("is_relevant", False):
        return "generate"
    return "rewrite"


# ==================== 4. 构建工作流 ====================

def create_rag_workflow():
    """创建完整的RAG工作流"""

    # 创建图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("relevance_evaluation", document_relevance_evaluation)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("general", general_node)  # 添加general节点

    # 设置开始节点
    workflow.add_edge(START, "agent")

    # agent节点后的条件边：是否需要检索？
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "retrieve": "retrieve",  # 需要检索
            "general": "general"  # 不需要检索，直接结束
        }
    )

    workflow.add_edge("retrieve", "relevance_evaluation")

    # retrieve节点后的条件边：文档是否相关？
    workflow.add_conditional_edges(
        "relevance_evaluation",
        relevance_condition,
        {
            "generate": "generate",  # 文档相关，生成答案
            "rewrite": "rewrite"  # 文档不相关，重写查询
        }
    )

    # rewrite节点后的边：重写后回到agent节点
    workflow.add_edge("rewrite", "agent")

    # general节点后的边：直接回答问题后结束
    workflow.add_edge("general", END)

    # generate节点后的边：生成答案后结束
    workflow.add_edge("generate", END)

    # 编译工作流
    app = workflow.compile()

    return app




# 如果你需要绘图功能，可以添加以下代码
def draw_workflow_graph(app, filename="workflow.png"):
    """绘制工作流图"""
    try:
        from draw_png import draw_graph
        draw_graph(app, filename)
        print(f"工作流图已保存到: {filename}")
    except ImportError:
        print("未找到draw_png模块，无法绘制图形")
    except Exception as e:
        print(f"绘图时出错: {e}")
# ==================== 7. 主函数 ====================

def draw_graph(graph, filename="graph.png"):
    """
    绘制工作流图并保存为PNG
    """
    try:
        # 方法1：使用本地pyppeteer渲染（推荐）
        from langchain_core.runnables.graph_mermaid import MermaidDrawMethod

        print("使用Pyppeteer本地渲染图表...")
        mermaid_code = graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.PYPPETEER,
            background_color="white"
        )

        with open(filename, "wb") as f:
            f.write(mermaid_code)

        print(f"图表已保存到: {filename}")

    except Exception as e:
        print(f"绘图时出错: {e}")

        # 方法2：输出mermaid代码到控制台
        print("\n" + "=" * 60)
        print("Mermaid图表代码:")
        print("=" * 60)
        mermaid_text = graph.get_graph().draw_mermaid()
        print(mermaid_text)

        # 保存为文本文件
        txt_filename = filename.replace(".png", ".md")
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(f"```mermaid\n{mermaid_text}\n```")
        print(f"\nMermaid代码已保存到: {txt_filename}")


if __name__ == "__main__":
    # 创建主工作流
    main_app = create_rag_workflow()

    # 可选的绘图功能
    #draw_graph(main_app, "workflow.png")
    #draw_workflow_graph(main_app)

    # 测试不同问题
    test_queries = [
        "你是谁",
        "什么是半导体封装？",
        "你好",
        "晶圆级封装的优势是什么？"
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"测试查询: {query}")
        print("=" * 80)

        # 正确初始化状态
        initial_state = {
            "messages": [HumanMessage(content=query)],  # 包含用户消息
            "should_retrieve": False,  # 默认值
            "is_relevant": False,  # 默认值
            "final_answer": ""  # 初始为空
        }

        try:
            # 运行工作流
            result = main_app.invoke(initial_state)

            print(f"\n最终结果:")
            print("-" * 40)

            # 检查是否有最终答案
            if "final_answer" in result and result["final_answer"]:
                print(f"最终回答: {result['final_answer'][:300]}...")
            else:
                print("未生成最终答案，工作流可能提前结束")

            # 打印最终状态摘要
            print(f"\n最终状态摘要:")
            print(f"是否执行检索: {result.get('should_retrieve', 'N/A')}")
            print(f"文档是否相关: {result.get('is_relevant', 'N/A')}")
            print(f"消息数量: {len(result.get('messages', []))}")

        except Exception as e:
            print(f"执行出错: {e}")
            import traceback

            traceback.print_exc()

        print("\n" + "-" * 80)

