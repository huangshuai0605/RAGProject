from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from llm_models.all_llm import llm
from tools.retriever_tools import retriever_tool
from typing import TypedDict, Annotated, Literal, List
from pydantic import BaseModel
#from mem0 import MemoryClient, Memory
from mem0 import  Memory
import operator
import os

# 初始化记忆客户端
MEM0_API_KEY = os.getenv("MEM0_API_KEY", "your-mem0-api-key")
#memory_client = MemoryClient(api_key=MEM0_API_KEY)
memory = Memory()


# ==================== 1. 定义状态 ====================

class AgentState(TypedDict):
    """Agent工作流状态"""
    messages: Annotated[List[HumanMessage | AIMessage], operator.add]
    should_retrieve: bool
    is_relevant: bool
    final_answer: str
    user_id: str
    has_memories: bool
    memory_context: str
    memories_are_relevant: bool  # 新增：记忆是否相关


# ==================== 2. 定义节点函数 ====================

def memory_check_node(state: AgentState) -> dict:
    """记忆检查节点：检查记忆中是否有相关信息，并评估相关性"""
    print("\n" + "=" * 60)
    print("[记忆检查节点] 检查记忆中是否有相关信息并评估相关性...")
    print("=" * 60)

    # 获取最新的用户消息
    messages = state.get("messages", [])
    if not messages:
        return {
            "has_memories": False,
            "memory_context": "",
            "memories_are_relevant": False,
            "messages": []
        }

    # 查找最新的人类消息
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_message = msg
            break
    else:
        last_message = messages[-1]

    question = last_message.content
    user_id = state.get("user_id", "default_user")

    # 从记忆中检索相关信息
    try:
        relevant_memories = memory.search(
            query=question,
            user_id=user_id,
            limit=5  # 增加检索数量，以便有更多选择
        )

        # 检查是否有记忆
        if relevant_memories and "results" in relevant_memories and relevant_memories["results"]:
            print(f"找到 {len(relevant_memories['results'])} 条原始记忆")

            # 评估每条记忆的相关性
            relevant_contexts = []

            for i, memory_entry in enumerate(relevant_memories["results"]):
                memory_content = memory_entry.get("memory", "")
                if not memory_content:
                    continue

                # 使用LLM评估这条记忆是否与当前问题相关
                is_relevant = evaluate_memory_relevance(question, memory_content)

                if is_relevant:
                    relevant_contexts.append(memory_content)
                    print(f"记忆 {i + 1}: 相关 ✓")
                else:
                    print(f"记忆 {i + 1}: 不相关 ✗")

            if relevant_contexts:
                # 构建记忆上下文
                memories_str = "\n".join(f"- {context}" for context in relevant_contexts[:3])  # 最多3条
                memory_context = f"相关记忆:\n{memories_str}"

                print(f"筛选后保留 {len(relevant_contexts)} 条相关记忆")
                print(f"记忆上下文预览: {memory_context[:200]}...")

                return {
                    "has_memories": True,
                    "memory_context": memory_context,
                    "memories_are_relevant": True,
                    "messages": []
                }
            else:
                print("没有找到相关记忆")
                return {
                    "has_memories": False,
                    "memory_context": "",
                    "memories_are_relevant": False,
                    "messages": []
                }
        else:
            print("没有找到任何记忆")
            return {
                "has_memories": False,
                "memory_context": "",
                "memories_are_relevant": False,
                "messages": []
            }

    except Exception as e:
        print(f"记忆检索出错: {e}")
        return {
            "has_memories": False,
            "memory_context": "",
            "memories_are_relevant": False,
            "messages": []
        }


def evaluate_memory_relevance(question: str, memory_content: str) -> bool:
    """评估记忆内容是否与问题相关"""
    evaluation_prompt = """评估以下记忆内容是否与用户问题相关：

用户问题：{question}

记忆内容：{memory}

判断标准：
1. 记忆内容直接回答了用户问题
2. 记忆内容提供了用户问题所需的背景信息
3. 记忆内容与用户问题在主题上高度相关
4. 记忆内容包含了用户问题中提到的关键词或概念

如果相关，回答"relevant"，否则回答"no relation"。

只回答"relevant"或"no relation"，不要其他内容。"""

    try:
        messages = [
            SystemMessage(content=evaluation_prompt.format(
                question=question,
                memory=memory_content[:500]  # 限制长度
            ))
        ]

        response = llm.invoke(messages)
        relevance = response.content.strip().lower()

        return "relevant" in relevance
    except Exception as e:
        print(f"记忆相关性评估出错: {e}")
        return False  # 出错时默认为不相关


def answer_from_memory_node(state: AgentState) -> dict:
    """从记忆中回答节点：基于记忆生成答案"""
    print("\n" + "=" * 60)
    print("[记忆回答节点] 基于记忆生成答案...")
    print("=" * 60)

    messages = state.get("messages", [])
    if not messages:
        return {
            "final_answer": "抱歉，我没有收到您的问题。",
            "messages": []
        }

    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_message = msg
            break
    else:
        last_message = messages[-1]

    question = last_message.content
    memory_context = state.get("memory_context", "")
    user_id = state.get("user_id", "default_user")

    # 基于记忆生成答案
    system_prompt = f"""你是一个有用的AI助手。请基于用户的记忆回答问题。

{memory_context}

用户问题：{{question}}

请根据记忆提供准确、有帮助的回答。只使用记忆中明确包含的信息，不要添加记忆中没有的内容。"""

    messages_to_send = [
        SystemMessage(content=system_prompt.format(question=question)),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages_to_send)
    final_answer = response.content

    # 将此次对话保存到记忆中
    try:
        conversation_memory = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": final_answer}
        ]
        memory.add(conversation_memory, user_id=user_id)
        print(f"已将对话保存到记忆")
    except Exception as e:
        print(f"保存记忆时出错: {e}")

    print(f"用户问题: {question}")
    print(f"基于记忆回答长度: {len(final_answer)} 字符")
    print(f"回答预览: {final_answer[:200]}...")

    return {
        "final_answer": final_answer,
        "messages": []
    }


def agent_node(state: AgentState) -> dict:
    """Agent节点：分析用户问题，决定是否需要检索"""
    print("\n" + "=" * 60)
    print("[agent节点] 分析用户问题，决定是否需要检索...")
    print("=" * 60)

    messages = state.get("messages", [])
    if not messages:
        return {
            "should_retrieve": False,
            "messages": []
        }

    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_message = msg
            break
    else:
        last_message = messages[-1]

    question = last_message.content

    # 系统提示：让LLM决定是否需要检索
    system_prompt = """判断用户问题是否需要检索半导体知识库。

半导体相关：包括半导体设计、制造、封装、测试、材料、设备、工艺等。

如果问题是半导体相关，回答"yes"，否则回答"no"。

只回答"yes"或"no"，不要其他内容。

问题：{question}"""

    messages_to_send = [
        SystemMessage(content=system_prompt.format(question=question)),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages_to_send)
    answer = response.content.strip().lower()

    should_retrieve = answer == "yes"
    print(f"用户问题: {question}")
    print(f"是否需要检索: {should_retrieve}")

    return {
        "should_retrieve": should_retrieve,
        "messages": []
    }


def general_node(state: AgentState) -> dict:
    """普通问答节点：直接回答不需要检索的问题"""
    print("\n" + "=" * 60)
    print("[general节点] 直接回答问题...")
    print("=" * 60)

    messages = state.get("messages", [])
    if not messages:
        return {
            "final_answer": "抱歉，我没有收到您的问题。",
            "messages": []
        }

    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_message = msg
            break
    else:
        last_message = messages[-1]

    question = last_message.content
    user_id = state.get("user_id", "default_user")

    # 直接回答问题
    answer_prompt = """你是一个有用的AI助手。请根据你的知识直接回答用户的问题。

用户问题：{question}

请提供准确、有帮助的回答。如果你不确定，请如实说明。"""

    messages_to_send = [
        SystemMessage(content=answer_prompt.format(question=question)),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages_to_send)
    final_answer = response.content

    # 将此次对话保存到记忆中
    try:
        conversation_memory = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": final_answer}
        ]
        memory.add(conversation_memory, user_id=user_id)
        print(f"已将对话保存到记忆")
    except Exception as e:
        print(f"保存记忆时出错: {e}")

    print(f"用户问题: {question}")
    print(f"直接回答长度: {len(final_answer)} 字符")
    print(f"回答预览: {final_answer[:200]}...")

    return {
        "final_answer": final_answer,
        "messages": []
    }


def retrieve_node(state: AgentState) -> dict:
    """检索节点：执行检索工具"""
    print("\n" + "=" * 60)
    print("[retrieve节点] 执行检索...")
    print("=" * 60)

    last_message = state["messages"][-1]
    question = last_message.content

    result = retriever_tool.invoke({"query": question})

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

    last_message = state["messages"][-1]
    retrieved_docs = last_message.content

    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    question = user_messages[-1].content if user_messages else ""

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
            docs=retrieved_docs[:1000]
        ))
    ]

    response = llm.invoke(messages)
    relevance = response.content.strip().lower()
    is_relevant = "relevant" in relevance

    print(f"用户问题: {question}")
    print(f"文档相关性: {'相关' if is_relevant else '不相关'}")

    return {
        "is_relevant": is_relevant,
        "messages": []
    }


def rewrite_node(state: AgentState) -> dict:
    """重写查询节点：优化查询词"""
    print("\n" + "=" * 60)
    print("[rewrite节点] 重写查询以获取更好结果...")
    print("=" * 60)

    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    original_question = user_messages[-1].content if user_messages else ""

    last_message = state["messages"][-1]
    retrieved_docs = last_message.content

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

    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    question = user_messages[-1].content if user_messages else ""

    tool_messages = [msg for msg in state["messages"]
                     if hasattr(msg, 'tool_calls') and msg.tool_calls]
    context = tool_messages[-1].content if tool_messages else ""

    generation_prompt = """基于以下文档回答用户问题：

用户问题：{question}

相关文档：
{context}

请提供准确、全面的回答。如果文档中没有相关信息，请如实说明。"""

    messages = [
        SystemMessage(content=generation_prompt.format(
            question=question,
            context=context[:2000]
        ))
    ]

    response = llm.invoke(messages)
    final_answer = response.content

    try:
        user_id = state.get("user_id", "default_user")
        conversation_memory = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": final_answer}
        ]
        memory.add(conversation_memory, user_id=user_id)
        print(f"已将对话保存到记忆")
    except Exception as e:
        print(f"保存记忆时出错: {e}")

    print(f"用户问题: {question}")
    print(f"最终答案长度: {len(final_answer)} 字符")
    print(f"答案预览: {final_answer[:200]}...")

    return {
        "final_answer": final_answer,
        "messages": []
    }


# ==================== 3. 条件判断函数 ====================

def memory_condition(state: AgentState) -> Literal["answer_from_memory", "agent"]:
    """判断是否有相关记忆"""
    if state.get("has_memories", False) and state.get("memories_are_relevant", False):
        return "answer_from_memory"
    return "agent"


def tools_condition(state: AgentState) -> Literal["retrieve", "general"]:
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

def create_rag_workflow_with_memory():
    """创建完整的RAG工作流（带记忆模块）"""

    workflow = StateGraph(AgentState)

    workflow.add_node("memory_check", memory_check_node)
    workflow.add_node("answer_from_memory", answer_from_memory_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("relevance_evaluation", document_relevance_evaluation)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("general", general_node)

    workflow.add_edge(START, "memory_check")

    workflow.add_conditional_edges(
        "memory_check",
        memory_condition,
        {
            "answer_from_memory": "answer_from_memory",
            "agent": "agent"
        }
    )

    workflow.add_edge("answer_from_memory", END)

    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "retrieve": "retrieve",
            "general": "general"
        }
    )

    workflow.add_edge("retrieve", "relevance_evaluation")

    workflow.add_conditional_edges(
        "relevance_evaluation",
        relevance_condition,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )

    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("general", END)
    workflow.add_edge("generate", END)

    app = workflow.compile()

    return app


if __name__ == "__main__":
    main_app = create_rag_workflow_with_memory()

    test_queries = [
        "我是蔡徐坤",
        "什么是半导体封装？",
        "我喜欢唱跳rap",
        "我是谁，我的爱好是什么",
        "晶圆级封装的优势是什么？"
    ]

    for i, query in enumerate(test_queries):
        print("\n" + "=" * 80)
        print(f"测试查询 {i + 1}: {query}")
        print("=" * 80)

        initial_state = {
            "messages": [HumanMessage(content=query)],
            "should_retrieve": False,
            "is_relevant": False,
            "final_answer": "",
            "user_id": "test_user_1",
            "has_memories": False,
            "memory_context": "",
            "memories_are_relevant": False
        }

        try:
            result = main_app.invoke(initial_state)

            print(f"\n最终结果:")
            print("-" * 40)

            if "final_answer" in result and result["final_answer"]:
                print(f"最终回答: {result['final_answer'][:300]}...")
            else:
                print("未生成最终答案，工作流可能提前结束")

            print(f"\n最终状态摘要:")
            print(f"是否有相关记忆: {result.get('has_memories', 'N/A')}")
            print(f"记忆是否相关: {result.get('memories_are_relevant', 'N/A')}")
            print(f"是否执行检索: {result.get('should_retrieve', 'N/A')}")
            print(f"文档是否相关: {result.get('is_relevant', 'N/A')}")
            print(f"用户ID: {result.get('user_id', 'N/A')}")

        except Exception as e:
            print(f"执行出错: {e}")
            import traceback

            traceback.print_exc()

        print("\n" + "-" * 80)