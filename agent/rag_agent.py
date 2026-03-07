import os
from langchain.agents import create_agent
from tools.retriever_tools import retriever_tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware

from llm_models.all_llm import llm,summary_model


# 创建agent - LangChain 1.0风格
agent = create_agent(
    model=llm,
    tools=[retriever_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=summary_model,
            max_tokens_before_summary=800,
            # 自定义摘要提示词
            summary_prompt="请用简短的语言总结对话历史，保留关键信息"
        )
    ],
    system_prompt="你是一个智能助手，若遇到'半导体'相关的问题，可以调用工具回答用户的问题"
)

# # 存储对话历史
# store = {}
#
# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     """获取或创建会话历史"""
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]
if __name__ == '__main__':
    #response = agent.invoke({'input': '请帮我写一个关于机器学习的Python代码'})  半导体和芯片
    # response = agent.invoke({
    #     "messages": [{"role": "user", "content": "25 乘以 8 等于多少？"}]
    # })


    # response = agent.invoke({
    #     "messages": [{"role": "user", "content": "半导体和芯片是什么"}]
    # })
    # print(response)
    #
    # print("\n完整消息历史：")
    # for i, msg in enumerate(response['messages'], 1):
    #     print(f"\n{'=' * 60}")
    #     print(f"消息 {i}: {msg.__class__.__name__}")
    #     print(f"{'=' * 60}")
    #
    #     if hasattr(msg, 'content') and msg.content:
    #         print(f"内容: {msg.content}")
    #
    #     if hasattr(msg, 'tool_calls') and msg.tool_calls:
    #         print(f"工具调用:")
    #         for tc in msg.tool_calls:
    #             print(f"  - 工具: {tc['name']}")
    #             print(f"  - 参数: {tc['args']}")
    #
    #     if hasattr(msg, 'name'):
    #         print(f"工具名: {msg.name}")

    # 测试2: 同一对话（有上下文）
    print("\n\n" + "=" * 60)
    print("测试2: 有上下文对话（同一会话）")
    print("=" * 60)


    config = {"configurable": {"thread_id": "customer_123"}}

    # 模拟客服对话
    conversations = [
        "什么是半导体封装？",
        "它有哪些主要类型？",
        "它的类型分别有什么用",  # 这里的"它"指代之前的半导体封装
    ]
    import pdb
    pdb.set_trace()
    for msg in conversations:
        print(f"\n客户: {msg}")
        response = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config=config
        )
        # for i, msg in enumerate(response['messages'], 1):
        #     print(f"\n{'=' * 60}")
        #     print(f"消息 {i}: {msg.__class__.__name__}")
        #     print(f"{'=' * 60}")
        #
        #     if hasattr(msg, 'content') and msg.content:
        #         print(f"内容: {msg.content}")
        #
        #     if hasattr(msg, 'tool_calls') and msg.tool_calls:
        #         print(f"工具调用:")
        #         for tc in msg.tool_calls:
        #             print(f"  - 工具: {tc['name']}")
        #             print(f"  - 参数: {tc['args']}")
        #
        #     if hasattr(msg, 'name'):
        #         print(f"工具名: {msg.name}")
        # print(f"客服: {response['messages'][-1].content}")
