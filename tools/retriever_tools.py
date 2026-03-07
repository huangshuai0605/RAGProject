from langchain_core.tools import create_retriever_tool  #create_retriever_tool: LangChain的核心工具，将检索器包装成Agent可用的工具
from documents.milvus_db import MilvusVectorSave  #MilvusVectorSave: 自定义类，用于操作Milvus向量数据库（类似一个专业的图书馆管理系统）

mv = MilvusVectorSave()   # 创建"图书馆管理员"实例
mv.create_connection()    # 建立到"图书馆"（Milvus数据库）的连接
retriever = mv.vector_store_saved.as_retriever(
    search_type='similarity',           # 仅返回相似度超过阈值的文档
    search_kwargs={
        "k": 4,                         #使用相似度搜索
        "score_threshold": 0.1,         #相似度阈值0.1（过滤不相关的）
        "ranker_type": "rrf",           #使用RRF算法
        "ranker_params": {"k": 100},    #使用RRF算法的参数
        'filter': {"category": "content"}   #只搜索"content"类别的文档
    }
)


retriever_tool = create_retriever_tool(  # 创建检索工具 tool
    retriever,      #上面创建的检索器
    'rag_retriever',        #工具名称
    '搜索并返回关于 ‘半导体和芯片’ 的信息, 内容涵盖：半导体和芯片的封装、测试、光刻胶等'
)

if __name__ == '__main__':
    # 测试查询
    test_queries = [
        "什么是芯片封装？",
        "光刻胶的作用是什么？",
        "芯片测试的流程是怎样的？",
        "半导体制造的关键步骤有哪些？"
    ]

    print("\n开始测试检索工具...")
    print("-" * 50)

    for i, query in enumerate(test_queries, 1):
        print(f"\n测试 {i}: '{query}'")
        print("-" * 30)

        try:
            # 使用工具进行搜索
            result = retriever_tool.invoke({"query": query})
            # import  pdb
            # pdb.set_trace()
            # 打印结果
            print(f"查询: {query}")
            print(f"找到文档长度{len(result)}")

            print(f"{i}",result)

        except Exception as e:
            print(f"执行查询时出错: {e}")
# from langchain_core.tools import tool
# from langchain_core.documents import Document
# from typing import List, Dict, Any
# import json
#
# # 导入MilvusVectorSave
# from documents.milvus_db import MilvusVectorSave
#
# # 创建Milvus连接实例
# mv = MilvusVectorSave()
# mv.create_connection()
#
# # 创建检索器
# retriever = mv.vector_store_saved.as_retriever(
#     search_type='similarity',
#     search_kwargs={
#         "k": 4,
#         "score_threshold": 0.1,
#         "ranker_type": "rrf",
#         "ranker_params": {"k": 100},
#         'filter': {"category": "content"}
#     }
# )
#
# @tool
# def rag_retriever(query: str) -> str:
#     """
#     搜索并返回关于'半导体和芯片'的信息，内容涵盖：半导体和芯片的封装、测试、光刻胶等
#
#     参数:
#         query: 搜索查询字符串
#
#     返回:
#         检索到的相关信息字符串
#     """
#     try:
#         result = retriever.invoke(query)
#         import pdb
#         pdb.set_trace()
#         return result
#     except Exception as e:
#         return f"执行查询时出错: {e}"
#
#
# # 测试工具
# if __name__ == "__main__":
#     print("=" * 60)
#     print("测试半导体信息检索工具")
#     print("=" * 60)
#
#     # 测试1: 使用基本工具
#     print("\n1. 测试基本检索工具:")
#     print("-" * 40)
#
#     test_queries = [
#         "什么是芯片封装？",
#         "光刻胶的作用是什么？",
#         "芯片测试的流程是怎样的？",
#         "半导体制造的关键步骤有哪些？"
#     ]
#
#     for i, query in enumerate(test_queries, 1):
#         print(f"\n测试 {i}: '{query}'")
#         print("-" * 30)
#
#         try:
#             result = rag_retriever.invoke({"query": query})
#             print(f"查询: {query}")
#             print(f"找到文档: {len(result) if isinstance(result, list) else 'N/A'}")
#             print(
#                 f"结果预览: {result[:200]}..." if isinstance(result, str) and len(result) > 200 else f"结果: {result}")
#         except Exception as e:
#             print(f"执行查询时出错: {e}")