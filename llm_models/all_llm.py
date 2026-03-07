from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI

from utils.env_utils import OPENAI_API_KEY, DEEPSEEK_API_KEY,OPENAI_BASE_URL, TAVILY_API_KEY

# llm = ChatOpenAI(  # openai的
#     temperature=0,
#     model='gpt-4o-mini',
#     api_key=OPENAI_API_KEY,
#     base_url=OPENAI_BASE_URL)

llm = ChatOpenAI(
    temperature=0.5,
    model='deepseek-chat',
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com")

summary_model = ChatOpenAI(
    temperature=0.5,
    model='deepseek-chat',
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    max_tokens=1024)


web_search_tool = TavilySearchResults(
    max_results=2,
    api_key=TAVILY_API_KEY)

# print(OPENAI_API_KEY)
# print(DEEPSEEK_API_KEY)
# print(OPENAI_BASE_URL)
print(TAVILY_API_KEY)


# 测试1：直接调用搜索工具
def test_search_tool_directly():
    """直接调用搜索工具"""
    print("=" * 50)
    print("测试 Tavily 搜索工具")
    print("=" * 50)

    # 定义搜索查询
    search_queries = [
        "2024年最新的人工智能发展",
        "OpenAI的最新动态",
        "Python编程技巧"
    ]

    for query in search_queries:
        print(f"\n🔍 搜索查询: {query}")
        print("-" * 30)

        try:
            # 执行搜索
            results = web_search_tool.invoke({"query": query})

            # 打印结果
            if isinstance(results, list):
                for i, result in enumerate(results, 1):
                    print(f"结果 {i}:")
                    if isinstance(result, dict):
                        for key, value in result.items():
                            if key in ['content', 'title', 'url', 'snippet']:
                                print(f"  {key}: {value[:150]}...")  # 截断显示
                    else:
                        print(f"  {result}")
                    print()
            else:
                print(f"搜索结果: {results}")

        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            print(f"错误类型: {type(e).__name__}")

# 运行测试
if __name__ == "__main__":
    #test_search_tool_directly()
    pass