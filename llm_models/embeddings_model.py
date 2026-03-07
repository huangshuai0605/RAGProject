from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from utils.env_utils import OPENAI_API_KEY,OPENAI_BASE_URL
import  numpy as np

openai_embedding = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL
)

# model_name = "BAAI/bge-small-zh-v1.5"
# model_kwargs = {"device": "cpu"}
# encode_kwargs = {"normalize_embeddings": True}
# bge_embedding = HuggingFaceEmbeddings(
#     model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
# )


# def test_openai_embedding():
#     """测试OpenAI嵌入模型"""
#     print("=== 测试OpenAI嵌入模型 ===")
#     try:
#         # 初始化OpenAI嵌入
#         openai_embedding = OpenAIEmbeddings(
#             openai_api_key=OPENAI_API_KEY,
#             openai_api_base=OPENAI_BASE_URL
#         )
#
#         # 测试文本
#         test_texts = [
#             "这是一个测试句子。",
#             "这是另一个测试句子。",
#             "测试中文文本嵌入功能。"
#         ]
#
#         # 生成嵌入
#         embeddings = openai_embedding.embed_documents(test_texts)
#
#         # 输出结果
#         print(f"成功生成嵌入数量: {len(embeddings)}")
#         print(f"第一个嵌入维度: {len(embeddings[0])}")
#         print(f"第二个嵌入维度: {len(embeddings[1])}")
#         print(f"示例嵌入前10个值: {embeddings[0][:10]}")
#
#         return True
#
#     except Exception as e:
#         print(f"OpenAI嵌入测试失败: {e}")
#         return False

# def test_huggingface_embedding():
#     """测试HuggingFace BGE嵌入模型"""
#     print("\n=== 测试HuggingFace BGE嵌入模型 ===")
#     try:
#         # 初始化BGE嵌入
#         model_name = "BAAI/bge-small-zh-v1.5"
#         model_kwargs = {"device": "cpu"}
#         encode_kwargs = {"normalize_embeddings": True}
#         bge_embedding = HuggingFaceEmbeddings(
#             model_name=model_name,
#             model_kwargs=model_kwargs,
#             encode_kwargs=encode_kwargs
#         )
#
#         # 测试文本
#         test_texts = [
#             "这是一个测试句子。",
#             "这是另一个测试句子。",
#             "测试中文文本嵌入功能。"
#         ]
#
#         # 生成嵌入
#         embeddings = bge_embedding.embed_documents(test_texts)
#
#         # 输出结果
#         print(f"成功生成嵌入数量: {len(embeddings)}")
#         print(f"嵌入维度: {len(embeddings[0])}")
#         print(f"示例嵌入前10个值: {embeddings[0][:10]}")
#         print(f"嵌入是否归一化: {np.linalg.norm(embeddings[0]):.6f}")
#
#         return True
#
#     except Exception as e:
#         print(f"HuggingFace嵌入测试失败: {e}")
#         return False

if __name__ == "__main__":
    print("开始测试嵌入模型...\n")

    # # 测试OpenAI
    # openai_success = test_openai_embedding()

    # 测试HuggingFace
    # huggingface_success = test_huggingface_embedding()