# from pymilvus import MilvusClient #Milvus客户端，用于操作嵌入式向量数据库
# import numpy as np
#
# #初始化Milvus客户端
# #参数 "./milvus_demo.db" 指定了本地数据库文件的路径
# client = MilvusClient("./milvus_demo.db")

from pymilvus import MilvusClient
import numpy as np

client = MilvusClient("http://localhost:19530")

# 创建集合（Collection）
# 集合类似于关系型数据库中的表，用于存储向量和其他字段
client.create_collection(
collection_name="demo_collection",  # 集合名称为 "demo_collection"
dimension=384  # 向量的维度为 384，表示每个向量是一个长度为 384 的浮点数数组
)

#准备数据：文档、向量和其他字段
docs = [
"Artificial intelligence was founded as an academic discipline in 1956.",
"Alan Turing was the first person to conduct substantial research in AI.",
"Born in Maida Vale, London, Turing was raised in southern England."
]

# 为每段文本生成一个随机的384维向量
#使用NumPy 的np.random.uniform～生成范围在-l到1之间的随机数
vectors = [[np.random.uniform(-1, 1) for _ in range(384)] for _ in range(len(docs))]

data = [
{"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
for i in range(len(vectors))
]

# 将数据插入到集合中
res = client.insert(
collection_name="demo_collection",  # 指定目标集合
data=data  # 要插入的数据列表
)
# 输出插入结果（通常返回成功状态或插入的记录数）
print("Insert result:", res)

# 执行相似性搜索
# 在集合中查找与查询向量最相似的记录
res = client.search(
collection_name="demo_collection",  # 指定目标集合
data=[vectors[0]],  # 查询向量（这里使用了第一个文档的向量）
filter="subject == 'history'",  # 筛选条件：只返回主题为 "history" 的记录
limit=2,  # 返回的最大结果数量（这里是 2 条）
output_fields=["text", "subject"],  # 指定返回的字段（这里返回 "text" 和 "subject"）
)
# 输出搜索结果
print("Search result:", res)
