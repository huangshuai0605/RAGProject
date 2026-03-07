from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import IndexType, MilvusClient, Function
from pymilvus.client.types import MetricType, DataType, FunctionType

from documents.markdown_parser import MarkdownParser
#from llm_models.embeddings_model import bge_embedding
from llm_models.embeddings_model import openai_embedding
from utils.env_utils import MILVUS_URI, COLLECTION_NAME

class MilvusVectorSave:
    """把新的document数据插入到数据库中"""

    def __init__(self) -> object:
        """自定义collection的索引"""
        self.vector_store_saved: Milvus = None

    def create_collection(self):
        client = MilvusClient(uri=MILVUS_URI)   #创建客户端连接
        schema = client.create_schema() #创建表结构定义
        schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)   #定义表的列结构：
        schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=6000, enable_analyzer=True,
                         analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})   #使用jieba进行中文分词，只保留中文，字母，数字
        schema.add_field(field_name='category', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filename', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filetype', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='category_depth', datatype=DataType.INT64)
        schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
        #schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=512)
        schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=1536)

        bm25_function = Function(       #自动从text字段计算BM25 稀疏向量，每次插入数据时，Milvus 会自动调用此函数生成 sparse 向量
            name="text_bm25_emb",  # Function name
            input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
            output_field_names=["sparse"],
            # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
            function_type=FunctionType.BM25,  # Set to `BM25`
        )
        schema.add_function(bm25_function)
        index_params = client.prepare_index_params()

        index_params.add_index(     #稀疏向量索引，用于BM25检索
            field_name="sparse",                        #索引字段
            index_name="sparse_inverted_index",         #索引名称
            index_type="SPARSE_INVERTED_INDEX",  # Inverted index type for sparse vectors   #稀疏倒排索引
            metric_type="BM25",                         #使用BM25评分
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",         #倒排索引算法
                # Algorithm for building and querying the index. Valid values: DAAT_MAXSCORE, DAAT_WAND, TAAT_NAIVE.
                "bm25_k1": 1.2,         #BM25参数k1，控制词频饱和度
                "bm25_b": 0.75          #控制文档长度归一化
            },
        )
        index_params.add_index(  #稠密向量索引（用于向量相似度检索）
            field_name="dense",                     #索引字段
            index_name="dense_inverted_index",      #索引名称
            index_type=IndexType.HNSW,  # Inverted index type for sparse vectors  #HNSW图索引算法
            metric_type=MetricType.IP,      #内积相似度
            params={"M": 16, "efConstruction": 64}  # M :邻接节点数, efConstruction: 搜索范围    #每个节点的最大连接数
        )

        if COLLECTION_NAME in client.list_collections():    #删除已存在的集合
            # 先释放， 再删除索引，再删除collection
            client.release_collection(collection_name=COLLECTION_NAME)
            client.drop_index(collection_name=COLLECTION_NAME, index_name='sparse_inverted_index')
            client.drop_index(collection_name=COLLECTION_NAME, index_name='dense_inverted_index')
            client.drop_collection(collection_name=COLLECTION_NAME)

        client.create_collection(           #创建新集合
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

    def create_connection(self):
        """创建一个Connection： milvus + langchain。pip install  langchain-milvus"""
        self.vector_store_saved = Milvus(
            embedding_function=openai_embedding,  #bge_embedding
            collection_name=COLLECTION_NAME,
            builtin_function=BM25BuiltInFunction(),
            vector_field=['dense', 'sparse'],
            consistency_level="Strong",
            auto_id=True,
            connection_args={"uri": MILVUS_URI}
        )

    def add_documents(self, datas: List[Document]):
        """把新的document保存到Milvus中"""
        try:
            result = self.vector_store_saved.add_documents(datas)
            # 显式调用flush和load
            self.vector_store_saved.client.flush(collection_name=COLLECTION_NAME)
            # 注意：LangChain-Milvus 封装可能不同，确保集合被加载
            #self.vector_store_saved.client.load_collection(collection_name=COLLECTION_NAME)
            print(f"插入结果: {result}")
            if result:  # 或者根据具体返回值判断
                print("✅ 文档插入成功！")
                return True
            else:
                print("❌ 文档插入失败！")
                return False
        except Exception as e:
            print(f"❌ 插入过程中发生错误: {e}")
            return False
    # def add_documents(self, datas: List[Document]):
    #     """把新的document保存到Milvus中"""
    #     self.vector_store_saved.add_documents(datas)

if __name__ == '__main__':
    # 解析文件内容
    file_path = r'D:\PyCharm\PROJECT\RAGProject\datas\md\tech_report_0tfhhamx.md'
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)

    # 写入Milvus数据库
    mv = MilvusVectorSave()
    mv.create_collection()  #创建表集合
    mv.create_connection()  #连接数据库
    mv.add_documents(docs)  #写入

    client = mv.vector_store_saved.client
    # 得到表结构
    desc_collection = client.describe_collection(
        collection_name=COLLECTION_NAME
    )
    print('表结构是: ', desc_collection)

    # 得到当前表的，所有的index
    res = client.list_indexes(
        collection_name=COLLECTION_NAME
    )
    print('表中的所有索引：', res)

    if res:
        for i in res:
            # 得到索引的描述
            desc_index = client.describe_index(
                collection_name=COLLECTION_NAME,
                index_name=i
            )
            print(desc_index)

    # result = client.query(
    #     collection_name=COLLECTION_NAME,
    #     filter="category == 'Title'",  # 查询 category == 'Title' 的所有数据
    #     output_fields=['text', 'category', 'filename']  # 指定返回的字段
    # )
    #print('测试 过滤查询的结果是: ', result)
    # result_all = client.query(
    #     collection_name=COLLECTION_NAME,
    #     limit=5,  # 限制返回条数
    #     output_fields=['id', 'text', 'category']  # 查看关键字段
    # )
    result_all = client.query(
        collection_name=COLLECTION_NAME,
        filter="id != 0",  # 或你的过滤条件
        output_fields=['id', 'text']  # 不包含sparse
    )
    print("所有数据样本:", result_all)
    print("所有数据样本数量:", len(result_all))

