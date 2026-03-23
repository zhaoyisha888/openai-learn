# 实现了一个混合搜索系统，结合了BM25和密集向量检索两种方法
import numpy as np
from rank_bm25 import BM25Okapi
import jieba
import json
import chromadb
from chromadb.config import Settings

from models import get_normal_client,ALI_TONGYI_EMBEDDING

# 查询的问题
query = "嘴唇肿起来了，怎么办"

# 1、读取文件准备处理
with open('train_zh.json', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# 记录数太多，只取50条数据
# instruction->症状描述；output->症状解释或治疗方案
print(len(data))
instructions = [entry['instruction'] for entry in data[0:50]]   # 取50条来用，切多了烧token，费钱
print(instructions[0])
outputs = [entry['output'] for entry in data[0:50]]
print(outputs[0])

#---------------------------------------------------------
#2、开始BM25进行全文检索
#在运用 BM25 算法进行全文检索时，需要对文档进行分词，以此把文本拆分成一个个独立的词语，方便后续计算词语在文档中的频率等统计信息
# 文档分词 jieba.lcut(doc)  函数会把 instructions 列表里的每个文档 doc 进行分词
tokenized_corpus = [jieba.lcut(doc) for doc in instructions]
print(tokenized_corpus)

# 初始化一个BM25Okapi对象，用于基于BM25算法的文本检索或相似度计算
#对传入的文档计算必要的统计信息
bm25 = BM25Okapi(tokenized_corpus)

# 问题分词 :对查询的问题也需要进行分词处理，这样才能计算查询词和文档的相似度分数。
tokenized_query = jieba.lcut(query)

# 通过BM25算法计算查询词与文档的相似度分数
bm25_scores = bm25.get_scores(tokenized_query)
# 通过BM25算法获取与查询最相关的前3个结果
bm25_results = bm25.get_top_n(tokenized_query, outputs, n=3)
print("BM25 Score: ", bm25_scores)
print("BM25 Results: ", bm25_results)

#---------------------------------------------------------
#3、开始词向量相似度检索
client = get_normal_client()

#封装向量模型与API的交互操作，通过自定义函数 get_embeddings 提供向量模型的调用。
def get_embeddings(texts, model=ALI_TONGYI_EMBEDDING):
    data = client.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]

#get_embeddings函数的变体版，因为各个模型对一次能处理的文本条数有限制且每个平台不一致，新增一个batch_size参数用以控制。
def get_embeddings_batch(texts, model=ALI_TONGYI_EMBEDDING, batch_size=10):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_text = texts[i:i + batch_size]
        data = client.embeddings.create(input=batch_text, model=model).data
        all_embeddings.extend([x.embedding for x in data])
    return all_embeddings

#封装与向量数据库（ChromaDB）的交互操作，通过自定义类 MyVectorDBConnector 提供添加文档和检索文档的功能。
# 具体步骤包括初始化数据库客户端、创建集合、添加文档及其对应的向量表示，以及根据查询向量检索相似文档。
class MyVectorDBConnector:
    # collection_name：向量数据库中集合的名称。embedding_fn：一个用于将文本转换为向量的函数。
    def __init__(self, collection_name, embedding_fn):
        chroma_client = chromadb.Client(Settings(allow_reset=True))
        chroma_client.reset()

        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        self.embedding_fn = embedding_fn

    # 添加文档与向量
    def add_documents(self, instructions, outputs):
        embeddings = self.embedding_fn(instructions)
        # 向 collection 中添加文档与向量
        self.collection.add(
            embeddings=embeddings,  # 每个文档的向量
            documents=outputs,  # 文档的原文
            ids=[f"id{i}" for i in range(len(instructions))]  # 每个文档的 id
        )

    # 定义检索函数， 在向量数据库里进行检索操作
    def search(self, query, top_n):
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            # 指定要返回的最相似文档的数量。
            n_results=top_n
        )
        return results


# 创建一个向量数据库对象
vector_db = MyVectorDBConnector("demo", get_embeddings_batch)
# 向向量数据库中添加文档
vector_db.add_documents(instructions, outputs)
# 根据问题进行检索，返回最相似的 top_n 个结果
results = vector_db.search(query, 3)
# 打印最相似的 top_n 个结果
print("vector_db:", results['documents'][0])
print("vector_db Score:", results['distances'][0])

#---------------------------------------------------------
# 4、组合BM25和词向量相似度检索的结果
#对全文检索和向量检索，都有相似度分数，基于分数进行融合？
#分数要进行标准化，学习L4也对数据标准化

#把python中的列表转为np所能支持的格式
bm25_score = np.array(bm25_scores)
#np.max找出其中最大的那个数，最大值做分母，得出的数一定小于1，所有的缩到了[0,1]，这叫归一化
bm25_scores_normalized = bm25_score / np.max(bm25_score)

# 获取查询的向量表示，并把结果转换为 NumPy 数组
query_embedding = np.array(get_embeddings_batch(query))
# 获取文档的向量表示，并把结果转换为 NumPy 数组
doc_embeddings = np.array(get_embeddings_batch(instructions))
# 计算查询向量和文档向量之间的欧氏距离
# np.linalg.norm 函数用于计算向量的范数，这里计算的是向量差的 L2 范数，即欧氏距离。axis=1 表示按行计算。
dense_scores = np.linalg.norm(query_embedding - doc_embeddings, axis=1)
#做归一化，1减去标准化的分数以后，得到相似度的分数
dense_scores_normalized = 1 - (dense_scores / np.max(dense_scores))
#分数的组合，引入一个权重的概念，目的根据业务的需求，决定文档的重要性，算出一个复合分数
combined_scores = 0.5 * bm25_scores_normalized + 0.5 * dense_scores_normalized

top_idx = combined_scores.argsort()[::-1]
print(top_idx)
hybrid_results = [outputs[i] for i in top_idx[:3]]

# 5. 输出混合搜索的结果
print("Hybrid Search Results: ", hybrid_results)

'''
肺结核：
是一种由结核菌(结核分枝杆菌)引发的传染病，俗称"肺痨"，主要影响肺部，有时也会波及气管和支气管。
患者常出现持续咳嗽、咳痰超过两周，有时痰中带血，还可能伴有低烧、夜间盗汗、身体乏力等症状。这种疾病通过空气传播。
胃溃疡：
(俗称"胃里长疮"”)是一种常见的胃部疾病，主要表现为胃内壁出现破损伤口。正常情况下，
胃里有一层天然的保护膜(医学称胃黏膜)。当这层保护膜因为感染、药物刺激或胃酸过多等原因受损时，
胃酸就会直接腐蚀胃壁，形成类似口腔溃疡的创面。这种创面可能引起上腹部疼痛(心窝附近)、饭后腹胀、反酸烧心等症状，
严重时甚至会导致胃出血。胃溃疡属于消化性溃疡的一种类型，这类溃疡的共同特点都是被胃酸腐蚀形成的伤口，
就像皮肤被酸液灼伤后出现的溃烂。
'''