import os

from openai import OpenAI


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
)

# 官方文档向量化模型示例
# completion = client.embeddings.create(
#     model="text-embedding-v3",
#     input=['衣服','漂亮'],          # 输入决定输出的对象数量，几个"Embedding"
#     dimensions=1024, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
#     encoding_format="float"
# )

# print(completion.model_dump_json)  # 打印的是整个响应对象，是两个对象类型
# print(completion.model_dump_json())  # 打印结果如下 data有两个"embedding"向量

'''
{
    "data": [
        {
            "embedding": [
                "-0.07328251749277115",
                "0.04997767135500908",
                "......"
            ],
            "index": 0,
            "object": "embedding"
        },
        {
            "embedding": [
                "-0.09794770181179047",
                "0.0643753856420517",
                ".....",
            ],
            "index": 1,
            "object": "embedding"
        }
    ],
    "model": "text-embedding-v3",
    "object": "list",
    "usage": {
        "prompt_tokens": 2,
        "total_tokens": 2
    },
    "id": "a5f6b5d5-eb57-9e52-830c-6c6e5d6737a8"
}
'''


def get_embeddings(texts, model="text-embedding-v4"):
    ''' 
    texts 是要获取嵌入向量的文本（列表、整数等类型都行），
    model 则是用来指定要使用的模型的名称
    client.embeddings.create(input=texts, model=model)  请求体
    '''
    # 生成文本的嵌入表示。获得的是原始data列表，列表元素是Embedding对象
    data = client.embeddings.create(input=texts, model=model).data  
    # print(type(data))  # <class 'list'> 
    # print(data)        # 打印data本身，[Embedding(embedding=[],index, object)]
    '''
    取data列表中所有Embedding对象的embedding，组成列表并返回
    返回的是[embedding，embedding，....]
    embedding 是每个文本的嵌入向量，本身也是列表，所以函数返回的是[[],[],...]]形式的数据
    '''
    return [x.embedding for x in data]    


# 向量模型以"列表元素"个数为单位生成对应数量的嵌入对象。
query = ["张三","法外狂徒"]   

vec = get_embeddings(query)
print(vec)  # 得到的是函数返回的结果[[...]],是[[],[],...]]形式的数据
print()

# 打印文本的嵌入向量和向量维度，维度由模型决定为一个固定值
for i,enmbedding in enumerate(vec):
    print(f'第{i}个文本的向量列表为 {enmbedding}，向量维度为 {len(enmbedding)}')   
    print()


