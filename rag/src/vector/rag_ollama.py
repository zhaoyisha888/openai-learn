import ollama  

# 本地部署向量化模型，断网也能跑哦

def get_embedding(text, model="bge-m3") -> list:
    response =  ollama.embed(model,text)  
    print(type(response))  # <class 'ollama._types.EmbedResponse'> 
    print()
    '''
    ollama.embed() 返回的是 Pydantic 模型对象EmbedResponse，该对象继承自 BaseModel 
    有类型提示和验证
    可以通过属性访问：response.embeddings
    也支持字典式访问：response['embeddings']（Pydantic 实现了 __getitem__ ）
    
    '''
    # 属性和字典效果一样
    # embedding =  response['embeddings']   
    # embedding =  response.embeddings
    # 这里也可以用model_dump()转成字典类型，然后跟字典取值一样操作取所有文本的向量
    embedding = response.model_dump()['embeddings']
    return embedding

test_query = ['飞飞鱼','玛卡巴卡']
vec = get_embedding(test_query)

for i,enmbedding in enumerate(vec):
    print(f'第{i}个文本是{test_query[i]}, 向量列表为 {enmbedding}，向量维度为 {len(enmbedding)}')   
    print()