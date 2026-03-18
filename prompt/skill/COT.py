#零样本提示（Zero-Shot）

#提示词在大模型应用中直接使用，也可以看到效果
from openai import OpenAI
import os

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
client = OpenAI(api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)

prompt = """
Q：罗杰有5个网球。他又买了2罐网球。每个罐子有3个网球。他现在有多少个网球?
A：罗杰一开始有5个球。2罐3个网球，等于6个网球。5 + 6 = 11。答案是11。
Q：自助餐厅有23个苹果。如果他们用20做午餐，又买了6个，他们有多少个苹果?
A：
"""
# 在上面的提示中，我们没有向模型提供任何示例。
def get_completion(prompt, model="qwen-max-latest"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
    )
    return response.choices[0].message.content
print(get_completion(prompt))

'''
自助餐厅一开始有23个苹果。  
他们用掉了20个苹果，所以剩下 $ 23 - 20 = 3 $ 个苹果。
然后他们又买了6个苹果，因此现在总共有 $ 3 + 6 = 9 $ 个苹果。

**答案是：** $\boxed{9}$
'''