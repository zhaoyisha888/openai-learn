#思维树(Tree-of-thought, ToT)
# pip install openai
#提示词在大模型应用中直接使用，也可以看到效果
from openai import OpenAI
import os

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
client = OpenAI(api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)

prompt = """
小明100米跑成绩：10.5秒，1500米跑成绩：3分20秒，铅球成绩：12米。他适合参加哪些搏击运动训练?

请根据以上成绩，分析候选人在速度、耐力、力量三方面素质的分档。分档包括：强（3），中（2），弱（1）三档

需要速度强的运动有哪些。给出10个例子,需要耐力强的运动有哪些。给出10个例子,需要力量强的运动有哪些。给出10个例子

分别分析上面给的10个运动对速度、耐力、力量方面素质的要求: 强（3），中（2），弱（1）

根据上面的分析：生成一篇小明适合那种运动训练的分析报告
"""

def get_completion(prompt, model="qwen-max-latest"):
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
    )
    return response.choices[0].message.content
print(get_completion(prompt))