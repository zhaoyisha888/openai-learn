import os
from openai import OpenAI


# 创建OpenAI客户端
client = OpenAI(   
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),   # 与本地环境变量一致
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 链接到模型
)

# 创建对话
completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    # model参数为当前模型名称，可按需要更换模型名称
    model="qwen-plus",  
    # messages参数为用户输入对话内容
    # role参数为角色。system(大模型身份)、user(用户)、assistant(助手)
    #       system ：全局身份，伴随整个复杂场景存在
    #       user ：用户角色给大模型提出的问题
    #       assistant ：多轮对话中用于上下文的记忆维持，把上一次交互的整个内容给assistant，再结合本次提问得出结果
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ]
)

# 返回结果以json形式展示
# print(completion.model_dump_json())
'''
{
    "id": "chatcmpl-268af057-4ec1-9aa0-9f48-dc79982fd08e",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": null,
            "message": {
                "content": " 你好！我是通义千问（Qwen），阿里巴巴集团旗下的超大规模语言模型。我能够回答问题、创作文字，比如写故事、写公文、写邮件、写剧本、逻辑推理、编程等等，还能表达观点，玩游戏等。如果你有任何问题或需要帮助，欢迎随时告诉我！😊",
                "refusal": null,
                "role": "assistant",
                "annotations": null,
                "audio": null,
                "function_call": null,
                "tool_calls": null
            }
        }
    ],
    "created": 1773678061,
    "model": "qwen-plus",
    "object": "chat.completion",
    "service_tier": null,
    "system_fingerprint": null,
    "usage": {
        "completion_tokens": 66,
        "prompt_tokens": 22,
        "total_tokens": 88,
        "completion_tokens_details": null,
        "prompt_tokens_details": {
            "audio_tokens": null,
            "cached_tokens": 0
        }
    }
}
'''

# 需要什么内容就获取那部分
print(completion.choices[0].message.content)
'''
你好！我是通义千问（Qwen），阿里巴巴集团旗下的超大规模语言模型。我能够回答问题、创作文字，比如写故事、写公文、写邮件、写剧本、逻辑推理、编程等等 ，还能表达观点，玩游戏等。如果你有任何问题或需要帮助，欢迎随时告诉我！😊
'''