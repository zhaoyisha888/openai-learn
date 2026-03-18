
## 大模型相关概念（建议先搜索了解）

1. 什么是人工智能       --- 图灵测试
2. 人工智能 --- 机器学习--- 深度学习 --- 大模型
3. 人工智能 AIGC
4. 特点：大语言 LLM
5. 常见模型： 通义千问、DeepSeek、字节的豆包、文心一言
6. 大模型名称含义    7B -- 参数    1M -- 上下文   QwQ推理模型    QvQ视频图片
7. 硬件配置    cpu   gpu 
8. 行业应用/发展/挑战


## 开发环境配置

### 获取api key

通义千问接口文档[https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api/?type=model&url=2712195]

DeepSeek接口文档[https://api-docs.deepseek.com/zh-cn/]

### 配置API Key到环境变量

[https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api/?type=model&url=2803795]
建议使用全局变量，方便后续使用（用户变量也可以）

DeepSeek API Key入口 [https://platform.deepseek.com/api_keys]

### 安装 SDK

建议先创建一个专属的虚拟开发环境，方便后续管理安装包

```bash
python -m venv .openai
```

激活环境
```bash
.openai\scripts\activate
```

进入该环境后在终端中运行以下命令安装SDK

```bash
# 如果运行失败，可以将pip替换成pip3再运行
pip install -U openai
```

**虚拟环境的包管理**

查看环境里的包
```bash
# 查看已安装的包列表
pip list

# 查看已安装的包列表（携带版本号）
pip freeze
```

创建一个 requirements.txt 文件，方便在不同平台移植安装包（记得安装新包后更新此文件）
```bash
# 生成 requirements.txt 文件
pip freeze > requeirments.txt

# 使用requirements.txt 安装包
pip install -r requirements.txt
```


## 大模型调用

### 创建例程运行代码

参考阿里云百炼平台官方文档：[https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api/?type=model&url=3016807]

>本项目创建了一个例程demo.py，可以直接运行尝试，返回标准输出如下，**需要根据实际需求取部分返回信息**。
```json
{"id":"chatcmpl-268af057-4ec1-9aa0-9f48-dc79982fd08e","choices":[{"finish_reason":"stop","index":0,"logprobs":null,"message":{"content":" 你好！我是通义千问（Qwen），阿里巴巴集团旗下的超大规模语言模型。我能够回答问题、创作文字，比如写故事、写公文、写邮件、写剧本、逻辑推理、编程等等，还能表达观点，玩游戏等。如果你有任何问题或需要帮助，欢迎随时告诉我！😊","refusal":null,"role":"assistant","annotations":null,"audio":null,"function_call":null,"tool_calls":null}}],"created":1773678061,"model":"qwen-plus","object":"chat.completion","service_tier":null,"system_fingerprint":null,"usage":{"completion_tokens":66,"prompt_tokens":22,"total_tokens":88,"completion_tokens_details":null,"prompt_tokens_details":{"audio_tokens":null,"cached_tokens":0}}}
```

可以看到返回的json数据，有点乱，可以复制到**爬虫工具库** [https://spidertools.cn/#/] 整理一下
```json
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
```

整理后可以看到，返回信息包含id、choices、created、model、object、usage等信息。
- choices是一个列表，包含一个字典，字典中包含finish_reason、index、logprobs、message等信息。
  - message是一个字典，包含content、refusal、role、annotations、audio、function_call、tool_calls等信息。
        - content是返回的文本内容，
        - refusal是拒绝回答的原因，
        - role是返回的角色，
        - annotations是返回的注释，
        - audio是返回的音频，
        - function_call是返回的函数调用，
        - tool_calls是返回的工具调用。
- created创建时间，
- model是模型名称，
- object是返回的对象，
- service_tier是服务层，
- system_fingerprint是系统指纹，
- usage是使用情况，包含completion_tokens、prompt_tokens、total_tokens、completion_tokens_details、prompt_tokens_details等信息。
  - completion_tokens是完成 tokens 数，
  - prompt_tokens是提示 tokens 数，
  - total_tokens是总 tokens 数，
  - completion_tokens_details是完成 tokens 详情，
  - prompt_tokens_details是提示 tokens 详情。

### 更换模型

目前模型名称：qwen-plus

可以在模型列表查看：[https://help.aliyun.com/zh/model-studio/getting-started/models]

更推荐 阿里云百炼模型广场：[https://bailian.console.aliyun.com/cn-beijing?spm=a2c4g.11186623.0.0.60917f21Nm09IM&tab=model#/model-market/all]