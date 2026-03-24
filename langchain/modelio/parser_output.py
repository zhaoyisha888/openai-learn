#学会使用提示词模版
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    PromptTemplate, AIMessagePromptTemplate

from models import get_lc_model_client

#获得访问大模型客户端
client = get_lc_model_client()

#聊天模版
#接收聊天消息/聊天消息列表
#消息分角色:系统消息，用户消息，助手消息(大模型的应答消息)
chat_template = ChatPromptTemplate.from_messages(
    [
        #用SystemMessagePromptTemplate来实现可以
        ('system',"请将以下的内容翻译成{language}"),
        HumanMessagePromptTemplate.from_template("{text}")
        #('human',"{text}"),
    ]
)

#print(client.invoke(chat_template.format(language="英文", text="你好，今天的天真蓝")))
#print(client.invoke(chat_template.format(language="法文", text="你好，今天的天真蓝")))


result = client.invoke(chat_template.format(language="法文", text="你好，今天的天真蓝"))
parser = StrOutputParser()
print(parser.invoke(result))
'''
调用大模型，获取原始响应，获得的是一个复杂的结构体
创建专门用于提取字符串内容的解析器
  StrOutputParser 是 LangChain 提供的工具，作用是从大模型的响应对象中只提取文本内容 content 字段
运行解析器，解析原始响应，提取纯文本获得字符串结果
'''


