#学会使用提示词模版
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    PromptTemplate, AIMessagePromptTemplate

from models import get_lc_model_client

#获得访问大模型客户端
client = get_lc_model_client()

#原始的文字模版，其中用{}的部分是占位符，可以在运行时动态替换
template_str = "您是一位专业的程序员。\n对于信息 {text} 进行简短描述"
fact_text = "langchain"

#聊天模版
#接收聊天消息/聊天消息列表
#消息分角色:系统消息，用户消息，助手消息(大模型的应答消息)
chat_template = ChatPromptTemplate.from_messages(
    [
        #用SystemMessagePromptTemplate来实现可以
        ('system',"请将以下的内容翻译成{language}"),
        HumanMessagePromptTemplate.from_template("{text}")
        #('human',"{text"),
    ]
)

parser = StrOutputParser()   # 创建一个输出解析器，提取 context 字符串内容

# 链式调用，从左到右执行
chain = chat_template | client | parser

#接收的字典类型
print(chain.invoke({'text': '你好，我是超级亚赛人', 'language': '英文'}))


