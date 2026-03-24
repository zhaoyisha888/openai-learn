#学会使用提示词模版
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    PromptTemplate, AIMessagePromptTemplate

from models import get_lc_model_client

#获得访问大模型客户端
client = get_lc_model_client()

# 1. 定义模板字符串
# 其中的 {text} 是一个“占位符”，可以在使用时被替换成真实内容
template_str = "您是一位专业的程序员。\n对于信息 {msg} 进行简短描述"
fact_text = "langchain"

# 2. 创建提示模板（方法一）
# 通过 from_template 自动识别模板中的占位符（如 {text}）
prompt = PromptTemplate.from_template(template_str)

# 3. 创建提示模板（方法二）
# 显式指定模板中用到的变量名，更清晰但稍显啰嗦
prompt2 = PromptTemplate(
    input_variables=["msg"],  # 明确告诉模板：这里有一个叫 msg 的变量需要填
    template=template_str
)

# 4. 使用模板：将占位符 {msg} 替换成真实内容
# 两种方法创建出来的 prompt 对象功能完全一样，使用方式也一样，只是创建方式不同
# 说明：
# 这里的 fact_text 可以从任何地方获取，比如：
# - 用户在网页上输入的内容
# - 数据库中查询到的信息
# - 文件中读取的数据
# 这使得模板可以复用，内容可以动态变化
# print(prompt.format(msg="langchain框架"))  # 也可以直接硬编码写死内容
print(prompt.format(msg=fact_text))
# print(prompt2.format(msg=fact_text))

# 5. 将组装好的提示词发送给大模型（如 LangChain 中的 client）
print(client.invoke(prompt.format(msg=fact_text)))

print()




# 聊天模板
# 和普通文本模板不同，聊天模板专门用于构造“多轮对话”场景
# 每条消息都带有“角色”：系统、用户、助手，让大模型更清楚自己的身份和任务

# 1. 创建聊天模板
chat_template = ChatPromptTemplate.from_messages([
    # 系统消息：设定大模型的“人设”或“全局指令”
    # 这里告诉模型：“你要做翻译，目标语言由用户指定”
    ('system', "请将以下的内容翻译成{language}"),
    
    # 用户消息：用户实际要问的内容
    # HumanMessagePromptTemplate 表示这是一条“用户”发的消息
    HumanMessagePromptTemplate.from_template("{text}"),
    # ('human', "{text}") 是简写方式，效果相同
])

# 2. 使用模板：替换占位符，生成完整的对话上下文
# 第一次调用：翻译成英文
print(client.invoke(chat_template.format(language="英文", text="你好，今天的天真蓝")))
# 第二次调用：翻译成法文，模板复用，只需替换不同的参数
print(client.invoke(chat_template.format(language="法文", text="你好，今天的天真蓝")))

