
## 环境配置

### 需要安装的库

```python

# 手动安装方法（太麻烦并不建议）
pip install openai
pip install langchain
pip install langchain-core
pip install langserve
pip install langchain-openai
pip install langchain-community
pip install chromadb
```

### 命令汇总
```python
# 创建一个虚拟环境
python -m venv .rag

# 激活环境
.rag/scripts/activate

# 查看已安装的包列表
pip list

# cd /当前项目工作目录/，找到对应的 requirements.txt 文件

# 使用requirements.txt 安装包
pip install -r requirements.txt

# 根据最新的环境包状态生成 requirements.txt 文件
pip freeze > requirements.txt

# 退出虚拟环境
deactivate
```

