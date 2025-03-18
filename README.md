# 解析并改造“基于langchain的深度学习RAG“

## 项目介绍

1. 采用本地缓存的嵌入模型CacheBackedEmbeddings，分离缓存与计算，适合频繁生成文本向量的场景
2. MultiQueryRetriever实现对于用户输入生成多个不同视角的查询，帮助自动化提示调优。
   详情见<https://www.langchain.com.cn/docs/how_to/MultiQueryRetriever/>
3. 

## ~~langchain~~

~~这么混乱的结构指不上唯手熟尔了，还是写个纲大题介绍一下。~~
算了还是直接另开一个文档介绍吧

### 可优化部分

- retriever
  - 如何对现存vectorstore进行检索:用as_retriever转换
  - 如何对网络上信息进行检索:duckduckgo
- parser
  - 如何对LLM输出结果进行解析:自定义OutputParser生成结构化对象

### Chrome

- Chrome的集合存储embedding, docu等，支持索引和AI检索过滤(语意距离)

### LCEL构造链

- 无LCEL
  
  ```python
  prompt_template = "Tell me a short joke about {topic}"
  client = openai.OpenAI()

  def call_chat_model(messages: List[dict]) -> str:
      response = client.chat.completions.create(
          model="gpt-3.5-turbo", 
          messages=messages,
      )
      return response.choices[0].message.content

  def invoke_chain(topic: str) -> str:
      prompt_value = prompt_template.format(topic=topic)
      messages = [{"role": "user", "content": prompt_value}]
      return call_chat_model(messages)

  invoke_chain("ice cream")
  ```

- 有LCEL

  ```python
  prompt = ChatPromptTemplate.from_template(
      "Tell me a short joke about {topic}"
  )
  output_parser = StrOutputParser()
  model = ChatOpenAI(model="gpt-3.5-turbo")
  chain = (
      {"topic": RunnablePassthrough()} 
      | prompt
      | model
      | output_parser
  )

  chain.invoke("ice cream")
  ```

## 主函数

- init_vectorstore进行数据预处理，构建ds知识库.

  调用utils, splitter, loader  
  1. get_record_manager创建记录管理器，用于管理文档的索引记录
  2. 初始化数据库表结构，clear_vector_store清空已有数据
  3. NoteSplitter创建文本分割器
  4. Loader从配置的路径加载笔记,并分割成小块docs
  5. note_index将docs索引到向量数据库中,返回本次统计过程的反馈信息
  6. pprint美观输出

- run_web 提供图形界面交互

  调用chain, retriever
    1. 加载 check_note_chain 用于判断用户问题是否与笔记相关。
    2. 加载 get_note_chain 用于生成回答，并绑定 OutCallbackHandler 实现流式输出。
    3. 循环
    - 调用 check_note_chain 检查问题是否合法
    - 创建Task对象添加到事件循环, 使得可以异步进行
    - 利用 OutCallbackHandler 实时流式输出回答片段
  
- run_shell 命令行 同理

## chain.py

- check_note_chain通过prompt, gpt在bool解析下判断是否为AI相关问题
- get_note_chain定义了本地、网络，多查询检索器。并通过四层RunnableMap将任务分解为数据检索、上下文生成、提示词生成、答案生成。
  - 第一层

### `as_retriever`方法

将`VectorStore`对象转换为`VectorStoreRetriever`对象,用于从向量存储中查找和检索最相关的文档.

- search_type
  - "similarity" 默认最相关
  - "mmr" 适合多样性
  - "similarity_score_threshold" 相似度阈值
- search_kwargs: dict{}
  - "k": 返回文档数
  - "score_threshold": 相似度最低值
  - "filter": 过滤器，如文档标题等

## 正在做

## 未完成
