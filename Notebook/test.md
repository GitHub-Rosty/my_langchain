# 解析并改造“基于langchain的深度学习RAG“

## 项目介绍

1. 采用本地缓存的嵌入模型CacheBackedEmbeddings，分离缓存与计算，适合频繁生成文本向量的场景
2. 

## 主函数

- init_vectorstore进行数据预处理，构建ds知识库.
  
  1. get_record_manager创建记录管理器，初始化数据库表结构并清空已有数据
  2. 创建文本分割器
  3. 从配置的路径加载笔记,并分割成小块docs
  4. 将docs索引到向量数据库中
  5. pprint美观输出

- run_shell 提供命令行交互

    1. 加载 check_law_chain 用于判断用户问题是否与法律相关。
    2. 加载 get_law_chain 用于生成回答，并绑定 OutCallbackHandler 实现流式输出。
    3. 循环：
   
        调用 check_law_chain 检查问题是否合法

        创建Task对象添加到事件循环, 使得可以异步进行

        利用 OutCallbackHandler 实时流式输出回答片段
  

- run_web 提供图形界面 同理

## chain.py

