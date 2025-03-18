from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import Callbacks
from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.indexes._api import _batch
# from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict
from collections import defaultdict 


def get_model(
        model: str="gpt-4o-mini",
        stream: bool=True,
        callbacks: Callbacks=None, # 很勾八难用
    ) -> ChatOpenAI:
    return ChatOpenAI(model=model, stream=stream)

# 缓存与计算分离的设计，适合需要频繁生成嵌入向量的应用场景
def get_cached_embedder() -> CacheBackedEmbeddings: # 带本地缓存的嵌入模型，生成文本嵌入向量
    fs = LocalFileStore("./.cache/embeddings")
    underlying_embeddings = OpenAIEmbeddings()

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, fs, namespace=underlying_embeddings.model
    )
    return cached_embedder

def get_record_manager(namespace: str="note") -> SQLRecordManager:
    return SQLRecordManager(f"chroma/{namespace}", db_url="sqlite:///note_manager_cache.sql")

def get_vector_store(collection_name: str="note") -> Chroma:
    vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=get_cached_embedder(),
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
    return vector_store

def clear_vector_store(collection_name) -> None:
    record_manager = get_record_manager(collection_name)
    vector_store = get_vector_store(collection_name)

    index([], record_manager, vector_store, cleanup="full", source_id_key="source")

def note_index(docs: List[Document], show_progress: bool = True) -> Dict: # 显示进度条
    info = defaultdict(int) # default为整数dict提供默认值0

    record_manager = get_record_manager("note")
    vectorstore = get_vector_store("note")

    pbar = None
    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(total=len(docs)) # 总进度为文档的数量 len(docs)

    for docs in _batch(10, docs):
        # ***核心函数****，获得当前批次文档索引的统计信息
        # result返回形式:
        # {'num_added': 0, 'num_updated': 0, 'num_skipped': 2, 'num_deleted': 0}
        result = index(
            docs,
            record_manager,
            vectorstore,
            cleanup=None, # 手动清理旧内容
            # cleanup="full", # 删除变更的先前版本和已不存在的原文档
            source_id_key="source",
        )
       
        for k, v in result.items():
            info[k] += v

        if pbar:
            pbar.update(len(docs))

    if pbar:
        pbar.close()

    return dict(info)
