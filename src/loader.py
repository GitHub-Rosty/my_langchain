# coding: utf-8
from typing import Any
from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader


class MDloader(DirectoryLoader):
    def __init__(self, path: str, **kwargs: Any) -> None:
        loader_cls = TextLoader # 不解析markdown标题头
        glob = "**/*.md"
        super().__init__(path, loader_cls=loader_cls, glob=glob, **kwargs)

