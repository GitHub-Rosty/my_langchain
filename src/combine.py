from typing import List
from collections import defaultdict

from langchain.docstore.document import Document


def combine_note_docs(docs: List[Document]) -> str:
    note = defaultdict(list)
    for doc in docs:
        metadata = doc.metadata
        if '"note"' in metadata:
            note[metadata["note"]].append(doc)

    str = ""
    for note, docs in note.items():
        str += f"相关笔记内容：《{note}》\n"
        str += "\n".join([doc.page_content.strip("\n") for doc in docs])
        str += "\n"

    return str


def combine_web_docs(docs: List[Document]) -> str:
    web_str = ""
    for doc in docs:
        web_str += f"相关网页：{doc.metadata['title']}\n"
        web_str += f"网页地址：{doc.metadata['link']}\n"
        web_str += doc.page_content.strip("\n") + "\n"
        web_str += "\n"

    return web_str
