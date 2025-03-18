from .utils import get_vector_store, get_model
from .retriever import WebRetriever, get_multi_query_retriever
from .prompt import NOTE_PROMPT, CHECK_NOTE_PROMPT, HYPO_QUESTION_PROMPT
from .combine import combine_note_docs, combine_web_docs

from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema.language_model import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import Callbacks
from langchain.chains.question_answering.stuff_prompt import PROMPT_SELECTOR
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.schema import format_document
from langchain.schema import BaseRetriever
from langchain.pydantic_v1 import Field
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import BooleanOutputParser
from langchain.schema.runnable import RunnableMap
from langchain.chains.base import Chain
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from typing import Any, Optional, List
from collections import defaultdict
from operator import itemgetter

# 将多个文档内容合并为一个字符串
class NoteStuffDocumentsChain(StuffDocumentsChain):
    # **kwargs 允许函数接受任意数量的关键字参数，这些参数会被打包成一个字典（dict）
    def _get_inputs(self, docs: List[Document], **kwargs: Any) -> dict:

        note = defaultdict(list)
        web = defaultdict(list)
        for doc in docs:
            metadata = doc.metadata
            if "note" in metadata:
                note[metadata["note"]].append(
                    format_document(doc, self.document_prompt).strip("\n"))
            elif 'link' in metadata:
                web[metadata["title"]].append(
                    format_document(doc, self.document_prompt).strip("\n"))

        str = ""
        for note, page_contents in note.items():
            str += f"《{note}》\n"
            str += "\n".join(page_contents)
            str += "\n\n"

        for web, page_contents in web.items():
            str += f"网页：{web}\n"
            str += "\n".join(page_contents)
            str += "\n\n"

        # {key: value for item in iterable if condition}快速字典生成式
        inputs = {
            k: v
            for k, v in kwargs.items() # 返回键值对列表
            if k in self.llm_chain.prompt.input_variables
        }
        inputs[self.document_variable_name] = str
        return inputs


class NoteQAChain(BaseRetrievalQA):
    vs_retriever: BaseRetriever = Field(exclude=True)
    web_retriever: BaseRetriever = Field(exclude=True)

    def _get_docs(
        self,
        question: str,
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        vs_docs = self.vs_retriever.get_relevant_documents(
            question, callbacks=run_manager.get_child()
        )

        web_docs = self.web_retriever.get_relevant_documents(
            question, callbacks=run_manager.get_child()
        )

        return vs_docs + web_docs

    async def _aget_docs(
        self,
        question: str,
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        vs_docs = await self.vs_retriever.aget_relevant_documents(
            question, callbacks=run_manager.get_child()
        )

        web_docs = await self.web_retriever.aget_relevant_documents(
            question, callbacks=run_manager.get_child()
        )

        return vs_docs + web_docs

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "QA"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseRetrievalQA:
        """Initialize from LLM."""
        _prompt = prompt or PROMPT_SELECTOR.get_prompt(llm)
        llm_chain = LLMChain(llm=llm, prompt=_prompt, callbacks=callbacks)
        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )

        combine_documents_chain = NoteStuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
            callbacks=callbacks,
        )

        return cls(
            combine_documents_chain=combine_documents_chain,
            callbacks=callbacks,
            **kwargs,
        )


def get_check_ai_chain(config: Any) -> Chain:
    model = get_model()

    check_chain = CHECK_NOTE_PROMPT | model | BooleanOutputParser()

    return check_chain


def get_note_chain(config: Any, out_callback: AsyncIteratorCallbackHandler) -> Chain:
    note_vs = get_vector_store("note")
    web_vs = get_vector_store("web")

    vs_retriever = note_vs.as_retriever(search_kwargs={"k": config.NOTE_VS_SEARCH_K})
    web_retriever = WebRetriever(
        vectorstore=web_vs,
        num_search_results=config.WEB_VS_SEARCH_K
    )

    multi_query_retriver = get_multi_query_retriever(vs_retriever, get_model())

    callbacks = [out_callback] if out_callback else []

    chain = ( # 将输入映射到多个输出
        RunnableMap(
            {
                "note_docs": itemgetter("question") | multi_query_retriver,
                'web_docs': itemgetter("question") | web_retriever,
                "question": itemgetter("question")}#lambda x: x["question"]}
        )
        | RunnableMap( # lambda中是上一层RunnableMap的参数
            {
                "note_docs": lambda x: x["note_docs"],
                "web_docs": lambda x: x["web_docs"],
                "note_context": lambda x: combine_note_docs(x["note_docs"]),
                "web_context": lambda x: combine_web_docs(x["web_docs"]),
                "question": lambda x: x["question"]}
        )
        | RunnableMap({
                "note_docs": lambda x: x["note_docs"],
                "web_docs": lambda x: x["web_docs"],
                "note_context": lambda x: x["note_context"],
                "web_context": lambda x: x["web_context"],
                "prompt": NOTE_PROMPT
            }
        )
        | RunnableMap({
            "note_docs": lambda x: x["note_docs"],
            "web_docs": lambda x: x["web_docs"],
            "note_context": lambda x: x["note_context"],
            "web_context": lambda x: x["web_context"],
            "answer": itemgetter("prompt") | get_model(callbacks=callbacks) | StrOutputParser()
        })
    )

    return chain


def get_hypo_questions_chain(config: Any) -> Chain:
    model = get_model()

    functions = [
        {
            "name": "hypothetical_questions",
            "description": "Generate hypothetical questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                    },
                },
                "required": ["questions"]
            }
        }
    ]

    chain = (
        {"context": lambda x: f"《{x.metadata["note"]}》{x.page_content}"}
        | HYPO_QUESTION_PROMPT
        | model.bind(functions=functions, function_call={"name": "hypothetical_questions"})
        | JsonKeyOutputFunctionsParser(key_name="questions")
    )

    return chain
