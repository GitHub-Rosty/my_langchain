# coding: utf-8
from typing import List

from langchain.schema.vectorstore import VectorStore
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.pydantic_v1 import Field, BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.chains import LLMChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from duckduckgo_search.exceptions import DuckDuckGoSearchException

from .prompt import MULTI_QUERY_PROMPT_TEMPLATE


class WebRetriever(BaseRetriever):
    
    vectorstore: VectorStore = Field(
        ..., description="Vector store for storing web pages"
    )

    search: DuckDuckGoSearchAPIWrapper = Field(..., description="DuckDuckGo Search API Wrapper")
    num_search_results: int = Field(1, description="Number of pages per Google search")

    text_splitter: TextSplitter = Field(
        RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50),
        description="Text splitter for splitting web pages into chunks",
    )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:

        try:
            results = self.search.results(query, self.num_search_results)
        except DuckDuckGoSearchException:
            results = []

        docs = []
        for res in results:
            docs.append(Document(
                page_content=res["snippet"],
                metadata={"link": res["link"], "title": res["title"]}
            ))

        docs = self.text_splitter.split_documents(docs)

        return docs


# 将LLM输出按行分割
class LineList(BaseModel):
    # 通过BaseModel定义数据模型
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


def get_multi_query_retriever(retriever: BaseRetriever, model: BaseModel) -> BaseRetriever:
    output_parser = LineListOutputParser()

    llm_chain = LLMChain(llm=model, prompt=MULTI_QUERY_PROMPT_TEMPLATE, output_parser=output_parser)

    retriever = MultiQueryRetriever(
        retriever=retriever, llm_chain=llm_chain, parser_key="lines"
    )

    return retriever
