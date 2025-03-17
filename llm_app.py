from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import requests

load_dotenv()

llm = ChatOpenAI(
    model = "gpt-4o-mini",
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url = "https://api.openai-proxy.org/v1",

)

prompts = ChatPromptTemplate.from_messages([
    ("system", "有用的帮手"),
    ("user", "{input}"),
])

output_parser = StrOutputParser()

chain = prompts | llm | output_parser

result = chain.invoke({"input": "说你好"})
print(result)