from src.utils import get_record_manager, clear_vector_store, note_index
from src.splitter import NoteSplitter
from src.loader import MDloader
from src.chain import get_check_ai_chain, get_note_chain
# from src.callback import AysncCustomHandler
from config import config

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from dotenv import load_dotenv

import sys
import asyncio
from pprint import pprint
import argparse

load_dotenv(override=True)

def init_vectorstore():
    record_manager = get_record_manager("note")
    record_manager.create_schema()

    clear_vector_store("note")

    # 重叠有助于减轻将语句与其相关的重要上下文分离的可能性
    text_splitter = NoteSplitter.from_tiktoken_encoder(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    docs = MDloader(config.NOTE_PATH).load_and_split(text_splitter=text_splitter)
    info = note_index(docs)
    pprint(info)

def run_web() -> None:
    import gradio as gr

    check_note_chain = get_check_ai_chain(config)
    chain = get_note_chain(config, out_callback=None)

    async def chat(message, history):
        # out_callback = CustomAysncHandler()

        is_AI = check_note_chain.ainvoke({"question": message})
        print(is_AI)
        if not is_AI:
            yield "我只接受AI相关问题, 请重新提问。"
            return

        output = await chain.ainvoke({"question": message})
        yield output
        # task = asyncio.create_task(chain.ainvoke({"question": message}))
            # chain.ainvoke({"question": message}, config={"callbacks": [out_callback]}))

        # async for new_token in out_callback.aiter():
        #     pass

        # out_callback.done.clear()

        # response = ""
        # async for new_token in out_callback.aiter():
        #     response += new_token
        #     yield response

        # res = await task
        # for new_token in ["\n\n", res["note_context"], "\n", res["web_context"]]:
        #     response += new_token
        #     yield response

    demo = gr.ChatInterface(
        fn=chat, title="AI小助手")

    demo.queue()
    demo.launch()
        # server_name=config.WEB_HOST, server_port=config.WEB_PORT,
        # auth=(config.WEB_USERNAME, config.WEB_PASSWORD),
        # auth_message="默认用户名密码: username / password")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="please specify only one operate method once time.")
    parser.add_argument(
        "-i",
        "--init",
        action="store_true",
        help=('''
            init vectorstore
        ''')
    )
    parser.add_argument(
        "-w",
        "--web",
        action="store_true",
        help=('''
            run web
        ''')
    )

    if len(sys.argv) <= 1:
        parser.print_help()
        exit()

    args = parser.parse_args()
    if args.init:
        init_vectorstore()
    if args.web:
        run_web()
