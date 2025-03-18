from langchain.prompts import PromptTemplate

note_prompt_template = """你是一个我的学习好帮手，你可以通过现有知识库{note_context}和网络资源{web_context}回答问题: {question}"""

NOTE_PROMPT = PromptTemplate(
    template=note_prompt_template, input_variables=["note_context", "web_context", "question"]
)

check_note_prompt_template = """判断提出的问题是否为深度学习、AI相关。如果你认为是则只回答YES, 认为不是则只回答NO. 不允许其它回答, 你的回答只应为YES或NO。
问题: {question}
"""

CHECK_NOTE_PROMPT = PromptTemplate(
    template=check_note_prompt_template, input_variables=["question"]
)

hypo_questions_prompt_template = """生成 5 个假设问题的列表，以下文档可用于回答这些问题:\n\n{context}"""

HYPO_QUESTION_PROMPT = PromptTemplate(
    template=hypo_questions_prompt_template, input_variables=["context"]
)


multi_query_prompt_template = """您是 AI 语言模型助手。您的任务是生成给定用户问题的3个不同版本，以从矢量数据库中检索相关文档。通过对用户问题生成多个视角，您的目标是帮助用户克服基于距离的相似性搜索的一些限制。提供这些用换行符分隔的替代问题，不要给出多余的回答。问题：{question}""" # noqa
MULTI_QUERY_PROMPT_TEMPLATE = PromptTemplate(
    template=multi_query_prompt_template, input_variables=["question"]
)
