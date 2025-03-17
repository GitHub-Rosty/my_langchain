from typing import Any, Dict, List

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain_core.outputs import LLMResult

class CustomAysncHandler(AsyncIteratorCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    # async def on_llm_start(
    #     self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    # ) -> None:
    #     """Run when chain starts running."""
    #     print("zzzz....")
    #     await asyncio.sleep(0.3)
    #     class_name = serialized["name"]
    #     print("Hi! I just woke up. Your llm is starting")

    # async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
    #     """Run when chain ends running."""
    #     print("zzzz....")
    #     await asyncio.sleep(0.3)
    #     print("Hi! I just woke up. Your llm is ending")