from vllm.logger import init_logger
from vllm.engine.async_llm_engine import AsyncLLMEngine, _AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs as vllmAsync

class AsyncEngineArgs(vllmAsync):
    pass
