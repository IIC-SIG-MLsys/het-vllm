from hetvllm.vllm_adapter import AsyncEngineArgs
from hetvllm.vllm_adapter import init_logger

logger = init_logger(__name__)

class HetvllmEngineArgs(AsyncEngineArgs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        