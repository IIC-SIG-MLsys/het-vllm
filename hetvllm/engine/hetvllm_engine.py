from typing import Type
from hetvllm.vllm_adapter import AsyncLLMEngine, _AsyncLLMEngine
from hetvllm.vllm_adapter import init_logger

logger = init_logger(__name__)

class _HetvllmEngine():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class HetvllmEngine(AsyncLLMEngine):
    _engine_class: Type[_HetvllmEngine] = _HetvllmEngine

    def __init__(self, *args, **kwargs):
        super.__init__(self, *args, **kwargs)
        
