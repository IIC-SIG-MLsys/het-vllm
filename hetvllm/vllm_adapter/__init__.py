from hetvllm.utils import detect_gpu_type

gpu_type = detect_gpu_type()

if gpu_type == 'hygon':
    from .hygon import *
elif gpu_type == 'cambricon':
    from .cambricon import *
elif gpu_type == 'moore':
    from .moore import *
elif gpu_type == 'huawei':
    from .ascend import *
else:
    from vllm.logger import init_logger
    # from vllm.core.scheduler import Scheduler
    from vllm.engine.async_llm_engine import AsyncLLMEngine, _AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs