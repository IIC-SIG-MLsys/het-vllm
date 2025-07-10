# het-vllm

- 扩展优化 vllm已有的 Tensor 并行和 Pipeline 并行分布式推理，以更好支持异构gpu协同推理
- 在官方支持 NVIDIA GPU、AMD CPU 和 GPU、Intel CPU 和 GPU、PowerPC CPU、TPU 和 AWS Neuron的基础上，兼容国产gpu自有的特定vllm版本

1. AsyncLLMEngine, AsyncEngineArgs -> HetvllmEngine, HetvllmEngineArgs
2. LLMEngine -> AsyncLLMEngine -> HetvllmEngine
    MQLLMEngine 
    Scgeduling -> MolinkScheduler
    Model Execution -> MolinkMultiprocessingDistributedExecutor
    Worker(1 Model runner) -> MolinkWorker
    Model runner -> MolinkGPUModelRunner
        model loader -> MolinkXXXModelLoader 

## Meeting Notes: 
#### 250703
Todo：
- nvidia haoyuan 0.9.1
- mlu wangxu 官方最新的vllm的容器
- huawei qingheng 官方最新的vllm的容器
- zhiseng(ray conntor) + baishuai (rpc)

#### 250710
1. pp size 是用来索引的？
2. 规划步骤后，是把request rpc 远程的worker？ ctx存的是future