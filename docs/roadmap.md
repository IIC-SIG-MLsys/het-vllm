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
TODO：
- vllm ray不同版本出现适配问题，nvidia,mlu：wangxu，haoyuan
- qingheng huawei环境排查问题
- zhiseng，yangshu 1. 异构torch能不能成组（如果依赖nccl的话可能不行，测试寒武纪和英伟达）；2. 设计一个类似flagCX的分层CCL，位置位于hetvllm/transport层
- baishuai：rpc实现应用到分层CCL的异构节点传输之间
- guoliang：swiftvllm scheduler外的其余部分

日期：2025-07-10 10:02:22
会议录制文件：https://meeting.tencent.com/crm/2BjMVEzX70