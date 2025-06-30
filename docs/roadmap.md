# het-vllm

- 扩展优化 vllm已有的 Tensor 并行和 Pipeline 并行分布式推理，以更好支持异构gpu协同推理
- 在官方支持 NVIDIA GPU、AMD CPU 和 GPU、Intel CPU 和 GPU、PowerPC CPU、TPU 和 AWS Neuron的基础上，兼容国产gpu自有的特定vllm版本

TODO：
- monitor：监控异构资源情况：GPU计算利用率，显存利用率，内存情况
- policy：负责根据资源情况、网络情况和模型需求情况，生成最优的调度策略
- scheduler：在
    - runner：将模型运行器调度到相应节点，并启动，基于ray actor
- runner: 基于vllm实现的模型运行器，运行指定层数的模型

NOT TODO：
- scheduler：
    - resource：根据调度策略，将所需模型文件等资源调度到对应服务器上（共享文件/传输）
- transport：
    - 优化异构场景的vllm kvtrans
    - 资源调度传输优化