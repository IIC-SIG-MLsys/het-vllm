# het-vllm
```
python : 3.12

pip install -e .

export VLLM_USE_MODELSCOPE=True # install model with modelscope
python -m hetvllm.entrypoints.api_server --model Qwen/Qwen2-7b --port 8080 --dtype=half --max_model_len 4096 --pipeline_parallel_size 1

cd tests && python test.py
```