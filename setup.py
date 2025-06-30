import subprocess
import sys
from setuptools import setup, find_packages

def run_cmd(cmd):
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def detect_gpu_type():
    if run_cmd(['nvidia-smi']):
        return 'nvidia'
    elif run_cmd(['rocm-smi']):
        return 'amd'
    elif run_cmd(['hy-smi']):
        return 'haiguang'
    elif run_cmd(['cnmon']):
        return 'cambricon'
    elif run_cmd(['mthreads-gmi']):
        return 'mthreads'
    elif run_cmd(['npu-smi', 'info']):
        return 'huawei'
    else:
        return 'cpu'

gpu_type = detect_gpu_type()

# 设置依赖版本
torch_dep = None
vllm_dep = None

if gpu_type == 'nvidia':
    torch_dep = 'torch>=2.1.0'
    vllm_dep = 'vllm>=0.3.0'
elif gpu_type == 'amd':
    torch_dep = 'torch>=2.1.0+rocm'  # 用户需手动指定正确的rocm源安装
    vllm_dep = 'vllm[cpu]>=0.3.0'    # vLLM 可能尚未支持 ROCm
elif gpu_type == 'haiguang':
    torch_dep = 'torch>=2.1.0+cpu'
    vllm_dep = 'vllm[cpu]>=0.3.0'
elif gpu_type == 'cambricon':
    torch_dep = 'torch==custom_cambricon'  # 你可以替换为具体包或从 cambricon repo 安装
    vllm_dep = 'vllm[cpu]>=0.3.0'
elif gpu_type == 'mthreads':
    torch_dep = 'torch==custom_mthreads'  # 同上，自行提供适配版本
    vllm_dep = 'vllm[cpu]>=0.3.0'
elif gpu_type == 'huawei':
    torch_dep = 'torch==custom_ascend'  # Ascend 自定义 PyTorch（CANN）版本
    vllm_dep = 'vllm[cpu]>=0.3.0'
else:  # cpu fallback
    torch_dep = 'torch>=2.1.0+cpu'
    vllm_dep = 'vllm[cpu]>=0.3.0'

setup(
    name='hetvllm',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        torch_dep,
        vllm_dep,
        # 你可以添加其他依赖，如 transformers 等
    ],
    include_package_data=True,
    description='Flexible and Efficient Heterogeneous vLLM',
    author='SDU-IIC',
    author_email='',
    url='',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
