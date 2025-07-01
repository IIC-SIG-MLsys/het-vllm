from setuptools import setup, find_packages
from hetvllm.utils import detect_gpu_type

gpu_type = detect_gpu_type()

# 设置依赖版本
deps = []

if gpu_type == 'nvidia':
    deps.append('torch>=2.7.0')
    deps.append('vllm>=0.9.0')
elif gpu_type == 'amd':
    deps.append('torch>=2.6.0')
    deps.append('vllm>=0.9.0')
elif gpu_type == 'hygon':
    # todo
    pass
elif gpu_type == 'cambricon':
    # todo
    pass
elif gpu_type == 'moore':
    # todo
    pass
elif gpu_type == 'huawei':
    # todo
    pass
else:
    deps.append('torch>=2.7.0+cpu')
    deps.append('vllm>=0.9.0')

setup(
    name='hetvllm',
    version='0.1.0',
    packages=find_packages(),
    install_requires=deps,
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
