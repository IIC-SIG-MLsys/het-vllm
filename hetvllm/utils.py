import subprocess

def run_cmd(cmd, xpu):
    try:
        #subprocess.check_output(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        if xpu in result.decode():  # 判断输出中是否包含 MLU
            return True
        else:
            return False
    except Exception:
        return False


def detect_gpu_type():

    # 使用 subprocess 检查 cnmon 输出并判断是否包含特定的 GPU 名称
    if run_cmd(['nvidia-smi'], 'NVIDIA'):
        return 'nvidia'
    elif run_cmd(['rocm-smi'], 'amd'):
        return 'amd'
    elif run_cmd(['hy-smi'], 'SMI'):
        return 'hygon'
    elif run_cmd(['/usr/bin/cnmon'], 'MLU'):
        return 'cambricon'
    elif run_cmd(['mthreads-gmi'], 'MTT'):
        return 'moore'
    elif run_cmd(['npu-smi', 'info'], 'npu-smi'):
        return 'huawei'
    else :
        return 'cpu'

# gpu =  detect_gpu_type()
# print(f"Detected GPU type: {gpu}")
