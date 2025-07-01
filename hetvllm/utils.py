import subprocess

def run_cmd(cmd):
    try:
        subprocess.check_output(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def detect_gpu_type():
    if run_cmd(['nvidia-smi']):
        return 'nvidia'
    elif run_cmd(['rocm-smi']):
        return 'amd'
    elif run_cmd(['hy-smi']):
        return 'hygon'
    elif run_cmd(['cnmon']):
        return 'cambricon'
    elif run_cmd(['mthreads-gmi']):
        return 'moore'
    elif run_cmd(['npu-smi', 'info']):
        return 'huawei'
    else:
        return 'cpu'
