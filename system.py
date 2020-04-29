import os
import psutil
from gpuinfo import GPUInfo

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.2f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def get_resources_usage(with_gpu=False):
    this_process = psutil.Process(os.getpid())
    memory = sizeof_fmt(this_process.memory_info().rss)
    cpu = sizeof_fmt(this_process.cpu_percent(interval=1))
    res = {'memory':memory,
            'cpu':cpu}
    if with_gpu:
        pid_list,percent,memory,gpu_used=GPUInfo.get_info()
        try:
            res['gpu'] = str(sizeof_fmt(gpu_used[0]))+'/'+str(sizeof_fmt(memory[0]))
        except:
            res['gpu'] = sizeof_fmt(0)
    return res