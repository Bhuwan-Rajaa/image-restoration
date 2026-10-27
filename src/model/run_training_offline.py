"""Launch train_v2 with cached weights and keep Windows awake during the run.

Accepts the same arguments as train_v2.py, including --resume.
"""
import ctypes
import os
from pathlib import Path
import runpy
import sys


def main():
    # Small CPU thread pools avoid oversubscription alongside loader workers.
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    import torch

    torch.set_num_threads(2)

    def cached_weights_only(*args, **kwargs):
        raise RuntimeError('A pretrained weight is missing from the local cache; '
                           'offline training never downloads weights.')

    torch.hub.download_url_to_file = cached_weights_only
    awake = False
    if os.name == 'nt':
        awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        print(f'Preventing automatic system sleep: {awake}', flush=True)
    print(f'Offline training launcher PID={os.getpid()}; CPU threads=2; cached weights only', flush=True)
    entrypoint = Path(__file__).with_name('train_v2.py')
    sys.path.insert(0, str(entrypoint.parent))
    try:
        runpy.run_path(str(entrypoint), run_name='__main__')
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
