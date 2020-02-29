from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import src.main.singleton as singleton
from runs.runs_manager import load_config
from configuration import config


class SyncedCounter:
    def __init__(self, i=0):
        ctx = mp.get_context('spawn')
        self._val = ctx.Manager().Value('i', i)
        self._lock = ctx.Lock()

    value = property(lambda self: self._val.value)

    def incr(self) -> int:
        """Increments the counter by 1 and returns the value before it was incremented"""
        with self._lock:
            result = self._val.value
            self._val.value += 1

        return result


def init_process(generation, run_name: str, proc_counter: SyncedCounter):
    """Because of how python spawns processes config and singleton need to be re-initialized on each new process"""
    singleton.instance = generation
    load_config(run_name)
    mp.current_process().name = str(proc_counter.incr())


def get_bp_eval_pool(generation) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(config.n_gpus * config.n_evals_per_gpu,
                               mp.get_context('spawn'),
                               initializer=init_process,
                               initargs=(generation, config.run_name, SyncedCounter()))
