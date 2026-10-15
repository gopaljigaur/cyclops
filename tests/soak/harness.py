"""Runs a scenario many times and counts what it leaves behind."""

import asyncio
import concurrent.futures.thread
import gc
import os
import subprocess
import sys
import threading
import tracemalloc
from collections import Counter
from dataclasses import astuple, dataclass
from typing import Awaitable, Callable, Dict

from cyclops.core import sync

RUNS = 1000
# On Python 3.10 LiteLLM fills caches of its own over about the first thousand runs.
WARMUP = 1000 if sys.version_info < (3, 11) else 100

# Thresholds of #89, on counts only.
MAX_TRACED_BYTES_PER_RUN = 200
MAX_TYPE_GROWTH = 50


@dataclass
class Resources:
    tasks: int
    threads: int
    open_files: int
    children: int

    def within(self, baseline: "Resources") -> bool:
        return all(n <= b for n, b in zip(astuple(self), astuple(baseline)))


@dataclass
class Soak:
    traced_bytes_per_run: float
    grown_types: Dict[str, int]
    cyclic_garbage: int
    before: Resources
    after: Resources

    def assert_steady(self) -> None:
        """Memory, object counts and resources held steady over the runs."""
        assert self.traced_bytes_per_run < MAX_TRACED_BYTES_PER_RUN, self
        assert not self.grown_types, self
        assert self.after.within(self.before), self


def type_counts() -> Counter:
    gc.collect()
    return Counter(type(o).__qualname__ for o in gc.get_objects())


async def _tasks() -> int:
    return len(asyncio.all_tasks()) - 1


def resources() -> Resources:
    tasks = len(asyncio.all_tasks()) - 1
    if sync._loop is not None:
        tasks += asyncio.run_coroutine_threadsafe(_tasks(), sync._loop).result()
    pgrep = subprocess.run(
        ["pgrep", "-P", str(os.getpid())], capture_output=True, text=True
    )
    # Thread pools add workers up to their cap on their own; count the rest.
    pool_workers = sum(t.is_alive() for t in concurrent.futures.thread._threads_queues)
    return Resources(
        tasks=tasks,
        threads=threading.active_count() - pool_workers,
        open_files=len(os.listdir("/dev/fd")),
        children=len(pgrep.stdout.split()),
    )


async def settle() -> None:
    """Wait until background work, such as logging threads, stops changing objects."""
    count = -1
    for _ in range(50):
        await asyncio.sleep(0.1)
        gc.collect()
        count, previous = len(gc.get_objects()), count
        if count == previous:
            return


async def soak(
    run: Callable[[], Awaitable[object]], runs: int = RUNS, warmup: int = WARMUP
) -> Soak:
    """Warm up, then do runs with the cyclic collector off and measure."""
    type_counts()  # The first count fills caches of its own.
    for _ in range(warmup):
        await run()
    await settle()
    before = resources()
    types_before = type_counts()
    tracemalloc.start()
    gc.disable()
    try:
        for _ in range(runs):
            await run()
        await asyncio.sleep(0.1)
        cyclic_garbage = gc.collect()
        gc.enable()
        # Also lets the loop run what collecting finalized, such as generator closes.
        await settle()
        traced = tracemalloc.get_traced_memory()[0]
    finally:
        gc.enable()
        tracemalloc.stop()
    types_after = type_counts()
    return Soak(
        traced_bytes_per_run=traced / runs,
        grown_types={
            name: grown
            for name, count in types_after.items()
            if (grown := count - types_before[name]) > MAX_TYPE_GROWTH
        },
        cyclic_garbage=cyclic_garbage,
        before=before,
        after=resources(),
    )
