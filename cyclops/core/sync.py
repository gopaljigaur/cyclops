"""Run the async Agent API from sync code on one shared event loop thread."""

import asyncio
import os
import queue
import threading
from contextlib import aclosing
from typing import Any, AsyncGenerator, Awaitable, Callable, Iterator, Optional, TypeVar

_DONE = object()
T = TypeVar("T")

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_lock = threading.Lock()


def _reset_after_fork() -> None:
    global _loop, _loop_lock
    _loop = None
    _loop_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    # A forked child inherits the loop but not the thread running it.
    os.register_at_fork(after_in_child=_reset_after_fork)


def _start_task(make: Callable[[], Awaitable], finished: threading.Event):
    """Start make() as a task on the shared loop; finished is set once it unwinds.

    One long-lived loop in a daemon thread serves every sync call, so LiteLLM's
    background logging can finish and a loop the caller runs is never blocked.
    That logging serves one loop at a time, so mixing sync and async calls in
    one process moves it between this loop and the caller's on each switch.
    """
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, daemon=True).start()
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is _loop:
        # Blocking here would wait on the loop this call is running on.
        raise RuntimeError("Use the async Agent methods from async code")

    def on_done(_: asyncio.Future) -> None:
        finished.set()

    async def start() -> asyncio.Future:
        task = asyncio.ensure_future(make())
        task.add_done_callback(on_done)
        return task

    return asyncio.run_coroutine_threadsafe(start(), _loop).result()


def _wait(task: asyncio.Future, finished: threading.Event) -> None:
    """Wait for a task; if interrupted (e.g. Ctrl-C), cancel it and let it unwind."""
    try:
        finished.wait()
    finally:
        if not finished.is_set():
            task.get_loop().call_soon_threadsafe(task.cancel)
            finished.wait()


def _run_coroutine_sync(fn: Callable[..., Awaitable], *args: Any, **kwargs: Any) -> Any:
    """Await fn(*args, **kwargs) from sync code on the shared loop."""
    finished = threading.Event()
    task = _start_task(lambda: fn(*args, **kwargs), finished)
    _wait(task, finished)
    return task.result()


def _iterate_sync(agen: AsyncGenerator[T, None]) -> Iterator[T]:
    """Drive an async generator from sync code in one task on the shared loop.

    One task keeps the generator's context across chunks. The task fetches a
    chunk only when the caller asks for one, like `async for` would.
    """
    chunks: queue.Queue = queue.Queue()
    demand = asyncio.Semaphore(0)
    finished = threading.Event()

    async def pump() -> None:
        try:
            async with aclosing(agen):
                while True:
                    await demand.acquire()
                    try:
                        chunks.put(await agen.__anext__())
                    except StopAsyncIteration:
                        return
        finally:
            chunks.put(_DONE)

    task = _start_task(pump, finished)
    loop = task.get_loop()
    try:
        while True:
            loop.call_soon_threadsafe(demand.release)
            chunk = chunks.get()
            if chunk is _DONE:
                break
            yield chunk
    finally:
        # Closing early cancels the task; this is a no-op once it is done.
        loop.call_soon_threadsafe(task.cancel)
        _wait(task, finished)
    task.result()
