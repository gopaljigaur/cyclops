"""Task execution engine for agents"""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional
from pydantic import BaseModel
import asyncio
from enum import Enum


class TaskStatus(Enum):
    """Task execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    """Task representation"""

    id: str
    name: str
    func: Callable
    args: tuple = ()
    kwargs: dict = {}
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None


class Executor(ABC):
    """Abstract task executor"""

    @abstractmethod
    async def execute(self, task: Task) -> Task:
        """Execute a single task"""
        pass

    @abstractmethod
    async def execute_batch(self, tasks: List[Task]) -> List[Task]:
        """Execute multiple tasks"""
        pass


class AsyncExecutor(Executor):
    """Asynchronous task executor"""

    async def execute(self, task: Task) -> Task:
        """Execute a single task"""
        task.status = TaskStatus.RUNNING

        try:
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                result = task.func(*task.args, **task.kwargs)

            task.result = result
            task.status = TaskStatus.COMPLETED
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED

        return task

    async def execute_batch(self, tasks: List[Task]) -> List[Task]:
        """Execute multiple tasks concurrently"""
        coroutines = [self.execute(task) for task in tasks]
        return await asyncio.gather(*coroutines)
