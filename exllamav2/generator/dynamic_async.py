from __future__ import annotations
from exllamav2.generator.dynamic import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob
import asyncio

class ExLlamaV2DynamicGeneratorAsync:
    """
    Async wrapper for dynamic generator. See definition of ExLlamaV2DynamicGenerator.
    """

    generator: ExLlamaV2DynamicGenerator
    jobs: dict[ExLlamaV2DynamicJob: ExLlamaV2DynamicJobAsync]
    condition: asyncio.Condition
    iteration_task: asyncio.Task

    def __init__(self, *args, **kwargs):
        self.generator = ExLlamaV2DynamicGenerator(*args, **kwargs)
        self.jobs = {}
        self.condition = asyncio.Condition()
        self.iteration_task = asyncio.create_task(self._run_iteration())

    async def _run_iteration(self):
        while True:
            async with self.condition:
                await self.condition.wait_for(lambda: len(self.jobs) > 0)
            results = self.generator.iterate()
            for result in results:
                job = result["job"]
                async_job = self.jobs[job]
                await async_job.put_result(result)
                if result["eos"]:
                    del self.jobs[job]
            await asyncio.sleep(0)

    def enqueue(self, job: ExLlamaV2DynamicJobAsync):
        assert job.job not in self.jobs
        self.jobs[job.job] = job
        self.generator.enqueue(job.job)
        asyncio.create_task(self._notify_condition())

    async def _notify_condition(self):
        async with self.condition:
            self.condition.notify_all()

    async def close(self):
        self.iteration_task.cancel()
        try:
            await self.iteration_task
        except asyncio.CancelledError:
            pass

    async def cancel(self, job: ExLlamaV2DynamicJobAsync):
        assert job.job in self.jobs
        self.generator.cancel(job.job)
        del self.jobs[job.job]


class ExLlamaV2DynamicJobAsync:
    """
    Async wrapper for dynamic generator job. See definition of ExLlamaV2DynamicJob.
    """

    job: ExLlamaV2DynamicJob
    queue: asyncio.Queue
    generator: ExLlamaV2DynamicGeneratorAsync

    def __init__(self, generator: ExLlamaV2DynamicGeneratorAsync, *args: object, **kwargs: object):
        self.generator = generator
        self.job = ExLlamaV2DynamicJob(*args, **kwargs)
        self.queue = asyncio.Queue()
        self.generator.enqueue(self)

    async def put_result(self, result):
        await self.queue.put(result)

    async def __aiter__(self):
        while True:
            result = await self.queue.get()
            yield result
            if result["eos"]:
                break

    async def cancel(self):
        await self.generator.cancel(self)
