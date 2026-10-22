"""One event-loop-owned FIFO scheduler and bounded concurrent batch execution."""
from __future__ import annotations

import asyncio
import math
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Generic, TypeVar

from .errors import BatcherClosed, BatchResultError, ItemTooLarge, QueueFull
from .policy import BatchPolicy, positive_int

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class BatchEvent:
    """One completed batch; no payloads or user identifiers are included."""

    items: int
    tokens: int
    cost: int
    oldest_wait_s: float
    execution_s: float
    failed: bool


@dataclass(frozen=True)
class Stats:
    submitted: int
    succeeded: int
    failed: int
    cancelled: int
    timed_out: int
    rejected: int
    batches: int
    queued_items: int
    outstanding_items: int
    outstanding_tokens: int
    running_batches: int
    observer_errors: int


@dataclass(eq=False)
class _Item(Generic[T, R]):
    value: T
    size: int
    future: asyncio.Future[R]
    enqueued: float
    dispatched: bool = False
    released: bool = False


def _seconds(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} and finite")
    return float(value)


def threaded(
    processor: Callable[[list[T]], Sequence[R]], *, executor: Executor | None = None,
) -> Callable[[list[T]], Awaitable[Sequence[R]]]:
    """Adapt blocking work without blocking the event loop.

    With no executor, asyncio.to_thread propagates contextvars. A supplied
    executor is owned/shut down by the caller. Threads cannot be force-stopped;
    request cancellation deliberately does not cancel an executing batch.
    """
    async def run(items: list[T]) -> Sequence[R]:
        if executor is None:
            return await asyncio.to_thread(processor, items)
        return await asyncio.get_running_loop().run_in_executor(executor, processor, items)
    return run


class AsyncBatcher(Generic[T, R]):
    """Batch compatible inputs on one asyncio event loop.

    ``processor`` receives a list and must return a finite, ordered sequence of
    exactly the same length. It must be async; wrap blocking functions with
    ``threaded``. Use one batcher per model/configuration. There is no tokenizer,
    HTTP framework, node discovery, routing, retry, or cross-process queue here.
    """

    def __init__(
        self,
        processor: Callable[[list[T]], Awaitable[Sequence[R]]],
        *,
        policy: BatchPolicy | None = None,
        max_wait_ms: float = 2,
        max_pending_items: int = 1024,
        max_pending_tokens: int | None = None,
        max_concurrent_batches: int = 1,
        on_batch: Callable[[BatchEvent], None] | None = None,
    ) -> None:
        if not callable(processor):
            raise TypeError("processor must be callable")
        if on_batch is not None and not callable(on_batch):
            raise TypeError("on_batch must be callable")
        positive_int(max_pending_items, "max_pending_items")
        positive_int(max_concurrent_batches, "max_concurrent_batches")
        if max_pending_tokens is not None:
            positive_int(max_pending_tokens, "max_pending_tokens")
        self._policy = policy or BatchPolicy()
        self._processor = processor
        self._wait = _seconds(max_wait_ms, "max_wait_ms", allow_zero=True) / 1000
        self._max_items = max_pending_items
        self._max_tokens = max_pending_tokens
        self._concurrency = max_concurrent_batches
        self._observer = on_batch
        self._queue: deque[_Item[T, R]] = deque()
        self._running: set[asyncio.Task[None]] = set()
        self._wake = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dispatcher: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = False
        self._outstanding_items = self._outstanding_tokens = 0
        self._counts = dict(submitted=0, succeeded=0, failed=0, cancelled=0,
                            timed_out=0, rejected=0, batches=0, observer_errors=0)

    @property
    def policy(self) -> BatchPolicy:
        """Immutable policy for this batcher; create another instance to change it."""
        return self._policy

    async def __aenter__(self) -> AsyncBatcher[T, R]:
        self._start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _bind_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("A batcher cannot be shared across event loops")
        self._loop = loop
        return loop

    def _start(self) -> None:
        self._bind_loop()
        if self._closing or self._closed:
            raise BatcherClosed("Batcher no longer accepts submissions")
        if self._dispatcher is None:
            self._dispatcher = asyncio.create_task(self._dispatch(), name="batchlet-dispatch")

    async def submit(self, item: T, *, size: int = 1, timeout: float | None = None) -> R:
        """Submit one indivisible item; timeout covers admission to returned result.

        Validation/admission never waits. Count and raw-size limits include running
        batches. Timing out stops waiting for the result, not shared GPU work.
        Preprocessing performed by the caller is outside this timeout.
        """
        positive_int(size, "size")
        if timeout is not None:
            timeout = _seconds(timeout, "timeout")
        self._start()
        if not self.policy.fits(1, size, size):
            self._counts["rejected"] += 1
            raise ItemTooLarge("Item exceeds the batch token budget")
        if self._max_tokens is not None and size > self._max_tokens:
            self._counts["rejected"] += 1
            raise ItemTooLarge("Item exceeds the outstanding token budget")
        if (self._outstanding_items >= self._max_items or
            (self._max_tokens is not None and
             self._outstanding_tokens + size > self._max_tokens)):
            self._counts["rejected"] += 1
            raise QueueFull("Outstanding item/token limit reached")
        loop = asyncio.get_running_loop()
        entry: _Item[T, R] = _Item(item, size, loop.create_future(), loop.time())
        self._queue.append(entry)
        self._outstanding_items += 1
        self._outstanding_tokens += size
        self._counts["submitted"] += 1
        self._wake.set()
        try:
            # TimeoutError raised by the processor is a processor failure, not a
            # client deadline. timeout.expired() lets us distinguish the two.
            deadline = asyncio.timeout(timeout)
            try:
                async with deadline:
                    return await entry.future
            except TimeoutError:
                if deadline.expired():
                    self._counts["timed_out"] += 1
                raise
        except asyncio.CancelledError:
            self._counts["cancelled"] += 1
            raise
        finally:
            if not entry.future.done():
                entry.future.cancel()
            if not entry.dispatched and not entry.released:
                # Remove immediately: repeated cancellations must not leave
                # unbounded tombstones behind a slow executing batch.
                self._queue.remove(entry)
                self._release(entry)
            self._wake.set()

    def _release(self, item: _Item[T, R]) -> None:
        if not item.released:
            item.released = True
            self._outstanding_items -= 1
            self._outstanding_tokens -= item.size

    def _select(self) -> tuple[list[_Item[T, R]], bool]:
        # Cancelled futures can be visible before submit's finally gets a turn.
        live: deque[_Item[T, R]] = deque()
        for item in self._queue:
            if item.future.done():
                self._release(item)
            else:
                live.append(item)
        self._queue = live
        selected: list[_Item[T, R]] = []
        total = longest = 0
        full = False
        for item in self._queue:
            if not self.policy.fits(len(selected) + 1, total + item.size, max(longest, item.size)):
                full = True
                break
            selected.append(item)
            total += item.size
            longest = max(longest, item.size)
            if (len(selected) == self.policy.max_items or
                (self.policy.max_tokens is not None and
                 self.policy.cost(len(selected), total, longest) == self.policy.max_tokens)):
                full = True
                break
        return selected, full

    async def _dispatch(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            self._wake.clear()
            if self._closing and not self._queue and not self._running:
                self._closed = True
                return
            if len(self._running) >= self._concurrency:
                await self._wake.wait()
                continue
            selected, full = self._select()
            if not selected:
                # _select may just have purged the last cancelled item.
                if self._closing and not self._running:
                    self._closed = True
                    return
                await self._wake.wait()
                continue
            remaining = selected[0].enqueued + self._wait - loop.time()
            if not full and remaining > 0 and not self._closing:
                try:
                    await asyncio.wait_for(self._wake.wait(), remaining)
                except TimeoutError:
                    pass
                continue
            for item in selected:
                assert self._queue.popleft() is item
                item.dispatched = True
            task = asyncio.create_task(self._execute(selected), name="batchlet-execute")
            self._running.add(task)
            task.add_done_callback(self._batch_done)

    def _batch_done(self, task: asyncio.Task[None]) -> None:
        self._running.discard(task)
        self._wake.set()

    async def _execute(self, entries: list[_Item[T, R]]) -> None:
        loop = asyncio.get_running_loop()
        active = []
        for item in entries:
            if item.future.done():
                self._release(item)
            else:
                active.append(item)
        if not active:
            return
        started = loop.time()
        failed = False
        self._counts["batches"] += 1
        try:
            results = await self._processor([item.value for item in active])
            if isinstance(results, (str, bytes, bytearray)) or len(results) != len(active):
                raise BatchResultError("processor must return exactly one output per input")
            # Materialize before resolving futures: malformed output must not
            # partially succeed and then fail the remainder of a batch.
            values = list(results)
            if len(values) != len(active):
                raise BatchResultError("processor output length changed during iteration")
            for item, result in zip(active, values, strict=True):
                if not item.future.done():
                    item.future.set_result(result)
                    self._counts["succeeded"] += 1
        except asyncio.CancelledError:
            # A processor may itself raise CancelledError. Resolve every waiter
            # rather than letting the shared task silently strand futures.
            failed = True
            self._fail(active, BatcherClosed("Batch processor was cancelled"))
        except Exception as exc:
            failed = True
            self._fail(active, exc)
        finally:
            finished = loop.time()
            for item in active:
                self._release(item)
            if self._observer is not None:
                total = sum(item.size for item in active)
                longest = max(item.size for item in active)
                event = BatchEvent(len(active), total,
                                   self.policy.cost(len(active), total, longest),
                                   started - active[0].enqueued, finished - started, failed)
                try:
                    self._observer(event)
                except Exception:
                    self._counts["observer_errors"] += 1

    def _fail(self, entries: Sequence[_Item[T, R]], error: Exception) -> None:
        for item in entries:
            if not item.future.done():
                item.future.set_exception(error)
                self._counts["failed"] += 1

    async def aclose(self, *, drain: bool = True, timeout: float | None = None) -> None:
        """Stop admission and wait for running work; flush queued work by default.

        drain=False fails queued requests immediately. Executing work is never
        force-cancelled. A close timeout/cancellation leaves shutdown running;
        call aclose again to await it. Processor/network deadlines belong in the
        callback. The instance cannot be restarted after closing.
        """
        self._bind_loop()
        if timeout is not None:
            timeout = _seconds(timeout, "timeout")
        self._closing = True
        if not drain:
            while self._queue:
                item = self._queue.popleft()
                self._fail([item], BatcherClosed("Queued request discarded during shutdown"))
                self._release(item)
        self._wake.set()
        if self._dispatcher is None:
            self._closed = True
            return
        await asyncio.wait_for(asyncio.shield(self._dispatcher), timeout)

    def stats(self) -> Stats:
        """A point-in-time snapshot; call from the batcher's event loop/thread."""
        return Stats(**self._counts, queued_items=len(self._queue),
                     outstanding_items=self._outstanding_items,
                     outstanding_tokens=self._outstanding_tokens,
                     running_batches=len(self._running))
