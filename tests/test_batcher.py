import asyncio
import random
import threading
import time
import unittest

from batchlet import (
    AsyncBatcher, BatchPolicy, BatcherClosed, BatchResultError,
    ItemTooLarge, QueueFull, threaded,
)


async def identity(items):
    return items


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


class PolicyTests(unittest.TestCase):
    def test_validation(self):
        for kwargs in [dict(max_items=0), dict(max_items=True), dict(max_tokens=-1),
                       dict(token_mode='other'), dict(pad_to_multiple=8),
                       dict(token_mode='padded', pad_to_multiple=0)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BatchPolicy(**kwargs)

    def test_rounding(self):
        p = BatchPolicy(max_tokens=32, token_mode='padded', pad_to_multiple=8)
        self.assertEqual(p.cost(2, 10, 9), 32)
        self.assertTrue(p.fits(2, 10, 9))
        self.assertFalse(p.fits(3, 11, 9))


class BatcherTests(unittest.IsolatedAsyncioTestCase):
    def batcher(self, processor=identity, **kwargs):
        batcher = AsyncBatcher(processor, **kwargs)
        self.addAsyncCleanup(batcher.aclose)
        return batcher

    async def test_count_batches_and_order(self):
        seen = []
        async def process(items):
            seen.append(items)
            return [x * 2 for x in items]
        b = self.batcher(process, policy=BatchPolicy(max_items=3), max_wait_ms=1)
        self.assertEqual(await asyncio.gather(*(b.submit(i) for i in range(8))),
                         [i * 2 for i in range(8)])
        self.assertEqual(seen, [[0, 1, 2], [3, 4, 5], [6, 7]])
        await b.aclose()
        self.assertEqual(b.stats().succeeded, 8)
        self.assertEqual(b.stats().outstanding_items, 0)

    async def test_sum_and_padded_exact_boundaries_fifo(self):
        for mode, expected in [('sum', [[0, 1, 2], [3]]),
                               ('padded', [[0, 1], [2, 3]])]:
            seen = []
            async def process(items):
                seen.append(items)
                return items
            b = self.batcher(process, policy=BatchPolicy(max_items=4, max_tokens=512,
                                                        token_mode=mode), max_wait_ms=1)
            sizes = [64, 64, 256, 256]
            self.assertEqual(await asyncio.gather(*(b.submit(i, size=n) for i, n in enumerate(sizes))),
                             list(range(4)))
            self.assertEqual(seen, expected)

    async def test_padding_rounding_enforced(self):
        seen = []
        async def process(items):
            seen.append(items)
            return items
        b = self.batcher(process, policy=BatchPolicy(max_tokens=32, token_mode='padded',
                                                    pad_to_multiple=8), max_wait_ms=1)
        await asyncio.gather(*(b.submit(i, size=9) for i in range(3)))
        self.assertEqual(seen, [[0, 1], [2]])

    async def test_count_mode_does_not_apply_token_budget(self):
        b = self.batcher(policy=BatchPolicy(max_items=2), max_wait_ms=1)
        self.assertEqual(await asyncio.gather(b.submit('a', size=100000), b.submit('b', size=1)),
                         ['a', 'b'])

    async def test_partial_batch_flushes(self):
        b = self.batcher(max_wait_ms=10)
        self.assertEqual(await asyncio.wait_for(b.submit(7), .5), 7)

    async def test_collection_wait_not_before_full_batch(self):
        b = self.batcher(policy=BatchPolicy(max_items=2), max_wait_ms=10000)
        self.assertEqual(await asyncio.wait_for(asyncio.gather(b.submit(1), b.submit(2)), .5), [1, 2])

    async def test_zero_wait(self):
        b = self.batcher(max_wait_ms=0)
        self.assertEqual(await b.submit(None), None)

    async def test_oversized_item_rejected_without_poisoning_queue(self):
        b = self.batcher(policy=BatchPolicy(max_tokens=8), max_wait_ms=0)
        with self.assertRaises(ItemTooLarge):
            await b.submit(1, size=9)
        self.assertEqual(await b.submit(2, size=8), 2)
        self.assertEqual(b.stats().rejected, 1)

    async def test_pending_single_item_limit(self):
        b = self.batcher(max_pending_tokens=5)
        with self.assertRaises(ItemTooLarge):
            await b.submit(1, size=6)

    async def test_admission_includes_running_items(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def process(items):
            started.set()
            await release.wait()
            return items
        b = self.batcher(process, max_pending_items=1, max_wait_ms=0)
        first = asyncio.create_task(b.submit('first'))
        await started.wait()
        try:
            with self.assertRaises(QueueFull):
                await b.submit('second')
        finally:
            release.set()
        self.assertEqual(await first, 'first')

    async def test_token_admission_includes_running_tokens(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def process(items):
            started.set()
            await release.wait()
            return items
        b = self.batcher(process, max_pending_tokens=10, max_wait_ms=0)
        first = asyncio.create_task(b.submit(1, size=8))
        await started.wait()
        try:
            with self.assertRaises(QueueFull):
                await b.submit(2, size=3)
        finally:
            release.set()
        await first

    async def test_cancel_pending_frees_admission_and_drops_item(self):
        b = self.batcher(max_pending_items=1, max_wait_ms=10000)
        task = asyncio.create_task(b.submit(1))
        await until(lambda: b.stats().queued_items == 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(b.stats().outstanding_items, 0)
        self.assertEqual(b.stats().queued_items, 0)
        self.assertEqual(b.stats().cancelled, 1)
        second = asyncio.create_task(b.submit(2))
        await until(lambda: b.stats().queued_items == 1)
        await b.aclose()
        self.assertEqual(await second, 2)

    async def test_timeout_pending_does_not_execute(self):
        seen = []
        async def process(items):
            seen.extend(items)
            return items
        b = self.batcher(process, max_wait_ms=10000)
        with self.assertRaises(TimeoutError):
            await b.submit(1, timeout=.01)
        await b.aclose()
        self.assertEqual(seen, [])
        self.assertEqual(b.stats().timed_out, 1)
        self.assertEqual(b.stats().cancelled, 0)

    async def test_timeout_during_execution_does_not_release_capacity_early(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def process(items):
            started.set()
            await release.wait()
            return items
        b = self.batcher(process, max_pending_items=1, max_wait_ms=0)
        task = asyncio.create_task(b.submit(1, timeout=.03))
        await started.wait()
        try:
            with self.assertRaises(TimeoutError):
                await task
            self.assertEqual(b.stats().outstanding_items, 1)
            with self.assertRaises(QueueFull):
                await b.submit(2)
        finally:
            release.set()
        await b.aclose()
        self.assertEqual(b.stats().outstanding_items, 0)

    async def test_cancelling_one_request_keeps_shared_batch_alive(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def process(items):
            started.set()
            await release.wait()
            return items
        b = self.batcher(process, policy=BatchPolicy(max_items=2))
        a, other = asyncio.create_task(b.submit(1)), asyncio.create_task(b.submit(2))
        await started.wait()
        a.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await a
        release.set()
        self.assertEqual(await other, 2)

    async def test_failure_fans_out_and_next_batch_recovers(self):
        async def process(items):
            if 0 in items:
                raise ValueError('model failed')
            return items
        b = self.batcher(process, policy=BatchPolicy(max_items=2), max_wait_ms=1)
        results = await asyncio.gather(b.submit(0), b.submit(1), return_exceptions=True)
        self.assertTrue(all(isinstance(x, ValueError) for x in results))
        self.assertEqual(await b.submit(3), 3)
        self.assertEqual(b.stats().failed, 2)

    async def test_processor_timeout_is_not_client_timeout(self):
        async def process(items):
            raise TimeoutError('backend timeout')
        b = self.batcher(process, max_wait_ms=0)
        with self.assertRaisesRegex(TimeoutError, 'backend'):
            await b.submit(1, timeout=1)
        self.assertEqual(b.stats().timed_out, 0)
        self.assertEqual(b.stats().failed, 1)

    async def test_wrong_output_count_fails_all(self):
        async def process(items):
            return [1]
        b = self.batcher(process, policy=BatchPolicy(max_items=2))
        results = await asyncio.gather(b.submit(0), b.submit(1), return_exceptions=True)
        self.assertTrue(all(isinstance(x, BatchResultError) for x in results))

    async def test_string_output_rejected(self):
        async def process(items):
            return 'x'
        b = self.batcher(process, max_wait_ms=0)
        with self.assertRaises(BatchResultError):
            await b.submit(1)

    async def test_processor_self_cancellation_resolves_waiter(self):
        async def process(items):
            raise asyncio.CancelledError
        b = self.batcher(process, max_wait_ms=0)
        with self.assertRaises(BatcherClosed):
            await asyncio.wait_for(b.submit(1), .5)

    async def test_concurrency_limit_and_out_of_order_completion(self):
        active = maximum = 0
        order = []
        async def process(items):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(.03 if items[0] == 0 else .001)
            order.extend(items)
            active -= 1
            return items
        b = self.batcher(process, policy=BatchPolicy(max_items=1), max_concurrent_batches=2)
        self.assertEqual(await asyncio.gather(*(b.submit(i) for i in range(8))), list(range(8)))
        self.assertEqual(maximum, 2)
        self.assertNotEqual(order, list(range(8)))

    async def test_close_flushes_and_cannot_restart(self):
        b = self.batcher(max_wait_ms=10000)
        task = asyncio.create_task(b.submit(1))
        await until(lambda: b.stats().queued_items == 1)
        await asyncio.wait_for(b.aclose(), .5)
        self.assertEqual(await task, 1)
        with self.assertRaises(BatcherClosed):
            await b.submit(2)
        await b.aclose()

    async def test_close_without_drain_discards_queue_not_running_work(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def process(items):
            started.set()
            await release.wait()
            return items
        b = self.batcher(process, policy=BatchPolicy(max_items=1))
        first = asyncio.create_task(b.submit(1))
        await started.wait()
        second = asyncio.create_task(b.submit(2))
        await until(lambda: b.stats().queued_items == 1)
        closing = asyncio.create_task(b.aclose(drain=False))
        with self.assertRaises(BatcherClosed):
            await second
        self.assertFalse(first.done())
        release.set()
        await closing
        self.assertEqual(await first, 1)

    async def test_close_timeout_does_not_cancel_processor(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def process(items):
            started.set()
            await release.wait()
            return items
        b = self.batcher(process, max_wait_ms=0)
        task = asyncio.create_task(b.submit(1))
        await started.wait()
        try:
            with self.assertRaises(TimeoutError):
                await b.aclose(timeout=.01)
            self.assertFalse(task.done())
        finally:
            release.set()
        await b.aclose()
        self.assertEqual(await task, 1)

    async def test_close_before_start(self):
        b = self.batcher()
        await b.aclose()
        with self.assertRaises(BatcherClosed):
            await b.submit(1)

    async def test_context_manager(self):
        b = AsyncBatcher(identity, max_wait_ms=0)
        async with b:
            self.assertEqual(await b.submit(5), 5)
        with self.assertRaises(BatcherClosed):
            await b.submit(6)

    async def test_observer_and_observer_failure_isolation(self):
        events = []
        def observer(event):
            events.append(event)
            raise RuntimeError('metrics backend unavailable')
        b = self.batcher(policy=BatchPolicy(max_tokens=8, token_mode='padded'),
                         on_batch=observer, max_wait_ms=0)
        self.assertEqual(await b.submit('ok', size=3), 'ok')
        await b.aclose()
        self.assertEqual((events[0].tokens, events[0].cost, events[0].failed), (3, 3, False))
        self.assertGreaterEqual(events[0].execution_s, 0)
        self.assertEqual(b.stats().observer_errors, 1)

    async def test_thread_adapter_does_not_block_loop(self):
        thread_ids = []
        def process(items):
            thread_ids.append(threading.get_ident())
            time.sleep(.01)
            return items
        b = self.batcher(threaded(process), max_wait_ms=0)
        self.assertEqual(await b.submit(2), 2)
        self.assertNotEqual(thread_ids[0], threading.get_ident())

    async def test_invalid_inputs_and_settings(self):
        for kwargs in [dict(max_wait_ms=-1), dict(max_wait_ms=float('nan')),
                       dict(max_wait_ms=float('inf')), dict(max_concurrent_batches=0),
                       dict(max_pending_items=0), dict(max_pending_tokens=0)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AsyncBatcher(identity, **kwargs)
        b = self.batcher()
        for size in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                await b.submit(1, size=size)
        with self.assertRaises(ValueError):
            await b.submit(1, timeout=0)

    async def test_randomized_mixed_lengths_and_result_mapping(self):
        for mode in ['sum', 'padded']:
            rng = random.Random(47)
            sizes = [rng.randint(1, 32) for _ in range(400)]
            seen = []
            async def process(items):
                lengths = [sizes[i] for i in items]
                cost = sum(lengths) if mode == 'sum' else len(lengths) * max(lengths)
                self.assertLessEqual(cost, 64)
                self.assertLessEqual(len(items), 7)
                seen.extend(items)
                await asyncio.sleep(0)
                return [i * 3 for i in items]
            b = self.batcher(process, policy=BatchPolicy(max_items=7, max_tokens=64,
                                                        token_mode=mode), max_wait_ms=0,
                             max_concurrent_batches=3)
            result = await asyncio.gather(*(b.submit(i, size=n) for i, n in enumerate(sizes)))
            self.assertEqual(result, [i * 3 for i in range(400)])
            self.assertEqual(seen, list(range(400)))
            await b.aclose()
            self.assertEqual(b.stats().outstanding_tokens, 0)


class LoopOwnershipTests(unittest.TestCase):
    def test_cross_loop_rejected(self):
        b = AsyncBatcher(identity, max_wait_ms=0)
        async def first():
            await b.submit(1)
            await b.aclose()
        asyncio.run(first())
        with self.assertRaisesRegex(RuntimeError, 'event loops'):
            asyncio.run(b.submit(2))
