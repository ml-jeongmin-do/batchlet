"""Batch a blocking function with a padded token budget."""
import asyncio

from batchlet import AsyncBatcher, BatchPolicy, threaded


def encode_batch(token_sequences: list[list[int]]) -> list[int]:
    print('batch lengths:', [len(ids) for ids in token_sequences])
    return [sum(ids) for ids in token_sequences]


async def main():
    policy = BatchPolicy(max_items=4, max_tokens=16, token_mode='padded')
    async with AsyncBatcher(threaded(encode_batch), policy=policy) as batcher:
        inputs = [[1] * n for n in [1, 2, 4, 8, 2, 8]]
        results = await asyncio.gather(*(batcher.submit(ids, size=len(ids)) for ids in inputs))
        print('results:', results)
    print('stats:', batcher.stats())


if __name__ == '__main__':
    asyncio.run(main())
