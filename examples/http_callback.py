"""Send batches to an HTTP endpoint.

POST {"inputs": [...]} -> {"outputs": [...]} in input order.
"""
import asyncio
import json
import os
from urllib.request import Request, urlopen

from batchlet import AsyncBatcher, BatchPolicy, threaded


def call_remote(items):
    request = Request(os.environ['BATCH_ENDPOINT'],
                      data=json.dumps({'inputs': items}).encode(),
                      headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=10) as response:
        return json.load(response)['outputs']


async def main():
    async with AsyncBatcher(threaded(call_remote), policy=BatchPolicy(max_items=8),
                            max_concurrent_batches=2) as batcher:
        print(await asyncio.gather(*(batcher.submit(text) for text in ['hello', 'world'])))


if __name__ == '__main__':
    asyncio.run(main())
