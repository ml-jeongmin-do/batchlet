"""Runnable gRPC demo with two localhost mock nodes; no GPU or protoc required.

Install: pip install -e '.[grpc]'
Run: python examples/grpc_demo.py
JSON is the demo wire serializer. Production generated Protobuf stubs work
through the same grpc_processor adapter; see README.md for that example.
"""
import asyncio
import itertools
import json

import grpc

from batchlet import AsyncBatcher, BatchPolicy
from batchlet.grpc import grpc_processor


async def main():
    servers, channels, rpcs = [], [], []
    try:
        for node in ['gpu-a-mock', 'gpu-b-mock']:
            async def infer(inputs, context, node=node):
                print(f'{node}: batch size={len(inputs)}')
                return [{'sum': sum(ids), 'node': node} for ids in inputs]
            server = grpc.aio.server()
            servers.append(server)
            server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
                'demo.Embedding', {'Infer': grpc.unary_unary_rpc_method_handler(
                    infer, request_deserializer=json.loads,
                    response_serializer=lambda x: json.dumps(x).encode())}),))
            port = server.add_insecure_port('127.0.0.1:0')
            await server.start()
            channel = grpc.aio.insecure_channel(f'127.0.0.1:{port}')
            channels.append(channel)
            rpcs.append(channel.unary_unary('/demo.Embedding/Infer',
                        request_serializer=lambda x: json.dumps(x).encode(),
                        response_deserializer=json.loads))

        # Only an example of caller-owned routing. No node management in the library.
        cycle = itertools.cycle(rpcs)
        def choose_rpc(request, **kwargs):
            return next(cycle)(request, **kwargs)

        processor = grpc_processor(choose_rpc, encode=list, decode=list, timeout=5)
        async with AsyncBatcher(processor,
                                policy=BatchPolicy(max_items=2, max_tokens=16),
                                max_concurrent_batches=2) as batcher:
            inputs = [[i] * 4 for i in range(8)]
            results = await asyncio.gather(*(batcher.submit(ids, size=len(ids)) for ids in inputs))
            print('ordered results:', results)
        print('stats:', batcher.stats())
    finally:
        await asyncio.gather(*(channel.close() for channel in channels))
        await asyncio.gather(*(server.stop(0) for server in servers))


if __name__ == '__main__':
    asyncio.run(main())
