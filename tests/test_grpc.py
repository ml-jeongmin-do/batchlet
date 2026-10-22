import asyncio
import json
import unittest

from batchlet import AsyncBatcher, BatchPolicy, BatchResultError
from batchlet.grpc import grpc_processor

try:
    import grpc
except ImportError:
    grpc = None


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_encode_decode_and_options_without_grpc_dependency(self):
        seen = []
        async def rpc(request, **kwargs):
            seen.append((request, kwargs))
            return {'results': [x * 2 for x in request['items']]}
        process = grpc_processor(rpc, encode=lambda xs: {'items': xs},
                                 decode=lambda reply: reply['results'], timeout=3,
                                 metadata=[('x-model', 'qwen')], wait_for_ready=True)
        async with AsyncBatcher(process, policy=BatchPolicy(max_items=2)) as b:
            self.assertEqual(await asyncio.gather(b.submit(2), b.submit(3)), [4, 6])
        self.assertEqual(seen, [({'items': [2, 3]},
                                {'timeout': 3, 'metadata': (('x-model', 'qwen'),),
                                 'wait_for_ready': True})])

    async def test_decode_error_propagates(self):
        async def rpc(request, **kwargs):
            return {}
        process = grpc_processor(rpc, encode=list, decode=lambda reply: reply['missing'])
        async with AsyncBatcher(process, max_wait_ms=0) as b:
            with self.assertRaises(KeyError):
                await b.submit(1)

    async def test_invalid_rpc_deadline(self):
        with self.assertRaises(ValueError):
            grpc_processor(None, encode=list, decode=list, timeout=0)


@unittest.skipIf(grpc is None, 'Install batchlet[grpc] for actual gRPC integration tests')
class GrpcIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.channels = []
        self.servers = []
        self.calls = []
        for node in range(2):
            async def infer(request, context, node=node):
                self.calls.append((node, request, dict(context.invocation_metadata())))
                if 'fail' in request:
                    await context.abort(grpc.StatusCode.UNAVAILABLE, 'temporary backend error')
                if 'slow' in request:
                    await asyncio.sleep(.2)
                if 'short' in request:
                    return []
                return [{'value': value, 'node': node} for value in request]
            server = grpc.aio.server()
            handler = grpc.unary_unary_rpc_method_handler(
                infer, request_deserializer=json.loads,
                response_serializer=lambda response: json.dumps(response).encode())
            server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
                'tests.Embedding', {'Infer': handler}),))
            port = server.add_insecure_port('127.0.0.1:0')
            self.assertNotEqual(port, 0)
            await server.start()
            self.servers.append(server)
            channel = grpc.aio.insecure_channel(f'127.0.0.1:{port}')
            self.channels.append(channel)
        self.rpcs = [channel.unary_unary(
            '/tests.Embedding/Infer', request_serializer=lambda items: json.dumps(items).encode(),
            response_deserializer=json.loads) for channel in self.channels]

    async def asyncTearDown(self):
        await asyncio.gather(*(c.close() for c in self.channels))
        await asyncio.gather(*(s.stop(0) for s in self.servers))

    def processor(self, rpc=None, **kwargs):
        return grpc_processor(rpc or self.rpcs[0], encode=list, decode=list, **kwargs)

    async def test_real_rpc_batches_metadata_and_output_order(self):
        process = self.processor(metadata=[('x-model', 'demo')])
        async with AsyncBatcher(process, policy=BatchPolicy(max_items=3)) as b:
            result = await asyncio.gather(*(b.submit(i) for i in range(3)))
        self.assertEqual([x['value'] for x in result], [0, 1, 2])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][2]['x-model'], 'demo')

    async def test_two_nodes_selected_in_caller_callback(self):
        index = 0
        def choose(request, **kwargs):
            nonlocal index
            rpc = self.rpcs[index % 2]
            index += 1
            return rpc(request, **kwargs)
        async with AsyncBatcher(self.processor(choose), policy=BatchPolicy(max_items=2),
                                max_concurrent_batches=2) as b:
            result = await asyncio.gather(*(b.submit(i) for i in range(8)))
        self.assertEqual([x['value'] for x in result], list(range(8)))
        self.assertEqual({x['node'] for x in result}, {0, 1})
        self.assertEqual(len(self.calls), 4)

    async def test_rpc_failure_fans_out_without_retry_and_recovers(self):
        async with AsyncBatcher(self.processor(), policy=BatchPolicy(max_items=2),
                                max_wait_ms=1) as b:
            errors = await asyncio.gather(b.submit('fail'), b.submit('other'),
                                          return_exceptions=True)
            self.assertTrue(all(isinstance(e, grpc.aio.AioRpcError) for e in errors))
            self.assertTrue(all(e.code() == grpc.StatusCode.UNAVAILABLE for e in errors))
            self.assertEqual((await b.submit('ok'))['value'], 'ok')
        self.assertEqual(len(self.calls), 2)

    async def test_rpc_deadline_and_client_deadline_are_separate(self):
        async with AsyncBatcher(self.processor(timeout=.02), max_wait_ms=0) as b:
            with self.assertRaises(grpc.aio.AioRpcError) as caught:
                await b.submit('slow', timeout=1)
            self.assertEqual(caught.exception.code(), grpc.StatusCode.DEADLINE_EXCEEDED)
            self.assertEqual(b.stats().timed_out, 0)
            self.assertEqual(b.stats().failed, 1)

    async def test_rpc_wrong_output_count(self):
        async with AsyncBatcher(self.processor(), max_wait_ms=0) as b:
            with self.assertRaises(BatchResultError):
                await b.submit('short')

    async def test_one_client_timeout_keeps_shared_rpc_alive(self):
        async with AsyncBatcher(self.processor(timeout=2), policy=BatchPolicy(max_items=2)) as b:
            results = await asyncio.gather(b.submit('slow', timeout=.03),
                                           b.submit('ok', timeout=2), return_exceptions=True)
            self.assertIsInstance(results[0], TimeoutError)
            self.assertEqual(results[1]['value'], 'ok')
        self.assertEqual(len(self.calls), 1)
