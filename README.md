# batchlet

Async batching with item and token budgets.

Combine concurrent inputs into batches and return each result to its caller.
Connect a local function, HTTP endpoint, or gRPC service.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Install from source until the package is published to PyPI:

```bash
git clone git@github.com:ml-jeongmin-do/batchlet.git
cd batchlet
uv sync --locked --no-dev
```

From another uv project:

```bash
uv add --editable /path/to/batchlet
```

## Usage

```python
import asyncio

from batchlet import AsyncBatcher, BatchPolicy


async def process(items: list[int]) -> list[int]:
    return [item * 2 for item in items]


async def main() -> None:
    async with AsyncBatcher(
        process,
        policy=BatchPolicy(max_items=4),
        max_wait_ms=2,
    ) as batcher:
        results = await asyncio.gather(
            *(batcher.submit(i) for i in range(10))
        )
        print(results)


asyncio.run(main())
```

Submit concurrently to a shared batcher. The processor must return one result
per input, in input order.

## Policies

```python
BatchPolicy(max_items=64)
BatchPolicy(max_items=64, max_tokens=8192, token_mode="sum")
BatchPolicy(max_items=64, max_tokens=8192, token_mode="padded")
```

Pass input length through `submit(item, size=token_count)`.
`sum` uses total input length; `padded` uses input count × longest input length.
All policies enforce `max_items` and select inputs in FIFO order.

## Examples

Run from the repository root:

```bash
uv run --locked --no-dev python examples/local.py
uv run --locked --no-dev --extra grpc python examples/grpc_demo.py
BATCH_ENDPOINT=http://localhost:8000/batch uv run --locked --no-dev python examples/http_callback.py
```

- [Local](examples/local.py): a blocking callback with `threaded()`.
- [gRPC](examples/grpc_demo.py): two local mock nodes with round-robin routing.
- [HTTP](examples/http_callback.py): an existing endpoint accepting
  `{"inputs": [...]}` and returning `{"outputs": [...]}` in order.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development.

## License

Copyright 2026 batchlet contributors. Licensed under [Apache 2.0](LICENSE).
