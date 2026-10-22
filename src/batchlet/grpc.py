"""Adapter for grpc.aio unary-unary calls; no grpc import or channel ownership."""
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, TypeVar

from .core import _seconds

T = TypeVar("T")
R = TypeVar("R")
Q = TypeVar("Q")
S = TypeVar("S")
Q_contra = TypeVar("Q_contra", contravariant=True)
S_co = TypeVar("S_co", covariant=True)
Metadata = Sequence[tuple[str, str | bytes]]


class UnaryRPC(Protocol[Q_contra, S_co]):
    def __call__(
        self, request: Q_contra, *, timeout: float | None = None,
        metadata: Metadata | None = None, wait_for_ready: bool | None = None,
    ) -> Awaitable[S_co]: ...


def grpc_processor(
    rpc: UnaryRPC[Q, S],
    *,
    encode: Callable[[list[T]], Q],
    decode: Callable[[S], Sequence[R]],
    timeout: float = 30,
    metadata: Metadata | None = None,
    wait_for_ready: bool = False,
) -> Callable[[list[T]], Awaitable[Sequence[R]]]:
    """Make a batch processor from an existing async unary RPC/stub method.

    The caller owns channel creation, TLS/auth, node selection and closure.
    encode/decode are fast synchronous converters (e.g. Protobuf messages).
    decode must return outputs in input order. Errors such as AioRpcError pass
    through untouched; no retry is performed. Streaming RPCs are not supported.
    The RPC timeout begins at dispatch, independently of submit's end-to-end
    timeout. Shared work is not cancelled when just one caller stops waiting.
    """
    timeout = _seconds(timeout, "timeout")
    headers = tuple(metadata) if metadata is not None else None

    async def process(items: list[T]) -> Sequence[R]:
        response = await rpc(encode(items), timeout=timeout, metadata=headers,
                             wait_for_ready=wait_for_ready)
        return decode(response)

    return process
