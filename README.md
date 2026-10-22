# batchlet

요청 수 또는 토큰 비용에 맞춰 입력을 모으고, 사용자가 제공한 배치 함수를 호출하는
작은 Python 라이브러리입니다. Python 3.11+, **코어 런타임 의존성 0개**.
FastAPI·PyTorch·특정 모델·HTTP 서버에 의존하지 않습니다.

```text
submit(item, size=token_count)
    → bounded FIFO queue
    → count / token-sum / padded-token budget
    → async process_batch(items)
    → 각 submit 호출에 해당 결과 반환
```

## 설치와 실행

```bash
git clone git@github.com:ml-jeongmin-do/batchlet.git
cd batchlet
python -m venv .venv
source .venv/bin/activate
pip install -e .                 # 로컬 editable 설치, 코어 의존성 없음
python examples/local.py

pip install -e '.[grpc,dev]'      # 선택: 실제 gRPC 통신 + 개발 도구
python examples/grpc_demo.py     # localhost 모의 노드 2개 자동 시작/종료
python -m unittest discover -s tests -v
python -m mypy
python -m build
```

아직 PyPI에는 발행하지 않았습니다. 다른 프로젝트에서는 로컬 경로 또는 빌드한
wheel을 설치하세요. gRPC 미설치 환경에서도 코어와 어댑터 단위 테스트는 실행되며,
실제 gRPC 통합 테스트만 skip됩니다.

## 가장 작은 사용 예

```python
import asyncio
from batchlet import AsyncBatcher, BatchPolicy

async def process_batch(items: list[int]) -> list[int]:
    return [x * 2 for x in items]

async def main():
    async with AsyncBatcher(
        process_batch,
        policy=BatchPolicy(max_items=4),
        max_wait_ms=2,
    ) as batcher:
        results = await asyncio.gather(*(batcher.submit(i) for i in range(10)))
        print(results)  # [0, 2, 4, ..., 18], 호출별 결과 매핑 유지

asyncio.run(main())
```

`for ...: await submit(...)`처럼 하나씩 기다리면 동시에 들어오는 입력이 없어
배치가 잘 만들어지지 않습니다. 실제 서버의 동시 요청이나 `asyncio.gather`를 사용하세요.
processor는 **입력과 동일한 순서·개수의 출력 sequence**를 반환해야 합니다.
출력 수가 다르면 해당 배치 전체가 실패합니다. 라이브러리가 출력 의미나 순서를
추측하지 않으므로 원격 서버가 순서를 바꾸면 `decode`에서 원래 순서로 복원하세요.

## 배치 정책

```python
# 1. 요청 개수만 제한
BatchPolicy(max_items=64)

# 2. 실제 토큰 수의 합 ≤ 8192, 요청 수 ≤ 64
BatchPolicy(max_items=64, max_tokens=8192, token_mode="sum")

# 3. 요청 수 × 최대 길이 ≤ 8192, 요청 수 ≤ 64
BatchPolicy(max_items=64, max_tokens=8192, token_mode="padded")

# 4. 엔진이 길이를 8의 배수로 padding한다면 비용에도 반영
BatchPolicy(max_items=64, max_tokens=8192,
            token_mode="padded", pad_to_multiple=8)
```

토큰화를 한 번 한 뒤 ID와 길이를 재사용합니다. `size`는 양의 정수이며 기본값은 1입니다.
실제 토큰 예산을 쓰려면 **모델에 실제로 넣을 입력 길이**를 전달해야 합니다.
프롬프트·특수 토큰·truncation 결과도 반영하세요. 라이브러리에는 tokenizer가 없습니다.

```python
features = await tokenize(text)   # 애플리케이션이 제공하는 함수
vector = await batcher.submit(features, size=len(features["input_ids"]), timeout=1.5)
```

요청 하나를 여러 조각으로 나누지 않습니다. FIFO 순서로 다음 입력이 예산을 넘으면
현재 배치를 먼저 실행하고 다음 배치에서 그 입력부터 처리합니다. 단일 입력부터
예산을 넘으면 `ItemTooLarge`를 발생시킵니다. 길이 정렬·버킷팅·요청 간 재정렬은 없습니다.

`size`는 이미지 패치 수 등 다른 단위여도 됩니다. 다만 의미와 비용 계산이 같은
입력끼리 사용하세요. 토큰 예산은 계산량·VRAM의 근사치이며 OOM 방지나 처리량 향상을
보장하지 않습니다. `padded` 정책은 padding을 수행하거나 제거하지 않습니다.

## 동기 로컬 모델 연결

블로킹 추론을 async 함수 안에서 그대로 호출하면 이벤트 루프가 멈춥니다.
`threaded`로 분리하거나 자신의 실행기를 연결하세요.

```python
from batchlet import AsyncBatcher, threaded

batcher = AsyncBatcher(threaded(model.encode_batch), max_concurrent_batches=1)
```

`threaded`는 기본적으로 `asyncio.to_thread`를 사용합니다. 동일 스레드에서만 모델을
호출해야 한다면 `ThreadPoolExecutor(max_workers=1)`을 직접 소유하고
`threaded(model.encode_batch, executor=pool)`로 전달하세요. 필요하면 모델 생성도
그 executor에서 실행하세요. 종료 순서는 **batcher drain → executor shutdown**입니다.
`max_concurrent_batches > 1`이면 processor와 모델의 동시 호출 안전성을 호출자가 보장해야 합니다.

## gRPC 연결

선택 설치: `pip install -e '.[grpc]'`. `batchlet.grpc` 어댑터 자체는 grpc를 import하지
않으며, 이미 만들어 둔 **`grpc.aio` unary-unary stub 메서드**를 받습니다.
채널·TLS·인증·Protobuf 스키마는 애플리케이션이 소유합니다.

기존 generated stub이 있는 경우의 예시입니다. 아래 `service_pb2`의 필드 이름은
자신의 서비스 스키마로 바꿔야 합니다.

```python
import asyncio
import grpc
import service_pb2 as pb
import service_pb2_grpc as rpc
from batchlet import AsyncBatcher, BatchPolicy
from batchlet.grpc import grpc_processor

async def main():
    # 로컬 개발 예시. 배포 환경의 채널/TLS 설정은 애플리케이션에서 결정.
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = rpc.EmbeddingStub(channel)
        processor = grpc_processor(
            stub.EmbedBatch,
            encode=lambda inputs: pb.EmbedBatchRequest(
                inputs=[pb.TokenInput(ids=ids) for ids in inputs]),
            decode=lambda reply: [list(row.values) for row in reply.embeddings],
            timeout=10,                 # 배치 RPC 기한, 초
            metadata=(("x-model", "qwen3"),),
        )
        async with AsyncBatcher(
            processor,
            policy=BatchPolicy(max_items=64, max_tokens=8192),
            max_concurrent_batches=2,
        ) as batcher:
            inputs = [[11, 22], [33, 44, 55]]
            vectors = await asyncio.gather(*(
                batcher.submit(ids, size=len(ids), timeout=12) for ids in inputs))
            print(vectors)
    # batcher가 먼저 종료되고 그다음 channel이 닫힙니다.

asyncio.run(main())
```

- gRPC 오류는 원래 `grpc.aio.AioRpcError` 그대로 각 대기 호출에 전달합니다.
- 자동 재시도는 하지 않습니다. 채널에 설정한 재시도 정책은 별도입니다.
- RPC timeout은 dispatch부터, submit timeout은 큐 진입부터 측정합니다.
- 한 요청이 timeout/cancel되어도 같은 배치의 다른 요청 때문에 공유 RPC는 유지됩니다.
- streaming RPC는 지원하지 않습니다. 배치 입력 한 개, 배치 결과 한 개인 RPC를 사용하세요.
- `encode`/`decode`는 빠른 동기 변환 함수여야 합니다.
- [공식 gRPC AsyncIO API](https://grpc.github.io/grpc/python/grpc_asyncio.html)

`examples/grpc_demo.py`는 **실제 gRPC transport** 위에 JSON 직렬화를 사용하는 독립 예시입니다.
Protobuf 생성 도구 없이 두 localhost 모의 노드로 전송하고 결과를 검증할 수 있습니다.
실제 GPU 추론 속도를 측정하는 예제는 아닙니다.

## 여러 GPU 노드 앞단에서 사용

```text
동시 호출 → AsyncBatcher → process_batch → 기존 LB / 선택한 GPU 노드
```

배처는 transport를 모르므로 HTTP·gRPC 모두 연결할 수 있습니다.
`examples/http_callback.py`는 표준 라이브러리 HTTP 콜백을 보여 줍니다.
원격 endpoint는 **배치 입력을 받아 배치 결과를 돌려주는 API**여야 합니다.

노드 선택은 processor에서 수행하거나 기존 로드밸런서에 맡기세요.
`max_concurrent_batches`는 이 배처의 전체 동시 실행 상한이며 **노드별 제한이 아닙니다**.
데모의 round-robin은 장애 감지·노드별 부하 추적이 없는 사용 예시입니다.
필요하면 호출자가 노드별 semaphore 또는 노드별 batcher를 소유하면 됩니다.

모델·버전·정밀도·옵션이 다른 입력은 배처를 분리하세요. 한 batcher에 섞으면
라이브러리가 호환성을 판별하지 않습니다. 요청의 인증·tenant 정보를 processor의
암묵적인 contextvars로 전달하지 마세요. 배치가 다른 요청 문맥을 합치므로 필요한
정보는 item에 명시하고 공유가 불가능한 입력은 별도 배처로 격리해야 합니다.

인스턴스는 **단일 이벤트 루프·프로세스 소유**입니다. 프로세스를 여러 개 띄우면
각자의 큐가 생깁니다. 분산 큐·노드 검색·헬스체크·재시도·GPU 배치는 이 패키지의 범위에
포함하지 않습니다. 이미 배칭하는 원격 서버 앞에 추가 배칭을 두면 대기가 중복될 수
있으므로 측정 후 사용하세요.

## 대기·과부하·종료 규칙

| 설정/동작 | 의미 |
|---|---|
| `max_wait_ms=2` | 첫 큐 입력 기준 배치 수집 기한. 가득 찼거나 다음 입력이 안 들어가면 먼저 실행 |
| `max_pending_items=1024` | 큐 + 실행 중 입력 수 상한. 초과 시 즉시 `QueueFull` |
| `max_pending_tokens=None` | 선택적 큐 + 실행 중 실제 size 합 상한. padding 비용과 별도 |
| `max_concurrent_batches=1` | processor의 동시 배치 실행 수 상한 |
| `submit(..., timeout=...)` | 해당 입력의 큐 진입부터 결과 대기까지. 전처리 시간 제외 |
| `aclose()` | 새 입력 거부, 큐를 즉시 flush하고 실행이 끝날 때까지 drain |
| `aclose(drain=False)` | 큐 입력에 `BatcherClosed`, 이미 실행 중인 배치는 완료까지 대기 |
| `aclose(timeout=...)` | 기다리는 시간만 제한. 종료 작업은 계속되며 다시 aclose 가능 |

수집 기한은 응답 SLA가 아닙니다. 실행 슬롯을 기다리는 시간이 추가됩니다.
큐에 있는 입력의 취소는 즉시 제거되어 용량을 반환합니다. 이미 실행된 입력은
클라이언트가 취소해도 실제 배치가 끝나기 전까지 용량을 점유합니다.
스레드/GPU 작업을 강제로 중단하지 않습니다. 멈출 수 있는 원격 작업에는 processor
수준의 네트워크 timeout을 설정하세요. processor 안에서 같은 배처에 재귀 submit하거나
종료를 기다리지 마세요.

`size`, 입력/설정 유효성 오류는 `ValueError`, 초과 입력은 `ItemTooLarge`, 큐 초과는
`QueueFull`, 닫힌 배처는 `BatcherClosed`, 결과 개수 불일치는 `BatchResultError`입니다.
processor 오류는 배치 내 아직 대기 중인 호출에 전달되며 이후 배치 처리는 계속됩니다.

## 계측

`stats()`는 제출·성공·실패·취소·타임아웃·거절 건수, 큐/실행 중 입력·토큰,
실행 중 배치 수를 반환합니다. `on_batch(event)`로 배치별 입력 수·실제 토큰·정책 비용,
최장 큐 대기·processor 실행 시간을 수집할 수 있습니다. 각 요청의 지연 분포는
호출자에서 별도로 측정하세요. 로컬 스레드/네트워크 콜백 시간은 순수 GPU 시간이 아닙니다.

observer는 짧은 동기 함수여야 합니다. 일반 예외는 격리하고 `observer_errors`로 셉니다.
토큰 비용은 실행에 들어간 입력 기준이며, 실행 도중 취소된 입력도 포함합니다.
성공/실패는 Future를 완료한 시점, 취소/타임아웃은 호출자가 대기를 중단한 시점의
카운터이므로 경계 시점의 취소와 겹칠 수 있습니다. 과금 집계 용도로 쓰지 마세요.

## 개발과 검증 범위

테스트는 `unittest` 기반이며 요청 수·두 토큰 정책·padding 반올림·FIFO 경계·저부하 flush,
혼합 길이 400개, 취소·타임아웃·상한·오류 복구·종료·동시 실행·결과 순서를 검증합니다.
선택적 gRPC 통합 테스트는 localhost 서버 2개를 띄워 실제 RPC, metadata,
오류·deadline·공유 RPC의 부분 timeout을 검사합니다. GPU 추론·운영 네트워크 성능·TLS
연동은 이 라이브러리 테스트에 포함하지 않습니다.
