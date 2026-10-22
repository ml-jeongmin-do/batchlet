# Contributing

## Development setup

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) from the repository root:

```bash
uv sync --locked --extra grpc
uv run --locked --extra grpc python -m unittest discover -s tests -v
uv run --locked mypy
uv build
```

`.python-version` selects Python 3.11. The `dev` group includes mypy;
the `grpc` extra enables integration tests against localhost servers.

After changing dependencies, run `uv lock` and review `pyproject.toml` and
`uv.lock`. Build artifacts are written to `dist/`.

## Changes

Describe the problem, the change, and the checks you ran. Add regression tests
for behavior changes. For documentation changes, check links and runnable examples.

Keep the core API small and transport dependencies optional.
