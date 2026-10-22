"""Failures callers can map to their transport's error responses."""


class BatcherClosed(RuntimeError):
    """The batcher is closing or has closed."""


class QueueFull(RuntimeError):
    """Accepted work (queued plus executing) has reached an admission limit."""


class ItemTooLarge(ValueError):
    """An individual input cannot fit the batch or admission budget."""


class BatchResultError(RuntimeError):
    """A processor did not return one output per input."""
