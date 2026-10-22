"""Pure batch cost policy; size is caller-supplied, not inferred from text."""
from dataclasses import dataclass
from typing import Literal


def positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class BatchPolicy:
    """Item ceiling with optional sum-of-sizes or padded-size budget.

    ``max_tokens=None`` means count-only batching. ``size`` may represent tokens,
    image patches, or another caller-defined positive integer unit. Padded cost
    is count * max(size), optionally rounding the maximum up to pad_to_multiple.
    The cost is a scheduling proxy, not a guarantee about GPU memory or FLOPs.
    """

    max_items: int = 32
    max_tokens: int | None = None
    token_mode: Literal["sum", "padded"] = "sum"
    pad_to_multiple: int = 1

    def __post_init__(self) -> None:
        positive_int(self.max_items, "max_items")
        if self.max_tokens is not None:
            positive_int(self.max_tokens, "max_tokens")
        positive_int(self.pad_to_multiple, "pad_to_multiple")
        if self.token_mode not in ("sum", "padded"):
            raise ValueError("token_mode must be 'sum' or 'padded'")
        if self.token_mode != "padded" and self.pad_to_multiple != 1:
            raise ValueError("pad_to_multiple requires token_mode='padded'")

    def cost(self, count: int, total: int, longest: int) -> int:
        if self.max_tokens is None:
            return count
        if self.token_mode == "sum":
            return total
        width = ((longest + self.pad_to_multiple - 1) // self.pad_to_multiple
                 * self.pad_to_multiple)
        return count * width

    def fits(self, count: int, total: int, longest: int) -> bool:
        return count <= self.max_items and (
            self.max_tokens is None
            or self.cost(count, total, longest) <= self.max_tokens
        )
