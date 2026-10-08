from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class AggregationConfig:
    agg: Sequence[str]
    over: Sequence[str]


class AggregationFeatures:
    def __init__(self): ...

    def fit(
        self,
    ) -> Self: ...

    def mean(self): ...
