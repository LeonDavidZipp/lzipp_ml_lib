import polars as pl
from hypothesis import strategies as st

NUMERIC_DTYPES = [pl.Float64, pl.Int64]


@st.composite
def frames(draw: st.DrawFn, min_rows: int = 0) -> pl.DataFrame:
    """Small mixed frames: float columns with nulls and NaNs, int columns with
    zeros, string columns with nulls."""
    n = draw(st.integers(min_rows, 30))
    n_cols = draw(st.integers(1, 5))
    data: dict[str, pl.Series] = {}
    for i in range(n_cols):
        kind = draw(st.sampled_from(["float", "int", "str"]))
        name = f"{kind}{i}"
        if kind == "float":
            values = st.one_of(
                st.none(),
                st.just(float("nan")),
                st.floats(-1e6, 1e6, allow_nan=False),
            )
            data[name] = pl.Series(
                name, draw(st.lists(values, min_size=n, max_size=n)), pl.Float64
            )
        elif kind == "int":
            values = st.one_of(st.none(), st.integers(-5, 5))
            data[name] = pl.Series(
                name, draw(st.lists(values, min_size=n, max_size=n)), pl.Int64
            )
        else:
            values = st.one_of(st.none(), st.sampled_from(["a", "b", "c"]))
            data[name] = pl.Series(
                name, draw(st.lists(values, min_size=n, max_size=n)), pl.String
            )
    return pl.DataFrame(list(data.values()))
