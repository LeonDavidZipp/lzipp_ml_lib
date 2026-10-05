import polars as pl

PolarsFrame = pl.DataFrame | pl.LazyFrame


def ensure_collected(data: PolarsFrame) -> pl.DataFrame:
    """Evaluates LazyFrames to DataFrames, passes DataFrames through."""
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    return data
