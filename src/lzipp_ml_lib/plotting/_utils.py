import polars as pl

PolarsFrame = pl.DataFrame | pl.LazyFrame


def ensure_collected(data: PolarsFrame) -> pl.DataFrame:
    """Evaluates LazyFrames to DataFrames, passes DataFrames through."""
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    return data


def numeric_columns(df: pl.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    return [c for c in df.columns if c not in exclude and df[c].dtype.is_numeric()]
