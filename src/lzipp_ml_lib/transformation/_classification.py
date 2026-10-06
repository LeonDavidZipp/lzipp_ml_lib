from collections.abc import Sequence

import polars as pl


class ClassificationFeatures:
    def __init__(self): ...


def to_xgboost(df: pl.DataFrame) -> pl.DataFrame: ...


def categorical_to_enum(
    df: pl.DataFrame, columns: Sequence[str] | None = None
) -> pl.DataFrame:
    """Transforms categorical columns to enums. If columns are specified, will treat all of them
    as columns to be transformed, else it will transform all pl.String and pl.Categorical columns

    Args:
        df (pl.DataFrame): The data to be transformed.
        columns (Sequence[str] | None): The columns to be transformed to enums.

    Returns:
        pl.DataFrame: The transformed data.
    """
    if columns:
        cols = columns
    else:
        cols = [
            col
            for col, dtype in df.schema.items()
            if dtype in (pl.Categorical(), pl.String())
        ]
    values = {
        col: df.get_column(col).drop_nulls().drop_nans().unique().sort().to_list()
        for col in cols
    }
    frame_cols = df.schema.names()
    return df.with_columns(
        pl.col(col).cast(pl.Enum(val)) for col, val in values.items()
    ).select(frame_cols)


def numeric_to_bin(): ...
