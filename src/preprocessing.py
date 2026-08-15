import numpy as np
import pandas as pd


def replace_false_with_nan(df):
    """
    Replace the placeholder string 'False'
    with proper missing values (NaN).
    """

    df = df.copy()

    return df.replace("False", np.nan)


def summarize_missingness(df):
    """
    Return missing value summary.
    """

    summary = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": df.isna().mean() * 100
    })

    return summary.sort_values(
        "missing_percent",
        ascending=False
    )

def identify_numeric_columns(df, threshold=0.80):
    """
    Identify columns that are likely numeric.

    A column is considered numeric if at least
    `threshold` proportion of its NON-MISSING values
    can be converted to numeric.
    """

    numeric_columns = []

    for col in df.columns:

        non_missing = df[col].dropna()

        if len(non_missing) == 0:
            continue

        converted = pd.to_numeric(
            non_missing,
            errors="coerce"
        )

        valid_ratio = converted.notna().mean()

        if valid_ratio >= threshold:
            numeric_columns.append(col)

    return numeric_columns

def identify_categorical_columns(df, numeric_columns):
    """
    Return columns that are not numeric.
    """

    categorical_columns = [
        col for col in df.columns
        if col not in numeric_columns
    ]

    return categorical_columns

def convert_numeric_columns(df, numeric_columns):
    """
    Convert selected columns to numeric.
    Handles special formatting in sample_num_prog_mob_total.
    """

    df = df.copy()

    for col in numeric_columns:

        # Special handling for sample_num_prog_mob_total
        if col == "sample_num_prog_mob_total":

            df[col] = df[col].apply(
                lambda x: str(x).replace(".", "")
                if pd.notna(x) and str(x).count(".") > 1
                else x
            )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df

def dataset_summary(df):
    """
    Print basic dataset information.
    """

    print("=" * 60)
    print("Dataset Shape")
    print(df.shape)

    print("\nNumeric Columns:")
    print(len(df.select_dtypes(include=np.number).columns))

    print("\nCategorical Columns:")
    print(len(df.select_dtypes(exclude=np.number).columns))

    print("\nTotal Missing Values:")
    print(df.isna().sum().sum())