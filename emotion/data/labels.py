from pathlib import Path

import pandas as pd
from emotion import root_dir
from emotion.utils import create_csv

LABELS_DIR = Path(root_dir / "data/raw/labels")
CSV_FILES = LABELS_DIR.iterdir()
EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
EMOTIONS_COLS = [f"Answer.{emotion}" for emotion in EMOTIONS]
SENTIMENT_COL = "Answer.sentiment"


def merge_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use 0 or 1 for presence of emotion and -1, 0 or 1 for polarity.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to transform.

    Returns
    -------
    pd.DataFrame

    """
    return pd.concat(
        [
            df["HITId"],
            df[EMOTIONS_COLS].applymap(lambda v: v > 0 and 1 or 0),
            df[SENTIMENT_COL].map(lambda v: v < 0 and -1 or v > 0 and 1 or v),
        ],
        axis=1,
    )


def merge_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only segments from which there is at most a disagreeement of 1 ordinality
    and take the median/mode as the 'winning' modality.

    Parameters
    ----------
    df : pd.DataFrame


    Returns
    -------
    pd.DataFrame
        The transformed DataFrame.
    """
    groupped = df.groupby(by="HITId")
    all_modalities = groupped[EMOTIONS_COLS + [SENTIMENT_COL]]

    return all_modalities.median()[(all_modalities.std(0) < 0.48).all(axis=1)]


def main():
    dfs = {csv_file.stem: pd.read_csv(csv_file) for csv_file in CSV_FILES}

    # Merge all dataframes
    df_init = pd.concat(dfs.values())

    # Relevant columns
    df = df_init[["HITId"] + EMOTIONS_COLS + [SENTIMENT_COL]]

    # Drop missing values
    df = df.dropna()

    df_merged_intensity = merge_intensity(df)
    df_merged_agreement = merge_agreement(df_merged_intensity)

    create_csv(root_dir / "data/interim/labels" / "interim.csv", df_merged_agreement)

    return df_merged_agreement


if __name__ == "__main__":
    main()
