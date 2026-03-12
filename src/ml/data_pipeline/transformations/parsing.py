import pandas as pd

def parsing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%d-%m-%Y"
    )
    return df
