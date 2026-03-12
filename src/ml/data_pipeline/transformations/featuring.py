import pandas as pd
from ml.data_pipeline.classes.anomaly import Anomaly
from ml.data_pipeline.classes.structural_change import StructuralChange

def featuring(df:pd.DataFrame, anomalies: list[Anomaly], structural_changes: list[StructuralChange]) ->pd.DataFrame:
    df = df.copy()
    df["Anomaly_Flag"] = 0
    for a in anomalies:
        df.loc[(df["Store"]==a.store) & (df["Date"] == a.date), "Anomaly_Flag"] = 1
    df["Structural_Change_Flag"] = 0
    for s in structural_changes:
        df.loc[(df["Store"]==s.store) & df["Date"].between(s.start_date, s.end_date), "Structural_Change_Flag"] = 1
    return df
