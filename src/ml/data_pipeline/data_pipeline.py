import pandas as pd
from ml.data_pipeline.classes.anomaly import Anomaly
from ml.data_pipeline.classes.datasets import Datasets
from ml.data_pipeline.transformations.parsing import parsing
from ml.data_pipeline.transformations.featuring import featuring
from ml.data_pipeline.classes.structural_change import StructuralChange

def run_pipeline(walmart_ds: pd.DataFrame, anomalies: list[Anomaly], structural_changes: list[StructuralChange]) -> Datasets:
    walmart_ds = parsing(walmart_ds)
    walmart_ds = walmart_ds.sort_values(by="Date", ascending=True)
    walmart_ds = featuring(walmart_ds, anomalies, structural_changes)
    walmart_ds = walmart_ds.rename(columns={
        "Date": "ds",
        "Weekly_Sales": "y",
        "Store": "unique_id"
    })
    walmart_ds = walmart_ds.sort_values(["unique_id", "ds"])
    walmart_minimal = walmart_ds[["unique_id", "ds", "y"]].copy()
    walmart_no_new_flags = walmart_ds.drop(["Anomaly_Flag", "Structural_Change_Flag"], axis = 1).copy()
    walmart_full = walmart_ds.copy()
    return Datasets(
        walmart_clean=walmart_ds, 
        walmart_minimal=walmart_minimal, 
        walmart_no_new_flags=walmart_no_new_flags, 
        walmart_full=walmart_full)
