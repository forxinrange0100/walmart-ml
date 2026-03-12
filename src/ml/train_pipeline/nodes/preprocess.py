import pandas as pd

def preprocess(walmart_minimal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    arima_stores_id = [4, 7, 14, 16, 18, 30, 33, 36, 38, 42, 44]
    walmart_minimal_arima = (
        walmart_minimal[walmart_minimal["unique_id"].isin(arima_stores_id)]
    ).sort_values(by=["unique_id", "ds"])
    walmart_minimal_sarima = (
        walmart_minimal[~walmart_minimal["unique_id"].isin(arima_stores_id)]
    ).sort_values(by=["unique_id", "ds"])
    return walmart_minimal_arima, walmart_minimal_sarima