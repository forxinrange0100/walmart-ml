import mlflow
import pandas as pd
def log_evaluation_metrics(config, eval_df: pd.DataFrame, name: str):
    mae_df = eval_df[eval_df["metric"] == "mae"]
    rmse_df = eval_df[eval_df["metric"] == "rmse"]
    mae_val = mae_df[name].mean()
    rmse_val = rmse_df[name].mean()
    mlflow.log_metric("mae", mae_val)
    mlflow.log_metric("rmse", rmse_val)
    for _, row in mae_df.iterrows():
        store_id = row["unique_id"]
        mlflow.log_metric(f"mae_store_{store_id}", row[name])
    for _, row in rmse_df.iterrows():
        store_id = row["unique_id"]
        mlflow.log_metric(f"rmse_store_{store_id}", row[name])
    print(config["name"], f"mae: {mae_val}", f"rmse: {rmse_val}")