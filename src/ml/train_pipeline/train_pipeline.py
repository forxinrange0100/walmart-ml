import time
import pickle
import mlflow
import pandas as pd
from statsforecast import StatsForecast
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse
from ml.config import Models
from statsforecast.models import AutoARIMA
from ml.train_pipeline.nodes.preprocess import preprocess
from ml.train_pipeline.classes.artifacts import Artifacts
from ml.train_pipeline.nodes.log_evaluation_metrics import log_evaluation_metrics

def run_pipeline(walmart_minimal: pd.DataFrame, artifacts_paths: Models) -> dict[str, Artifacts]:
    pipeline_artifacts = {}
    walmart_minimal_arima, walmart_minimal_sarima= preprocess(walmart_minimal)
    shared_runtime = {
        "freq": "W",
        "n_jobs": -1
    }
    shared_cv = {
        "h": 12,
        "n_windows": 3,
    }
    configs = [
        {
            "input": walmart_minimal_arima,
            "input_name": "walmart_minimal_arima",
            "model": AutoARIMA(alias="ARIMA"),
            "name": "ARIMA",
            "model_params": None,
            "cv_preds_path": artifacts_paths.arima.cv_predictions,
            "metrics_path": artifacts_paths.arima.metrics,
            "model_path": artifacts_paths.arima.model
        },
        {
            "input": walmart_minimal_sarima,
            "input_name": "walmart_minimal_sarima",
            "model": AutoARIMA(alias="SARIMA", season_length=52),
            "name": "SARIMA",
            "model_params": {
                "season_length": 52
            },
            "cv_preds_path": artifacts_paths.sarima.cv_predictions,
            "metrics_path": artifacts_paths.sarima.metrics,
            "model_path": artifacts_paths.sarima.model
        }
    ]

    for config in configs:
        with mlflow.start_run(run_name=config["name"]):
            start = time.time()
            mlflow.log_params({
                "model": config["name"],
                **(config["model_params"] or {}),
                "dataset": config["input_name"],
                **shared_runtime,
                **shared_cv
            })
            sf = StatsForecast(
                models=[config["model"]],
                **shared_runtime
            )
            cv = sf.cross_validation(
                df=config["input"],
                **shared_cv
            )
            cv.to_csv(config["cv_preds_path"], index = False)
            mlflow.log_artifact(config["cv_preds_path"])
            eval_df = evaluate(cv, metrics=[mae, rmse])
            eval_df.to_csv(config["metrics_path"], index=False)
            mlflow.log_artifact(config["metrics_path"])
            log_evaluation_metrics(config, eval_df,config["name"])
            sf.fit(config["input"])
            with open(config["model_path"], "wb") as f:
                pickle.dump(sf, f)
            mlflow.log_artifact(config["model_path"])

            pipeline_artifacts[config["name"]] = Artifacts(
                    name=config["name"],
                    cv_preds= cv,
                    metrics=eval_df,
                    model=sf,
                )
            elapsed = time.time() - start
            mlflow.log_metric("fit_time_sec", elapsed)
    return pipeline_artifacts