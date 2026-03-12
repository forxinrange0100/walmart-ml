import mlflow
import pandas as pd
from hydra import main
from ml.config import Config
from ml.train_pipeline.train_pipeline import run_pipeline

@main(
    version_base=None, 
    config_path="../../../conf", 
    config_name="config")

def main(cfg: Config) -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "walmart_models_train_pipeline"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location="/mlruns/"
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    walmart_minimal = pd.read_parquet(cfg.paths.processed_data.walmart_minimal)
    artifacts_paths = cfg.paths.models
    pipeline_artifacts = run_pipeline(walmart_minimal, artifacts_paths)
    print(pipeline_artifacts)
if __name__ == "__main__":
    main()
