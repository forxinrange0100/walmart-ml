import pandas as pd
from hydra import main
from ml.config import Config
from ml.data_pipeline.classes.anomaly import Anomaly
from ml.data_pipeline.classes.datasets import Datasets
from ml.data_pipeline.data_pipeline import run_pipeline
from ml.data_pipeline.classes.structural_change import StructuralChange

@main(
    version_base=None, 
    config_path="../../../conf", 
    config_name="config")
def main(cfg: Config) -> None:
    walmart_ds = pd.read_csv(cfg.paths.raw_data.walmart_original)
    anomalies = [
        Anomaly(store=18, date="2011-09-02")
    ]
    structural_changes = [
        StructuralChange(store=30, start_date="2011-08-25", end_date="2011-09-29"),
        StructuralChange(store=33, start_date="2010-02-05", end_date="2010-07-20"),
        StructuralChange(store=35, start_date="2010-02-05", end_date="2010-09-20"),
        StructuralChange(store=43, start_date="2010-02-05", end_date="2010-10-25")
    ]
    datasets: Datasets = run_pipeline(walmart_ds, anomalies, structural_changes)
    datasets.walmart_clean.to_parquet(cfg.paths.interim_data.walmart_clean, index=False)
    log_dataset_saved("walmart_clean", datasets.walmart_clean, cfg.paths.interim_data.walmart_clean)
    datasets.walmart_minimal.to_parquet(cfg.paths.processed_data.walmart_minimal, index=False)
    log_dataset_saved("walmart_minimal", datasets.walmart_minimal, cfg.paths.processed_data.walmart_minimal)
    datasets.walmart_no_new_flags.to_parquet(cfg.paths.processed_data.walmart_no_new_flags, index=False)
    log_dataset_saved("walmart_no_new_flags", datasets.walmart_no_new_flags, cfg.paths.processed_data.walmart_no_new_flags)
    datasets.walmart_full.to_parquet(cfg.paths.processed_data.walmart_full, index=False)
    log_dataset_saved("walmart_full", datasets.walmart_full, cfg.paths.processed_data.walmart_full)

def log_dataset_saved(name: str, df: pd.DataFrame, path: str) -> None:
    print(
        f"[DATASET SAVED] {name:<22} "
        f"shape={df.shape} "
        f"path='{path}'"
    )

if __name__ == "__main__":
    main()
