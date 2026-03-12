import pandas as pd
from statsforecast import StatsForecast

class Artifacts:
    def __init__(
            self, 
            name: str,
            cv_preds: pd.DataFrame, 
            metrics: pd.DataFrame, 
            model: StatsForecast,
            ):
        self.name = name
        self.model = model
        self.metrics = metrics
        self.cv_preds = cv_preds
    def __str__(self) -> str:
        return (
            f"Artifacts(name='{self.name}', "
            f"cv_preds_shape={self.cv_preds.shape}, "
            f"metrics_shape={self.metrics.shape}, "
            f"models={[m.alias for m in self.model.models]})"
        )

    def __repr__(self) -> str:
        return self.__str__()