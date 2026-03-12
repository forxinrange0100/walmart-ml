from dataclasses import dataclass

@dataclass
class Arima:
    model: str
    cv_predictions: str
    metrics: str
@dataclass
class Sarima:
    model: str
    cv_predictions: str
    metrics: str

@dataclass
class Models:
    arima: Arima
    sarima: Sarima

@dataclass
class RawData:
    walmart_original: str
@dataclass
class InterimData:
    walmart_clean: str
@dataclass
class ProcessedData:
    walmart_minimal: str
    walmart_no_new_flags: str
    walmart_full: str
@dataclass
class PathsConfig:
    raw_data: RawData
    interim_data: InterimData
    processed_data: ProcessedData
    models: Models
@dataclass
class Config:
    paths: PathsConfig
