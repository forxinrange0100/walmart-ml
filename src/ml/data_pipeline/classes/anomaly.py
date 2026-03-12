import pandas as pd

class Anomaly:
    def __init__(self, store: int, date: str):
        self.store = store
        self.date = pd.to_datetime(date)