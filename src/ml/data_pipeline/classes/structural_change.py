import pandas as pd

class StructuralChange:
    def __init__(self, store: int, start_date: str, end_date: str):
        self.store = store
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)