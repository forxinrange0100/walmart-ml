import pandas as pd

class Datasets:
    def __init__(self, walmart_clean: pd.DataFrame, walmart_minimal: pd.DataFrame, walmart_no_new_flags: pd.DataFrame, walmart_full: pd.DataFrame):
        self.walmart_clean = walmart_clean
        self.walmart_minimal = walmart_minimal
        self.walmart_no_new_flags = walmart_no_new_flags
        self.walmart_full = walmart_full