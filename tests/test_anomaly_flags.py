import pandas as pd
from ml_walmart.data_preparation import run_pipeline
from ml_walmart.classes.anomaly import Anomaly
from ml_walmart.classes.structural_change import StructuralChange

def test_anomaly_flag_is_true_on_exact_date():
    df = pd.DataFrame({
        "Store": [18, 18, 18],
        "Date": ["26-08-2011", "02-09-2011", "09-09-2011"],
        "Weekly_Sales": [100, 200, 150]
    })

    anomalies = [Anomaly(store=18, date="2011-09-02")]
    structural_changes = []

    result = run_pipeline(df, anomalies, structural_changes)

    flagged = result.loc[result["Date"] == "2011-09-02", "Anomaly_Flag"]

    assert flagged.iloc[0] == 1

def test_structural_change_flag_in_range():
    df = pd.DataFrame({
        "Store": [30, 30, 30],
        "Date": ["20-08-2011", "01-09-2011", "05-10-2011"],
        "Weekly_Sales": [100, 200, 150]
    })

    changes = [
        StructuralChange(
            store=30,
            start_date="2011-08-25",
            end_date="2011-09-29"
        )
    ]

    result = run_pipeline(df, [], changes)

    assert result.loc[
        result["Date"] == "2011-09-01",
        "Structural_Change_Flag"
    ].iloc[0] == 1
    assert result.loc[
        result["Date"] == "2011-08-20", 
        "Structural_Change_Flag"].iloc[0] == 0
