
import pandas as pd

def class_imbalance_report(y: pd.Series):
    vc = y.value_counts(dropna=False)
    ratio = (vc / vc.sum() * 100).rename("ratio_%")
    return pd.concat([vc.rename("count"), ratio], axis=1)
