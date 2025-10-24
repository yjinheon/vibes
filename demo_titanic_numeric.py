import pandas as pd
import matplotlib.pyplot as plt

from edakit import EDAKit, EDAConfig
from edakit.preprocessing.impute import simple_impute
from edakit.preprocessing.scale import standard_scale, robust_scale
from edakit.encoding.text_cat import one_hot, rare_bucket
from edakit.selection.vif import compute_vif_matrix
from edakit.selection.mutual_info import mutual_info_approx
from edakit.drift.psi import psi_frame
from edakit.features import time_features

# 1) Load Titanic (try seaborn; else fallback to sample)
try:
    import seaborn as sns

    df = sns.load_dataset("titanic")
except Exception:
    df = pd.read_csv("titanic_sample.csv")

cfg = EDAConfig(memory_downcast=True, parse_datetimes=True, outlier_method="iqr")
eda = EDAKit(df, target="survived", config=cfg)

# object -> category conversion by suggestion
eda.to_categoricals()

print("=== Types Table ===")
print(eda.types_table().head(), "\n")

print("=== Numeric Summary ===")
print(eda.numeric_summary().head(), "\n")

print("=== Categorical (top 10) ===")
print(eda.categorical_summary(top_k=10).head(15), "\n")

# Pairwise viz (if seaborn present)
try:
    eda.pairgrid_with_corr(
        cols=[c for c in ["age", "fare", "pclass"] if c in df.columns], height=2.0
    )
    plt.show()
except Exception as e:
    print("PairGrid skipped:", e)
