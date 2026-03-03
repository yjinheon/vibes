
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
    eda.pairgrid_with_corr(cols=[c for c in ["age","fare","pclass"] if c in df.columns], height=2.0)
    plt.show()
except Exception as e:
    print("PairGrid skipped:", e)

# 2) Preprocess demo: impute + scale + rare bucket + one-hot
df2 = simple_impute(eda.df, num_strategy="median", cat_fill="Missing")
if "embarked" in df2.columns:
    df2 = rare_bucket(df2, "embarked", threshold=0.05)
    df2 = one_hot(df2, cols=["embarked"], drop_first=True)

num_cols = [c for c in df2.columns if str(df2[c].dtype).startswith(("int","float"))]
df2_scaled = standard_scale(df2, cols=num_cols)

# 3) Feature selection demos
if "survived" in df2_scaled.columns:
    print("=== Mutual Information (approx) ===")
    print(mutual_info_approx(df2_scaled, target="survived").head(), "\n")

print("=== VIF (relative) ===")
print(compute_vif_matrix(df2_scaled, cols=[c for c in ["age","fare","sibsp","parch"] if c in df2_scaled.columns]), "\n")

# 4) Time features demo (item 9 emphasized)
# Create a synthetic timestamp if missing
if "timestamp" not in df2_scaled.columns:
    df2_scaled["timestamp"] = pd.date_range("1912-04-10", periods=len(df2_scaled), freq="H")

df_tf = time_features.add_calendar_parts(df2_scaled, "timestamp", prefix="ts")
df_tf = time_features.add_cyclical(df_tf, "ts_hour", max_val=24)
df_tf = time_features.add_elapsed_since(df_tf, "timestamp", unit="days", prefix="ts")
df_tf = time_features.add_diff_between(df_tf, "timestamp", "timestamp", unit="hours", prefix="self_diff")  # zero
df_tf = time_features.add_lags(df_tf, key=df_tf.index.name or "idx", dt_col="timestamp", value_col="fare" if "fare" in df_tf.columns else df_tf.columns[0], lags=(1,2))
df_tf = time_features.add_leads(df_tf, key=df_tf.index.name or "idx", dt_col="timestamp", value_col="fare" if "fare" in df_tf.columns else df_tf.columns[0], leads=(1,))
df_tf = time_features.add_rolling_stats(df_tf, key=df_tf.index.name or "idx", dt_col="timestamp", value_col="fare" if "fare" in df_tf.columns else df_tf.columns[0], windows=(3,7))

print("=== Time features created (head) ===")
print(df_tf.filter(regex="^ts_|_lag_|_lead_|roll").head())

print("\\nAll done.")
