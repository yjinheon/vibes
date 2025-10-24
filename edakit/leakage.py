
import pandas as pd

CHECKLIST = [
    "Target timestamp보다 이후에만 알 수 있는 집계 변수 포함 여부",
    "ID/키가 타깃과 1:1 연결되는지 (고유값=행수)",
    "Fold 분할 전에 표준화/인코딩/선택이 수행되지 않았는지",
    "Target encoding 시 CV 내에서만 fit 했는지",
]

def leakage_quick_checks(df: pd.DataFrame, target: str):
    notes = []
    if target in df.columns:
        if df[target].isna().any():
            notes.append("타깃에 결측 존재")
        nunique = df[target].nunique(dropna=True)
        notes.append(f"타깃 고유값 수: {nunique}")
    return {"checks": CHECKLIST, "notes": notes}
