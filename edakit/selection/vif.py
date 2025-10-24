
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def compute_vif_matrix(df: pd.DataFrame, cols=None):
    # 간단한 VIF 계산: (1 - R^2)^{-1}, 선형회귀 해석적 R^2 근사
    cols = cols or [c for c in df.columns if is_numeric_dtype(df[c])]
    X = df[cols].dropna()
    if X.shape[1] < 2: 
        return pd.DataFrame()
    X = (X - X.mean()) / X.std(ddof=0)
    X = X.fillna(0.0)

    XtX = X.T @ X
    try:
        inv = np.linalg.inv(XtX.values)
    except np.linalg.LinAlgError:
        # 정칙화가 필요할 때 작은 가중치 추가
        inv = np.linalg.pinv(XtX.values)
    # 각 변수에 대한 R^2 근사
    # 대각 원소의 역수를 이용한 간단 근사: VIF ~ diag(inv(XtX)) * (n-1)
    diag = np.diag(inv)
    vif = pd.Series(diag / diag.min(), index=cols)  # 스케일상 상대적 크기
    return vif.rename("vif").to_frame()
