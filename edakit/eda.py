from typing import Optional, List, Dict
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

from .config import EDAConfig
from .utils.types import ensure_datetime_cols, downcast_numeric
from .utils.outliers import outlier_mask, iqr_bounds
from .diagnostics.types_table import types_table as _types_table
from .diagnostics.numeric import numeric_summary as _numeric_summary
from .diagnostics.categorical import categorical_summary as _categorical_summary
from .diagnostics.datetime_summary import datetime_summary as _datetime_summary
from .diagnostics.text_summary import text_summary as _text_summary
from .diagnostics.duplicates import duplicates_overview as _duplicates_overview
from .diagnostics.low_variance import low_variance_columns as _low_variance_columns
from .diagnostics.missing import missing_matrix as _missing_matrix
from .diagnostics.correlation import correlation_numeric as _correlation_numeric
from .diagnostics.association import association_categorical as _association_categorical
from .diagnostics.outliers_table import outliers_table as _outliers_table
from .viz.pairgrid import pairgrid_with_corr as _pairgrid_with_corr


class EDAKit:
    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        config: Optional[EDAConfig] = None,
    ):
        self.original = df.copy()
        self.cfg = config or EDAConfig()
        self.target = target
        self.df = self._prepare(self.original)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.sample_n is not None:
            df = df.sample(n=self.cfg.sample_n, random_state=self.cfg.random_state)
        elif self.cfg.sample_frac is not None:
            df = df.sample(
                frac=self.cfg.sample_frac, random_state=self.cfg.random_state
            )
        if self.cfg.parse_datetimes:
            df = ensure_datetime_cols(df, parse=True)
        if self.cfg.memory_downcast:
            df = downcast_numeric(df)
        return df

    def suggest_categoricals(self) -> List[str]:
        cands = []
        for c in self.df.columns:
            s = self.df[c]
            if (
                s.dtype == object
                and s.nunique(dropna=True) <= self.cfg.categorize_threshold
            ):
                cands.append(c)
        return cands

    def to_categoricals(self, cols: Optional[List[str]] = None) -> None:
        cols = cols or self.suggest_categoricals()
        for c in cols:
            self.df[c] = self.df[c].astype("category")

    # diagnostics
    def types_table(self):
        return _types_table(self.df)

    def numeric_summary(self):
        return _numeric_summary(
            self.df,
            self.cfg.outlier_method,
            self.cfg.iqr_factor,
            self.cfg.zscore_threshold,
            self.cfg.mad_threshold,
        )

    def categorical_summary(self, top_k: Optional[int] = None):
        return _categorical_summary(
            self.df, max_levels=top_k or self.cfg.max_levels_preview
        )

    def datetime_summary(self):
        return _datetime_summary(self.df)

    def text_summary(self):
        return _text_summary(self.df)

    def duplicates_overview(self, subset=None):
        return _duplicates_overview(self.df, subset=subset)

    def low_variance_columns(self):
        return _low_variance_columns(self.df, threshold=self.cfg.low_variance_threshold)

    def missing_matrix(self):
        return _missing_matrix(self.df)

    def correlation_numeric(self, method="pearson"):
        return _correlation_numeric(self.df, method=method)

    def association_categorical(self):
        return _association_categorical(self.df)

    def outliers_table(self):
        return _outliers_table(
            self.df,
            self.cfg.outlier_method,
            self.cfg.iqr_factor,
            self.cfg.zscore_threshold,
            self.cfg.mad_threshold,
        )

    # transforms
    def transform_outliers(self, cols: Optional[List[str]] = None) -> pd.DataFrame:
        df = self.df.copy()
        num_cols = cols or [c for c in df.columns if is_numeric_dtype(df[c])]
        for c in num_cols:
            s = df[c]
            m = outlier_mask(
                s,
                self.cfg.outlier_method,
                self.cfg.iqr_factor,
                self.cfg.zscore_threshold,
                self.cfg.mad_threshold,
            )
            if self.cfg.winsorize:
                lo, hi = iqr_bounds(s, self.cfg.iqr_factor)
                df.loc[s < lo, c] = lo
                df.loc[s > hi, c] = hi
            elif self.cfg.remove_outliers:
                df = df.loc[~m]
        return df

    # viz
    def pairgrid_with_corr(self, cols: Optional[List[str]] = None, height: float = 2.0):
        return _pairgrid_with_corr(self.df, cols=cols, height=height)

    def full_report(self) -> Dict[str, pd.DataFrame]:
        # user asked to exclude exporting; keep internal dict assembly
        return {
            "types": self.types_table(),
            "numeric": self.numeric_summary(),
            "categorical": self.categorical_summary(),
            "datetime": self.datetime_summary(),
            "text": self.text_summary(),
            "duplicates": self.duplicates_overview(),
            "low_variance": self.low_variance_columns(),
            "outliers": self.outliers_table(),
            "corr_numeric_pearson": self.correlation_numeric("pearson"),
            "assoc_categorical": self.association_categorical(),
            "missing_matrix": self.missing_matrix(),
        }
