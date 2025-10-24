
from dataclasses import dataclass
from typing import Optional

@dataclass
class EDAConfig:
    sample_n: Optional[int] = None
    sample_frac: Optional[float] = None
    random_state: int = 42

    outlier_method: str = "iqr"      # iqr|zscore|mad
    iqr_factor: float = 1.5
    zscore_threshold: float = 3.0
    mad_threshold: float = 3.5

    winsorize: bool = False
    remove_outliers: bool = False

    categorize_threshold: int = 30
    low_variance_threshold: float = 0.0
    rare_level_threshold: float = 0.01

    memory_downcast: bool = False
    parse_datetimes: bool = True

    max_levels_preview: int = 20
