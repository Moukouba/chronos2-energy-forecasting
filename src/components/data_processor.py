"""
Data processor component
"""

import re
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from tsflex.features import FeatureCollection, MultipleFeatureDescriptors


class DataProcessor:
    """Process and engineer features for time series forecasting"""
    
    def __init__(self, target_columns: List[str], lag_set: List[int] = None, 
                 rolling_windows: List[int] = None):
        self.target_columns = target_columns
        self.lag_set = lag_set or [1, 2, 3, 6, 12, 24]
        self.rolling_windows = rolling_windows or [3, 6, 12, 24]
        self.past_cov_columns: List[str] = []
        self.future_cov_columns: List[str] = []
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from parquet file."""
        return pd.read_parquet(data_path, engine='fastparquet')
    
    def _rename_tsflex_cols(self, columns: pd.Index) -> Dict[str, str]:
        """Rename tsflex auto-generated columns."""
        mapping = {}
        for col in columns:
            parts = col.split("__")
            if len(parts) < 3:
                continue
            series = parts[0]
            func = parts[1]
            w_raw = parts[2].replace("w=", "")
            try:
                hours = int(pd.Timedelta(w_raw).total_seconds() // 3600)
                w_val = str(hours)
            except Exception:
                w_val = w_raw
            mapping[col] = f"{series}_{func}_{w_val}"
        return mapping
    
    def _build_rolling_fc(self, series_names: List[str], 
                         extra_funcs: bool = True) -> FeatureCollection:
        """Build rolling feature collection."""
        def roll_mean(x: np.ndarray) -> float:
            return float(np.nanmean(x))
        
        def roll_std(x: np.ndarray) -> float:
            return float(np.nanstd(x, ddof=1)) if len(x) > 1 else 0.0
        
        def roll_min(x: np.ndarray) -> float:
            return float(np.nanmin(x))
        
        def roll_max(x: np.ndarray) -> float:
            return float(np.nanmax(x))
        
        def roll_range(x: np.ndarray) -> float:
            return float(np.nanmax(x) - np.nanmin(x))
        
        def roll_skew(x: np.ndarray) -> float:
            if len(x) < 3:
                return 0.0
            mu, sigma = np.nanmean(x), np.nanstd(x)
            return float(np.nanmean(((x - mu) / (sigma + 1e-8)) ** 3))
        
        funcs = [roll_mean, roll_std]
        if extra_funcs:
            funcs.extend([roll_min, roll_max, roll_range, roll_skew])
        
        return FeatureCollection(
            MultipleFeatureDescriptors(
                functions=funcs,
                series_names=series_names,
                windows=[f"{w}h" for w in self.rolling_windows],
                strides="1h",
            )
        )
    
    def fill_time_series_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values in time series data."""
        df = df.copy()
        lag_cols = [c for c in df.columns if "_lag_" in c]
        roll_cols = [c for c in df.columns if "roll_" in c]
        diff_cols = [c for c in df.columns if "diff_" in c]
        ewm_cols = [c for c in df.columns if "ewm_" in c]
        
        df[lag_cols] = df[lag_cols].fillna(0)
        df[roll_cols] = df[roll_cols].ffill().fillna(0)
        df[diff_cols] = df[diff_cols].fillna(0)
        df[ewm_cols] = df[ewm_cols].ffill().fillna(0)
        
        return df.fillna(0)
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features from raw data."""
        df = df.copy()
        
        # Future covariates: raw forecast inputs + aggregations
        load_cols = [c for c in df.columns if "mtlf_fc" in c]
        wind_cols = [c for c in df.columns if "_ws_" in c and "_fc" in c]
        dew_cols = [c for c in df.columns if "_td_" in c and "_fc" in c]
        
        if load_cols:
            df["total_forecast_load"] = df[load_cols].sum(axis=1)
        if wind_cols:
            df["avg_wind_fc"] = df[wind_cols].mean(axis=1)
        if dew_cols:
            df["avg_dewpoint_fc"] = df[dew_cols].mean(axis=1)
        
        # Past covariates: rolling stats
        shifted_df = df[self.target_columns].shift(1)
        roll_fc = self._build_rolling_fc(self.target_columns)
        roll_feats = roll_fc.calculate(shifted_df, return_df=True, approve_sparsity=True)
        roll_feats = roll_feats.rename(columns=self._rename_tsflex_cols(roll_feats.columns))
        df = df.join(roll_feats)
        
        # Past covariates: lags, EWM, diff
        for tcol in self.target_columns:
            for lag in self.lag_set:
                df[f"{tcol}_lag_{lag}"] = df[tcol].shift(lag)
            df[f"{tcol}_ewm_12"] = df[tcol].shift(1).ewm(span=12).mean()
            df[f"{tcol}_ewm_24"] = df[tcol].shift(1).ewm(span=24).mean()
            df[f"{tcol}_diff_1"] = df[tcol].diff(1)
            df[f"{tcol}_diff_24"] = df[tcol].diff(24)
        
        # Market structure
        if len(self.target_columns) >= 3:
            df["michigan_minn_spread"] = df[self.target_columns[1]] - df[self.target_columns[2]]
            df["aeci_michigan_spread"] = df[self.target_columns[0]] - df[self.target_columns[1]]
        
        # System-level features
        load_cols_real = [c for c in df.columns if "mtlf_fc" in c]
        if load_cols_real:
            df["load_imbalance"] = df[load_cols_real].max(axis=1) - df[load_cols_real].min(axis=1)
            df["load_mean"] = df[load_cols_real].mean(axis=1)
        
        wind_real = [c for c in df.columns if "_ws_" in c and "_fc" not in c]
        if wind_real:
            df["wind_variability"] = df[wind_real].std(axis=1)
        
        return self.fill_time_series_nans(df)
    
    def classify_covariates(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Classify columns into past and future covariates."""
        _FUTURE_PATTERNS = [
            "_fc", "total_forecast", "avg_wind", "avg_dewpoint"
        ]
        
        _PAST_PATTERNS = [
            r"_lag_\d+",
            r"roll_\w+_\d+",
            r"_ewm_\d+",
            r"_diff_\d+",
            r"_spread$",
            "load_imbalance",
            "load_mean",
            "wind_variability",
        ]
        
        def _matches(col: str, patterns: List[str]) -> bool:
            for p in patterns:
                if any(c in p for c in r"^$\\d+"):
                    if re.search(p, col):
                        return True
                else:
                    if p in col:
                        return True
            return False
        
        past_cov = []
        future_cov = []
        
        for col in df.columns:
            if col in self.target_columns:
                continue
            if _matches(col, _FUTURE_PATTERNS):
                future_cov.append(col)
            elif _matches(col, _PAST_PATTERNS):
                past_cov.append(col)
            else:
                if "_fc" in col:
                    future_cov.append(col)
                else:
                    past_cov.append(col)
        
        self.past_cov_columns = past_cov
        self.future_cov_columns = future_cov
        
        return past_cov, future_cov
    
    def create_time_series(self, df: pd.DataFrame, 
                          freq: str = "H") -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
        """Create DARTS TimeSeries objects."""
        train_series = TimeSeries.from_dataframe(
            df, value_cols=self.target_columns, freq=freq
        )
        train_past_cov = TimeSeries.from_dataframe(
            df, value_cols=self.past_cov_columns, freq=freq
        )
        train_future_cov = TimeSeries.from_dataframe(
            df, value_cols=self.future_cov_columns, freq=freq
        )
        
        return train_series, train_past_cov, train_future_cov
    
    def scale_covariates(self, train_past: TimeSeries, train_future: TimeSeries,
                        val_past: TimeSeries = None, val_future: TimeSeries = None
                        ) -> Tuple[TimeSeries, TimeSeries, TimeSeries, TimeSeries]:
        """Scale covariates using StandardScaler."""
        past_scaler = Scaler(StandardScaler())
        future_scaler = Scaler(StandardScaler())
        
        train_past_scaled = past_scaler.fit_transform(train_past)
        train_future_scaled = future_scaler.fit_transform(train_future)
        
        val_past_scaled = past_scaler.transform(val_past) if val_past is not None else None
        val_future_scaled = future_scaler.transform(val_future) if val_future is not None else None
        
        return train_past_scaled, train_future_scaled, val_past_scaled, val_future_scaled
