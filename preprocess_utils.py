import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

def _compute_overview_and_stats(df: pd.DataFrame) -> Dict[str, Any]:
    overview = {
        'total_rows': int(df.shape[0]),
        'total_columns': int(df.shape[1]),
        'missing_values': int(df.isna().sum().sum()),
        'numeric_columns': int(df.select_dtypes(include=[np.number]).shape[1]),
    }

    column_info = []
    for col in df.columns:
        series = df[col]
        missing_percent = float(series.isna().mean() * 100.0)
        dtype_str = str(series.dtype)
        unique_values = int(series.nunique(dropna=True))
        column_info.append({
            'name': col,
            'dtype': dtype_str,
            'missing_percent': missing_percent,
            'unique_values': unique_values,
        })

    numeric_df = df.select_dtypes(include=[np.number])
    stats: Dict[str, Dict[str, float]] = {}
    if not numeric_df.empty:
        desc = numeric_df.describe().to_dict()
        # transpose to {col: {metric: value}}
        for col, metrics in desc.items():
            stats[col] = {}
            for metric, value in metrics.items():
                # Cast numpy types to python native for JSON
                if pd.isna(value):
                    stats[col][metric] = None
                else:
                    stats[col][metric] = float(value)

    return {
        'data_overview': overview,
        'column_info': column_info,
        'statistics': stats,
    }

def _apply_preprocessing(df: pd.DataFrame, steps: list[str]) -> pd.DataFrame:
    result = df.copy()
    numeric_cols = list(result.select_dtypes(include=[np.number]).columns)
    cat_cols = list(result.select_dtypes(exclude=[np.number]).columns)

    if 'drop_missing' in steps:
        result = result.dropna()
        numeric_cols = list(result.select_dtypes(include=[np.number]).columns)
        cat_cols = list(result.select_dtypes(exclude=[np.number]).columns)

    if 'fill_mean' in steps and numeric_cols:
        result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].mean())

    if 'fill_median' in steps and numeric_cols:
        result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].median())

    if 'fill_mode' in steps and cat_cols:
        for c in cat_cols:
            mode_val = result[c].mode(dropna=True)
            if not mode_val.empty:
                result[c] = result[c].fillna(mode_val.iloc[0])

    # One-hot encoding
    if 'one_hot' in steps and cat_cols:
        result = pd.get_dummies(result, columns=cat_cols, drop_first=True)
        numeric_cols = list(result.select_dtypes(include=[np.number]).columns)

    ## Outliear Treatment using IQR Mapping
    if 'treat_outliers' in steps and numeric_cols:
        for col in numeric_cols:
            Q1 = result[col].quantile(0.25)
            Q3 = result[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            result[col] = np.where(result[col] < lower_bound, lower_bound,
                                   np.where(result[col] > upper_bound, upper_bound, result[col]))

    # Scaling / Normalization (apply to numeric columns only)
    numeric_cols = list(result.select_dtypes(include=[np.number]).columns)
    if numeric_cols:
        if 'standardize' in steps:
            scaler = StandardScaler()
            result[numeric_cols] = scaler.fit_transform(result[numeric_cols])
        if 'minmax' in steps:
            scaler = MinMaxScaler()
            result[numeric_cols] = scaler.fit_transform(result[numeric_cols])
        if 'robust' in steps:
            scaler = RobustScaler()
            result[numeric_cols] = scaler.fit_transform(result[numeric_cols])
        if 'normalize_l2' in steps:
            normalizer = Normalizer(norm='l2')
            result[numeric_cols] = normalizer.fit_transform(result[numeric_cols])

    return result