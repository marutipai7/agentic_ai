import io
import base64
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, Any
import matplotlib.pyplot as plt

def _figure_to_base64() -> str:
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def _generate_plots(df: pd.DataFrame) -> Dict[str, Any]:
    plots: Dict[str, Any] = {}
    numeric_df = df.select_dtypes(include=[np.number])

    # Correlation heatmap
    if numeric_df.shape[1] >= 2:
        plt.figure(figsize=(6, 5))
        corr = numeric_df.corr(numeric_only=True)
        sns.heatmap(corr, cmap='coolwarm', annot=False)
        plots['heatmap'] = _figure_to_base64()

    # Histograms and Boxplots for up to 6 numeric columns
    cols = list(numeric_df.columns)[:6]

    if cols:
        ## Historgram subplot
        n_cols = 2 # 2 plots per row
        n_rows = (len(cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
        axes = axes.flatten()
        for i, col in enumerate(cols):
            series = numeric_df[col].dropna()
            if series.empty:
                axes[i].axis('off')
                continue
            sns.histplot(series, kde=True, ax=axes[i], color='#6366F1')
            axes[i].set_title(f'Histogram of {col}')
        ## Turn off any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plots['histograms'] = _figure_to_base64()

        ## Boxplot subplot
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 2.5))
        axes = axes.flatten()
        for i, col in enumerate(cols):
            series = numeric_df[col].dropna()
            if series.empty:
                axes[i].axis('off')
                continue
            sns.boxplot(x=series, ax=axes[i], color='#22C55E')
            axes[i].set_title(f'Boxplot of {col}')
        ## Turn off any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plots['boxplots'] = _figure_to_base64()

    return plots
