import numpy as np
import pandas as pd
from scipy import stats

def _mode(s):
    if s.empty:
        return "-"
    counts = s.value_counts()
    if counts.iloc[0] == 1:
        return "no repeated value"
    return ", ".join(f"{v:g}" for v in sorted(counts[counts == counts.iloc[0]].index)[:3])


PARAMETERS = {
    "Count": lambda s: s.count(),
    "Missing": lambda s: s.isna().sum(),
    "Sum": lambda s: s.sum(),
    "Mean": lambda s: s.mean(),
    "Median": lambda s: s.median(),
    "Mode": lambda s: _mode(s),
    "Variance": lambda s: s.var(),
    "Std deviation": lambda s: s.std(),
    "Std error (SEM)": lambda s: s.sem(),
    "Coeff. of variation": lambda s: s.std() / s.mean() if s.mean() else np.nan,
    "Minimum": lambda s: s.min(),
    "Maximum": lambda s: s.max(),
    "Range": lambda s: s.max() - s.min(),
    "Q1 (25%)": lambda s: s.quantile(.25),
    "Q3 (75%)": lambda s: s.quantile(.75),
    "IQR": lambda s: s.quantile(.75) - s.quantile(.25),
    "Skewness": lambda s: s.skew(),
    "Kurtosis": lambda s: s.kurtosis(),
}


def numeric_columns(df: pd.DataFrame) -> list:
    # Called before any data is loaded too, so tolerate None.
    if df is None or not isinstance(df, pd.DataFrame):
        return []
    return df.select_dtypes(include=np.number).columns.tolist()


def summary_table(df: pd.DataFrame, columns: list, params: list) -> pd.DataFrame:
    rows = {}
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce")
        rows[col] = {p: PARAMETERS[p](s.dropna() if p != "Missing" else s) for p in params}
    return pd.DataFrame(rows).T


def pdf_samples(df: pd.DataFrame, mode: str, choice, row_range=None, columns=None) -> pd.Series:
    """Numeric sample for a density.

    mode == "Column": values of column `choice`, optionally limited to a 1-based (start, end) row range.
    mode == "Row":    numeric values of row number `choice` (1-based) across `columns`.
    """
    if mode == "Column":
        s = df[choice]
        if row_range:
            s = s.iloc[row_range[0] - 1:row_range[1]]
        return pd.to_numeric(s, errors="coerce").dropna().astype(float)
    row = df.iloc[int(choice) - 1]
    if columns:
        row = row[columns]
    return pd.to_numeric(row, errors="coerce").dropna().astype(float)


def density_curves(values: pd.Series, points: int = 400):
    """Returns x grid plus normal-PDF and KDE values and the fitted parameters."""
    mu, sigma = float(values.mean()), float(values.std(ddof=1))
    pad = 3 * sigma if sigma > 0 else 1.0
    lo, hi = min(values.min(), mu - pad), max(values.max(), mu + pad)
    x = np.linspace(lo, hi, points)
    normal = stats.norm.pdf(x, mu, sigma) if sigma > 0 else np.full_like(x, np.nan)
    kde_obj = None
    if values.nunique() > 1 and sigma > 0:
        try:
            kde_obj = stats.gaussian_kde(values)
        except (np.linalg.LinAlgError, ValueError):
            # Repeated or nearly identical observations can make KDE covariance singular.
            kde_obj = None
    kde = kde_obj(x) if kde_obj is not None else np.full_like(x, np.nan)
    return x, normal, kde, mu, sigma, kde_obj


# ---- Correlation ----
def correlation_table(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Pearson correlation coefficients for the selected numeric columns."""
    if len(columns) < 2:
        return pd.DataFrame()
    return df[columns].apply(pd.to_numeric, errors="coerce").corr(method="pearson")


def strongest_correlation(df: pd.DataFrame, columns: list):
    """Return the strongest absolute Pearson correlation among selected columns."""
    corr = correlation_table(df, columns)
    if corr.empty or len(corr.columns) < 2:
        return None
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if pd.notna(value):
                pairs.append((cols[i], cols[j], float(value)))
    if not pairs:
        return None
    return max(pairs, key=lambda x: abs(x[2]))


# ---- Hypothesis testing ----
def hypothesis_test(df: pd.DataFrame, test_name: str, col1: str, col2=None, null_value=0.0, alpha=0.05):
    """Run common hypothesis tests and return a result dictionary."""
    alpha = float(alpha)
    s1 = pd.to_numeric(df[col1], errors="coerce").dropna()
    if test_name == "One-sample t-test":
        if len(s1) < 2:
            raise ValueError("At least 2 numeric observations are required.")
        stat, p = stats.ttest_1samp(s1, float(null_value))
        h0 = f"mean({col1}) = {null_value:g}"
        ha = f"mean({col1}) ≠ {null_value:g}"
    elif test_name == "Independent two-sample t-test":
        if not col2:
            raise ValueError("Choose a second numeric column.")
        s2 = pd.to_numeric(df[col2], errors="coerce").dropna()
        if len(s1) < 2 or len(s2) < 2:
            raise ValueError("Each sample needs at least 2 numeric observations.")
        stat, p = stats.ttest_ind(s1, s2, equal_var=False, nan_policy="omit")
        h0 = f"mean({col1}) = mean({col2})"
        ha = f"mean({col1}) ≠ mean({col2})"
    elif test_name == "Paired t-test":
        if not col2:
            raise ValueError("Choose a second numeric column.")
        d = pd.concat([s1.rename("a"), pd.to_numeric(df[col2], errors="coerce").rename("b")], axis=1).dropna()
        if len(d) < 2:
            raise ValueError("At least 2 paired observations are required.")
        stat, p = stats.ttest_rel(d["a"], d["b"])
        h0 = f"mean({col1} - {col2}) = 0"
        ha = f"mean({col1} - {col2}) ≠ 0"
    elif test_name == "Pearson correlation significance":
        if not col2:
            raise ValueError("Choose a second numeric column.")
        d = pd.concat([s1.rename("a"), pd.to_numeric(df[col2], errors="coerce").rename("b")], axis=1).dropna()
        if len(d) < 3:
            raise ValueError("At least 3 paired observations are required.")
        stat, p = stats.pearsonr(d["a"], d["b"])
        h0 = f"ρ({col1}, {col2}) = 0"
        ha = f"ρ({col1}, {col2}) ≠ 0"
    else:
        raise ValueError("Unknown hypothesis test.")
    decision = "Reject H₀" if p < alpha else "Fail to reject H₀"
    remark = (f"At α = {alpha:g}, the p-value ({p:.4g}) is below α, so there is statistically significant evidence "
              f"against H₀."
              if p < alpha else
              f"At α = {alpha:g}, the p-value ({p:.4g}) is not below α, so there is not enough statistical evidence "
              f"to reject H₀.")
    return {
        "test": test_name, "statistic": float(stat), "p_value": float(p), "alpha": alpha,
        "h0": h0, "ha": ha, "decision": decision, "remark": remark,
        "n1": len(s1), "n2": len(s2) if col2 and test_name == "Independent two-sample t-test" else None
    }
