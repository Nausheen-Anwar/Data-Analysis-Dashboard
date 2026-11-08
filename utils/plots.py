import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from utils.palette import CMAP, INK as _INK

CHARTS = ["Line plot", "Scatter plot", "Bar chart", "Area chart", "Histogram", "Box plot",
          "Violin plot", "Density plot", "Pie chart", "Count plot", "Heatmap (correlation)"]

NEEDS_XY = {"Line plot", "Scatter plot", "Bar chart", "Area chart"}
NEEDS_ONE = {"Histogram", "Box plot", "Violin plot", "Density plot", "Pie chart", "Count plot"}


PALETTES = {
    # High-contrast chart colours.
    "teal": ["#176B78", "#243B53", "#2A9D8F", "#D97706", "#6D4C9A", "#C44536"],
    "deep": ["#176B78", "#243B53", "#2A9D8F", "#D97706", "#6D4C9A", "#C44536"],
}
INK = _INK


def apply_limits(ax, xlim, ylim):
    if xlim[0] is not None or xlim[1] is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim[0] is not None or ylim[1] is not None:
        ax.set_ylim(ylim[0], ylim[1])


def make_chart(df, kind, x=None, ys=None, col=None, heat_cols=None, title="", xlim=(None, None),
               ylim=(None, None), grid=True, palette="deep", size=(10, 5.5)):
    """Plot the selected source values without replacing category labels with row numbers."""
    sns.set_theme(style="whitegrid" if grid else "white", palette=PALETTES.get(palette, palette),
                  rc={"figure.facecolor": "#F8FBFC", "axes.facecolor": "#F8FBFC", "savefig.facecolor": "#F8FBFC",
                      "axes.edgecolor": "#243B53", "grid.color": "#C5D2D9", "text.color": "#20353C",
                      "axes.labelcolor": "#20353C", "xtick.color": "#20353C", "ytick.color": "#20353C"})
    fig, ax = plt.subplots(figsize=size)
    ys = ys or []
    plot_df = df.copy()

    if kind in NEEDS_XY:
        x_cols = x if isinstance(x, (list, tuple)) else ([x] if x else [])
        if not x_cols or not ys:
            raise ValueError("Choose an X-axis column and at least one numeric Y-axis column.")
        missing = [c for c in [*x_cols, *ys] if c not in plot_df.columns]
        if missing:
            raise ValueError("A selected column is no longer available. Please choose the columns again.")
        for y in ys:
            converted = pd.to_numeric(plot_df[y], errors="coerce")
            if plot_df[y].notna().any() and converted.notna().sum() == 0:
                raise ValueError(f"Y-axis column '{y}' has no numeric values. Choose a numeric column.")
            plot_df[y] = converted
        if kind != "Scatter plot" and len(x_cols) != 1:
            raise ValueError(f"{kind} requires exactly one X-axis column. Choose one X column.")
        if kind == "Scatter plot":
            for xc in x_cols:
                if not pd.api.types.is_numeric_dtype(plot_df[xc]):
                    converted_x = pd.to_numeric(plot_df[xc], errors="coerce")
                    if converted_x.notna().sum() == 0:
                        raise ValueError(f"Scatter plot requires numeric X values. '{xc}' contains categories/text; use Line plot or Bar chart to preserve category labels.")
                    plot_df[xc] = converted_x
        x = x_cols if kind == "Scatter plot" else x_cols[0]
        subset_cols = [*x_cols, *ys]
        plot_df = plot_df.dropna(subset=subset_cols, how="all")
        if plot_df.empty:
            raise ValueError("The selected columns do not contain any plottable values.")

    if kind == "Line plot":
        
        x_values = plot_df[x].astype(str).tolist()
        positions = np.arange(len(plot_df))
        for y in ys:
            ax.plot(positions, plot_df[y].to_numpy(dtype=float), marker="o", linewidth=2,
                    markersize=4, label=y)
        ax.set_xticks(positions)
        ax.set_xticklabels(x_values, rotation=45 if len(x_values) > 10 else 0,
                           ha="right" if len(x_values) > 10 else "center")
        ax.set_xlabel(x)
        y_values = plot_df[ys].to_numpy(dtype=float)
        finite = y_values[np.isfinite(y_values)]
        if finite.size and np.allclose(finite, np.round(finite)):
            distinct_y = np.unique(finite)
            if 1 < len(distinct_y) <= 10:
                ax.set_yticks(distinct_y)
            else:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    elif kind == "Scatter plot":
        x_cols = x if isinstance(x, (list, tuple)) else [x]
        for xc in x_cols:
            for y in ys:
                d = plot_df[[xc, y]].dropna()
                if d.empty:
                    continue
                ax.scatter(d[xc], d[y], label=f"{y} vs {xc}", alpha=.8)
        ax.set_xlabel(", ".join(x_cols))
        ax.set_ylabel(", ".join(ys))
        if len(x_cols) * len(ys) > 1:
            ax.legend(frameon=True)
    elif kind == "Bar chart":
        if len(plot_df) > 60:
            raise ValueError("Bar chart supports up to 60 rows. Narrow the row range first.")
        positions = np.arange(len(plot_df)); width = .8 / max(len(ys), 1)
        for i, y in enumerate(ys):
            vals = plot_df[y].to_numpy(dtype=float)
            ax.bar(positions + (i - (len(ys)-1)/2)*width, vals, width=width, label=y)
        ax.set_xticks(positions)
        ax.set_xticklabels(plot_df[x].astype(str), rotation=45, ha="right")
        ax.set_xlabel(x); ax.set_ylabel(", ".join(ys))
    elif kind == "Area chart":
        for y in ys:
            ax.fill_between(np.arange(len(plot_df)), plot_df[y].to_numpy(dtype=float), alpha=.25, label=y)
            ax.plot(np.arange(len(plot_df)), plot_df[y].to_numpy(dtype=float), linewidth=1.8)
        ax.set_xticks(np.arange(len(plot_df)))
        ax.set_xticklabels(plot_df[x].astype(str), rotation=45, ha="right")
        ax.set_xlabel(x)
    elif kind == "Histogram":
        sns.histplot(data=df, x=col, kde=True, ax=ax, color="#176B78", edgecolor="#FFFFFF", alpha=.78,
                     line_kws={"color": "#243B53", "linewidth": 2.5})
    elif kind == "Box plot":
        sns.boxplot(data=df, y=col, ax=ax, color="#2A9D8F", saturation=.85)
    elif kind == "Violin plot":
        sns.violinplot(data=df, y=col, ax=ax, color="#2A9D8F", saturation=.85)
    elif kind == "Density plot":
        sns.kdeplot(data=df, x=col, fill=True, ax=ax, color="#176B78", alpha=.35, linewidth=2.5)
    elif kind == "Pie chart":
        counts = df[col].astype("string").fillna("Missing").value_counts().head(10)
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
    elif kind == "Count plot":
        order = df[col].astype("string").fillna("Missing").value_counts().head(30).index
        sns.countplot(data=df.assign(**{col: df[col].astype("string").fillna("Missing")}), x=col, order=order, ax=ax)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    elif kind == "Heatmap (correlation)":
        if not heat_cols or len(heat_cols) < 2:
            raise ValueError("Choose at least two numeric columns for the correlation heatmap.")
        corr = df[heat_cols].apply(pd.to_numeric, errors="coerce").corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=CMAP, vmin=-1, vmax=1, ax=ax)
    else:
        raise ValueError("Unknown chart type.")

    if kind not in {"Pie chart", "Heatmap (correlation)"}:
        apply_limits(ax, xlim, ylim)
    if kind == "Scatter plot":
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(handles, labels, frameon=True, title="Series", loc="best")
    elif kind in {"Line plot", "Bar chart", "Area chart"} and len(ys) > 1:
        ax.legend(frameon=True, title="Series", loc="best")
    elif ax.get_legend() is not None:
        ax.get_legend().remove()
    if title.strip():
        ax.set_title(title.strip(), fontsize=14, weight="bold", color=INK)
    fig.tight_layout()
    return fig

def fig_bytes(fig, fmt="png") -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=200, bbox_inches="tight")
    return buf.getvalue()


MIME = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}


def density_figure(values, x, normal, kde, column, method, mu, sigma, title=""):
    sns.set_theme(style="whitegrid", rc={
        "figure.facecolor": "#F8FBFC", "axes.facecolor": "#F8FBFC",
        "savefig.facecolor": "#F8FBFC", "axes.edgecolor": "#243B53",
        "grid.color": "#C5D2D9", "text.color": "#20353C",
        "axes.labelcolor": "#20353C", "xtick.color": "#20353C", "ytick.color": "#20353C"
    })
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if method in ("Kernel density estimate", "Both") and np.isfinite(kde).any():
        ax.plot(x, kde, lw=2.8, label="Kernel density estimate", color="#168A8A", zorder=4)
        ax.fill_between(x, kde, alpha=.20, color="#168A8A")
    if method in ("Normal distribution PDF", "Both") and np.isfinite(normal).any():
        ax.plot(x, normal, lw=3.0, ls="--", color="#243B53", label=f"Normal (μ={mu:.3g}, σ={sigma:.3g})", zorder=5)
    ax.set_xlabel(column, fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_title(title.strip() or f"Probability density: {column}", fontsize=14, weight="bold", color="#20353C", pad=12)
    ax.grid(True, alpha=.35, linewidth=.8)
    ax.tick_params(colors="#20353C")
    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#C5D2D9")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
