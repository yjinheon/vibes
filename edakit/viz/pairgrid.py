import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype


def pairgrid_with_corr(df: pd.DataFrame, cols=None, height=2.0):
    cols = cols or [c for c in df.columns if is_numeric_dtype(df[c])]
    data = df[cols].dropna()
    try:
        import seaborn as sns
        from scipy import stats

        sns.set_style("white")

        def _corrdot(x, y, **kws):
            r = pd.Series(x).corr(pd.Series(y), method="pearson")
            ax = plt.gca()
            ax.set_axis_off()
            txt = f"{r:.2f}"
            fs = abs(r) * 80 + 5
            ax.annotate(
                txt,
                (0.5, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=fs,
            )

        def _corrstars(x, y, **kws):
            r, p = stats.pearsonr(x, y)
            stars = (
                "***" if p <= 0.001 else "**" if p <= 0.01 else "*" if p <= 0.05 else ""
            )
            ax = plt.gca()
            ax.set_axis_off()
            ax.annotate(
                stars, (0.65, 0.6), xycoords="axes fraction", color="red", fontsize=70
            )

        def _diag_hist_kde(x, **kws):
            ax = plt.gca()
            x = pd.Series(x).dropna()
            sns.histplot(x, bins="fd", ax=ax)
            try:
                sns.kdeplot(x, ax=ax)
            except Exception:
                pass

        def _lower_reg(x, y, **kws):
            sns.regplot(
                x=x,
                y=y,
                lowess=True,
                ci=False,
                scatter_kws={"s": 20},
                line_kws={"lw": 1},
                ax=plt.gca(),
            )

        g = sns.PairGrid(
            data, vars=cols, height=height, aspect=1.2, diag_sharey=False, despine=False
        )
        g.map_lower(_lower_reg)
        g.map_diag(_diag_hist_kde)
        g.map_upper(_corrdot)
        g.map_upper(_corrstars)

        for ax in g.axes.flatten():
            ax.set_xlabel("")
            ax.set_ylabel("")
        for ax, col in zip(np.diag(g.axes), cols):
            ax.set_title(col, y=0.82, fontsize=14)

        g.fig.subplots_adjust(wspace=0, hspace=0.02)
        return g.fig

    except Exception:
        from pandas.plotting import scatter_matrix

        fig = plt.figure()
        axs = scatter_matrix(
            data, figsize=(len(cols) * 2.5, len(cols) * 2.5), diagonal="hist"
        )
        n = len(cols)
        for i in range(n):
            for j in range(n):
                ax = axs[i, j]
                if i < j:
                    x = data[cols[j]].to_numpy()
                    y = data[cols[i]].to_numpy()
                    try:
                        from scipy import stats

                        r, p = stats.pearsonr(x, y)
                    except Exception:
                        r, p = (np.corrcoef(x, y)[0, 1], 1.0)
                    ax.clear()
                    ax.set_axis_off()
                    txt = f"{r:.2f}"
                    fs = abs(r) * 80 + 5
                    ax.annotate(
                        txt,
                        (0.5, 0.6),
                        xycoords="axes fraction",
                        ha="center",
                        va="center",
                        fontsize=fs,
                    )
                    stars = (
                        "***"
                        if p <= 0.001
                        else "**" if p <= 0.01 else "*" if p <= 0.05 else ""
                    )
                    if stars:
                        ax.annotate(
                            stars,
                            (0.7, 0.3),
                            xycoords="axes fraction",
                            color="red",
                            fontsize=28,
                        )
                elif i == j:
                    ax.set_title(cols[i], y=0.82, fontsize=12)
        plt.tight_layout()
        return fig
