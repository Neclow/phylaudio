"""Figure 2a / Supp Fig S5a: posterior root age distributions."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ._config import (
    BURNIN_FRAC,
    COGNATE_BEAST_DIR,
    DEFAULT_IMG_DIR,
    DEFAULT_STYLE,
    FIG2_COLOR_COGNATE,
    FIG2_COLOR_SPEECH,
    FIG2_SIZE,
    SPEECH_BEAST_DIR,
)

IMG_DIR = f"{DEFAULT_IMG_DIR}/fig2"
SPEECH_LOG_FILE = f"{SPEECH_BEAST_DIR}/input_v1_101.log"
SPEECH_PRIOR_LOG = f"{SPEECH_BEAST_DIR}/prior_v1_101.log"
COGNATE_LOG_FILE = f"{COGNATE_BEAST_DIR}/raw.log"
COGNATE_PRIOR_LOG = f"{COGNATE_BEAST_DIR}/prior.log"


def _load_root_age(log_file):
    df = pd.read_csv(log_file, sep="\t", comment="#")
    n_burnin = int(len(df) * BURNIN_FRAC)
    return df["TreeHeight.t:tree"].iloc[n_burnin:]


def _plot_root_age(ax, posterior_log_file, color, label, prior_log_file=None):
    ages = _load_root_age(posterior_log_file)
    sns.kdeplot(
        ages, ax=ax, fill=True, color=color, alpha=0.35, label=f"{label} (posterior)"
    )
    median = ages.median()
    hpd_lo, hpd_hi = ages.quantile(0.025), ages.quantile(0.975)
    ax.axvspan(
        hpd_lo,
        hpd_hi,
        color=color,
        alpha=0.08,
        zorder=0,
        label=f"95% HPD [{hpd_lo:.2f}, {hpd_hi:.2f}]",
    )
    ax.axvline(
        median, color="darkred", ls="dashed", lw=1, label=f"Median ({median:.2f})"
    )
    if prior_log_file is not None:
        prior_ages = _load_root_age(prior_log_file)
        sns.kdeplot(
            prior_ages,
            ax=ax,
            fill=False,
            color=color,
            alpha=0.5,
            linestyle=":",
            lw=1,
            label=f"{label} (prior)",
        )


def plot_root_age(output_name="fig2a_root_age"):
    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(4, 2.5))

        _plot_root_age(ax, SPEECH_LOG_FILE, FIG2_COLOR_SPEECH, "Root age")

        ax.set_xlabel("Age (ka BP)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper left", frameon=False)
        ax.invert_xaxis()
        sns.despine(ax=ax)

        for ext in ("pdf", "svg"):
            fig.savefig(f"{IMG_DIR}/{output_name}.{ext}", bbox_inches="tight")
        print(f"Saved figure to {IMG_DIR}/{output_name}.{{pdf,svg}}")
        plt.show()


def plot_root_age_overlaid(output_name="figS5a_root_age"):
    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=FIG2_SIZE)

        _plot_root_age(
            ax,
            SPEECH_LOG_FILE,
            FIG2_COLOR_SPEECH,
            "Speech",
            SPEECH_PRIOR_LOG,
        )
        _plot_root_age(
            ax,
            COGNATE_LOG_FILE,
            FIG2_COLOR_COGNATE,
            "Cognates",
            COGNATE_PRIOR_LOG,
        )

        ax.set_ylabel("Density", fontsize=8)
        ax.set_xlabel("Root age (ka BP)", fontsize=8)
        ax.legend(fontsize=7, loc="upper left", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.invert_xaxis()

        plt.tight_layout()
        for ext in ("pdf", "svg"):
            fig.savefig(f"{IMG_DIR}/{output_name}.{ext}", dpi=300, bbox_inches="tight")
        print(f"Saved figure to {IMG_DIR}/{output_name}.{{pdf,svg}}")
        plt.show()


if __name__ == "__main__":
    os.makedirs(IMG_DIR, exist_ok=True)
    plot_root_age()
    plot_root_age_overlaid()
