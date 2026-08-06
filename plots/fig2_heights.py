"""Figure 2a: posterior distribution of root age (speech + cognate)."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ._config import BURNIN_FRAC, COGNATE_BEAST_DIR, DEFAULT_IMG_DIR, DEFAULT_STYLE, SPEECH_BEAST_DIR

IMG_DIR = f"{DEFAULT_IMG_DIR}/fig2"
SPEECH_LOG_FILE = f"{SPEECH_BEAST_DIR}/input_v1_101.log"
COGNATE_LOG_FILE = f"{COGNATE_BEAST_DIR}/raw.log"


def load_root_age(log_file):
    df = pd.read_csv(log_file, sep="\t", comment="#")
    n_burnin = int(len(df) * BURNIN_FRAC)
    df = df.iloc[n_burnin:]
    root_age = df["TreeHeight.t:tree"]
    print(
        f"  {log_file}: {len(root_age)} post-burnin samples, "
        f"median={root_age.median():.2f}, mean={root_age.mean():.2f}, "
        f"95% HPD=[{root_age.quantile(0.025):.2f}, {root_age.quantile(0.975):.2f}]"
    )
    return root_age


def plot_root_age(speech_age, cognate_age):
    with plt.style.context(DEFAULT_STYLE):
        fig, ax = plt.subplots(figsize=(4, 2.5))

        for ages, color, label in [
            (speech_age, "#414487", "Speech"),
            (cognate_age, "#7ad151", "Cognates"),
        ]:
            avg = ages.mean()
            q025 = ages.quantile(0.025)
            q975 = ages.quantile(0.975)

            sns.kdeplot(ages, ax=ax, fill=True, color=color, alpha=0.35, label=label)
            ax.axvline(avg, color=color, ls="--", lw=1)
            ax.fill_between(
                [q025, q975], 0, ax.get_ylim()[1], color=color, alpha=0.05,
            )

        ax.set_xlabel("Root age (ka)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.invert_xaxis()
        sns.despine(ax=ax)

        for ext in ("pdf", "svg"):
            output_path = f"{IMG_DIR}/fig2a_root_age.{ext}"
            fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved figure to {IMG_DIR}/fig2a_root_age.{{pdf,svg}}")
        plt.show()


if __name__ == "__main__":
    os.makedirs(IMG_DIR, exist_ok=True)
    speech_age = load_root_age(SPEECH_LOG_FILE)
    cognate_age = load_root_age(COGNATE_LOG_FILE)
    plot_root_age(speech_age, cognate_age)
