"""
Note: tree height is equivalent to root age here:
 - Speech trees are ultrametric
 - Cognate trees are not ultrametric, but we take the MRCA of modern languages as the tree height
    so it is equivalent to root age of the subtree containing only modern languages.
"""

# pylint: disable=redefined-outer-name
import os
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    MetavarTypeHelpFormatter,
)

import matplotlib.pyplot as plt
import pandas as pd
import rpy2
import rpy2.robjects as ro
import seaborn as sns
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

from src._config import DEFAULT_BEAST_DIR, DEFAULT_THREADS_NEXUS
from src.tasks.plot import clear_axes

rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda *args: None

XLSR_DIR = f"{DEFAULT_BEAST_DIR}/eab44e7f-54cc-4469-87d1-282cc81e02c2/0.25"
IECOR_DIR = f"{DEFAULT_BEAST_DIR}/iecor"

MODERN_LANGS = [
    "ArmenianEastern",
    "Assamese",
    "Belarusian",
    "Bengali",
    "Bulgarian",
    "Catalan",
    "Czech",
    "Danish",
    "Dutch",
    "English",
    "French",
    "German",
    "Greek",
    "Hindi",
    "Italian",
    "KurdishCJafi",
    "Latvian",
    "Lithuanian",
    "Macedonian",
    "Marathi",
    "Nepali",
    "NorwegianBokmal",
    "Pashto",
    "PersianTehran",
    "Polish",
    "Portuguese",
    "Punjabi",
    "Romanian",
    "Russian",
    "SerboCroatian",
    "Slovak",
    "Slovene",
    "Spanish",
    "Swedish",
    "Ukrainian",
    "Urdu",
]
PALETTE = ["#5e5eb5", "#bfbfe1", "#6ab06a", "#c3dfc3"]


class MixedHelpFormatter(ArgumentDefaultsHelpFormatter, MetavarTypeHelpFormatter):

    def _format_action_invocation(self, action):
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            (metavar,) = self._metavar_formatter(action, default)(1)
            return metavar
        else:
            parts = []
            # if the Optional doesn't take a value, format is: -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)
            # if the Optional takes a value, format is: -s ARGS, --long ARGS
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    parts.append("%s" % option_string)
                parts[-1] += " %s" % args_string
            return ", ".join(parts)


def parse_args():
    parser = ArgumentParser(formatter_class=MixedHelpFormatter)
    parser.add_argument(
        "-s1",
        "--speech_posterior",
        type=str,
        default=f"{XLSR_DIR}/long_v3_44.trees",
        help="Speech posterior trees",
    )
    parser.add_argument(
        "-s2",
        "--speech_prior",
        type=str,
        default=f"{XLSR_DIR}/prior/long_v3_888.trees",
        help="Speech prior trees",
    )
    parser.add_argument(
        "-c1",
        "--cognate_posterior",
        type=str,
        default=f"{IECOR_DIR}/raw.trees",
        help="Cognate posterior trees",
    )
    parser.add_argument(
        "-c2",
        "--cognate_prior",
        type=str,
        default=f"{IECOR_DIR}/prior/raw.trees",
        help="Cognate prior trees",
    )
    parser.add_argument(
        "-of",
        "--output_full",
        type=str,
        default="img/tree_height_prior_vs_posterior.pdf",
        help="Output file for full plot",
    )
    parser.add_argument(
        "-os",
        "--output_speech",
        type=str,
        default="img/tree_height_prior_vs_posterior_speech-only.pdf",
        help="Output file for speech-only plot",
    )
    parser.add_argument(
        "-nc",
        "--n_cores",
        type=int,
        default=DEFAULT_THREADS_NEXUS,
        help="Number of cores to use for parallel processing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing height data",
    )
    parser.add_argument(
        "--burnin",
        type=float,
        default=0.1,
        help="Proportion of samples to discard as burn-in",
    )
    parser.add_argument(
        "--xlabel",
        type=str,
        default="Root age (Years BP)",
        help="X-axis label for the plot",
    )
    return parser.parse_args()


def load_data(
    speech_post_file,
    speech_prio_file,
    cognate_post_file,
    cognate_prio_file,
    output_col="Root age (Years BP)",
    cores=4,
    overwrite=False,
    burnin=0.1,
):
    dfs = {}
    file_data = {
        "Speech (Posterior)": speech_post_file,
        "Speech (Prior)": speech_prio_file,
        "Cognates (Posterior)": cognate_post_file,
        "Cognates (Prior)": cognate_prio_file,
    }
    output_file = f"{DEFAULT_BEAST_DIR}/summary_height.csv"
    if not os.path.exists(output_file) or overwrite:
        for col, file in file_data.items():
            print(f"Processing {col}: {file}")
            with localconverter(ro.default_converter + numpy2ri.converter):
                ro.globalenv["file"] = file
                ro.globalenv["modern_langs"] = ro.StrVector(MODERN_LANGS)
                ro.globalenv["cores"] = cores
                sub_df = ro.r(
                    """
                    source("src/tasks/phylo/beast.R")
                    cat("Extracting BEAST tree heights...\\n")
                    sub_df <- extract_beast_heights(
                        file = file,
                        subset = modern_langs,
                        cores = cores
                    )
                    cat("Done.\\n")
                    sub_df
                    """
                ).flatten()

                # Ignore burn-in (10%)
                dfs[col] = sub_df[int(burnin * sub_df.shape[0]) :]
        df = pd.DataFrame.from_dict(dfs, orient="index").T
        df.to_csv(f"{DEFAULT_BEAST_DIR}/summary_height.csv", index=False)
    else:
        print(f"Loading existing height data from {output_file}")
        df = pd.read_csv(output_file, index_col=0)

    df_melt = df.melt(var_name="Model", value_name=output_col)

    return df_melt


def plot(
    data, output_full, output_speech, figsize=(5.5, 3), xlabel="Root age (Years BP)"
):
    with plt.style.context(".matplotlib/paper.mplstyle"):
        _, ax = plt.subplots(figsize=figsize)

        sns.kdeplot(
            data=data,
            x=xlabel,
            hue="Model",
            fill=True,
            common_norm=False,
            palette=PALETTE,
            alpha=0.5,
            ax=ax,
        )
        # Flip x-axis
        ax.set_xlim(0, 10)
        ax.invert_xaxis()
        clear_axes(ax=ax)
        plt.savefig(output_full)
        plt.show()

        # Speech only
        _, ax = plt.subplots(figsize=figsize)
        sns.kdeplot(
            data=data.query("Model.str.contains('Speech')"),
            x=xlabel,
            hue="Model",
            common_norm=False,
            fill=True,
            palette=PALETTE[:2],
            alpha=0.5,
            ax=ax,
        )
        # Flip- x-axis
        ax.set_xlim(0, 10)
        ax.invert_xaxis()
        clear_axes(ax=ax)
        plt.savefig(output_speech)
        # plt.savefig(output_speech.replace(".pdf", ".svg"))
        plt.show()


if __name__ == "__main__":
    args = parse_args()

    print("Loading data...")
    data = load_data(
        speech_post_file=args.speech_posterior,
        speech_prio_file=args.speech_prior,
        cognate_post_file=args.cognate_posterior,
        cognate_prio_file=args.cognate_prior,
        cores=args.n_cores,
        overwrite=args.overwrite,
        burnin=args.burnin,
        output_col=args.xlabel,
    )
    print("Plotting data...")
    plot(
        data=data,
        output_full=args.output_full,
        output_speech=args.output_speech,
        xlabel=args.xlabel,
    )
    print("Done.")
