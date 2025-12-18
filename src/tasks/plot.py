import matplotlib.pyplot as plt
import seaborn as sns


def clear_axes(
    ax=None, top=True, right=True, left=False, bottom=False, minorticks_off=True
):
    """
    A more forcing version of sns.despine.

    Parameters
    ----------
    ax : matplotlib axes, optional
        Specific axes object to despine. Ignored if fig is provided.
    top, right, left, bottom : boolean, optional
        If True, remove that spine.
    minorticks_off: boolean, optional
        If True, remove all minor ticks
    """
    if ax is None:
        axes = plt.gcf().axes
    else:
        axes = [ax]

    for ax_i in axes:
        sns.despine(ax=ax_i, top=top, right=right, left=left, bottom=bottom)
        if minorticks_off:
            ax_i.minorticks_off()
        ax_i.tick_params(axis="x", which="both", top=not top)
        ax_i.tick_params(axis="y", which="both", right=not right)
        ax_i.tick_params(axis="y", which="both", left=not left)
        ax_i.tick_params(axis="x", which="both", bottom=not bottom)
