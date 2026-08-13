"""Plotting constants"""

from typing import Final

from src._config import (
    DEFAULT_BEAST_DIR,
    DEFAULT_EMBEDDING_DIR,
)

DATASET: Final = "fleurs-r"
DEFAULT_IMG_DIR: Final = "img_v2"
DEFAULT_STYLE: Final = ".matplotlib/paper.mplstyle"
BURNIN_FRAC: Final = 0.10
SPEECH_BEAST_DIR: Final = (
    f"{DEFAULT_BEAST_DIR}/ba9f2d2a-27f3-4100-a1c0-43f8fe1c39fc/0.05_brsupport_dev_test"
)
COGNATE_BEAST_DIR: Final = f"{DEFAULT_BEAST_DIR}/iecor"
XLS_R_EMBEDDING_DIR: Final = (
    f"{DEFAULT_EMBEDDING_DIR}/fleurs-r/67c9af47-6177-4d06-bcc5-7c64b43e4b06"
)

# NMF component display order (0-indexed column indices from Q matrix).
# Display order: C9, C5, C7, C1, C6, C4, C2, C8, C3
NMF_COMP_ORDER: Final = [8, 4, 6, 0, 5, 3, 1, 7, 2]

TAXONSET_ORDER: Final = [
    "germanic",
    "celtic",
    "slavic",
    "romance",
    "indoaryan",
    "iranian",
    "baltic",
    "greek",
    "armenian",
]

TAXONSET_DISPLAY: Final = {
    "germanic": "Germanic",
    "celtic": "Celtic",
    "slavic": "Slavic",
    "romance": "Romance",
    "indoaryan": "Indo-Aryan",
    "iranian": "Iranian",
    "baltic": "Baltic",
    "greek": "Greek",
    "armenian": "Armenian",
}

TAXONSET_PALETTE: Final = {
    "germanic": "Reds_r",
    "celtic": ["orange", "darkorange"],
    "indoaryan": "Purples_r",
    "slavic": "Greens_r",
    "baltic": "blend:lightgreen,darkseagreen",
    "romance": "blend:darkkhaki,olive",
    "iranian": "blend:midnightblue,steelblue",
    "greek": ["gold"],
    "armenian": ["cyan"],
}

NMF_COMP_LABELS: Final = [
    "E. South Asian",
    "W. South Asian",
    "Iran. Plateau",
    "E. Slavic",
    "C. European",
    "W. Balkan",
    "NW. European",
    "Gallo-Iberian",
    "Italo-Lusitanic",
]

# NMF component palette
PALETTE: Final = [
    "#6a4b8c",  # Indo-Aryan (Paired: dark purple)
    "#c8b7d1",  # Indo-Aryan (Paired: light purple)
    "#35749e",  # Iranian (Paired: dark blue)
    "#206327",  # Slavic (Dark: green)
    "#428f3d",  # Slavic (Paired: green)
    "#b3d297",  # Slavic (Paired: light green)
    "#c5383a",  # Germanic (Paired: red)
    "#d97f26",  # Italic (Paired: dark orange)
    "#e8bc84",  # Italic (Paired: light orange)
]

FIG2_SIZE = (6.7, 2.5)
FIG2_COLOR_SPEECH = "#414487"
FIG2_COLOR_COGNATE = "#7ad151"
