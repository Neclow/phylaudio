"""Plotting constants"""

from typing import Final

from src._config import (
    COGNATE_BEAST_DIR,
    EXCLUDE_LANGUAGES,
    GEOJSON_EXPANSION,
    GEOJSON_PATH,
    NE_COUNTRIES_PATH,
    NE_LAND_PATH,
    SPEECH_BEAST_DIR,
)

DEFAULT_IMG_DIR: Final = "img_v2"
DEFAULT_STYLE: Final = ".matplotlib/paper.mplstyle"
BURNIN_FRAC: Final = 0.10

# NMF component display order (0-indexed column indices from Q matrix).
# Display order: C9, C5, C7, C1, C6, C4, C2, C8, C3
NMF_COMP_ORDER: Final = [8, 4, 6, 0, 5, 3, 1, 7, 2]

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
