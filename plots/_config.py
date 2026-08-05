"""Plotting constants"""

from typing import Final

from src._config import DEFAULT_BEAST_DIR

BEAST_DIR: Final = (
    f"{DEFAULT_BEAST_DIR}/ba9f2d2a-27f3-4100-a1c0-43f8fe1c39fc/0.05_brsupport_dev_test"
)
DEFAULT_IMG_DIR: Final = "img_v2"
DEFAULT_STYLE: Final = ".matplotlib/paper.mplstyle"

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
