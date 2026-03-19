"""Shared constants for phylo plotting and analysis scripts."""

# ── Language name mappings ────────────────────────────────────────────────────

# Maps cognate tree language names to speech tree language names
COGNATE_TO_SPEECH = {
    "ArmenianEastern": "Armenian",
    "GaelicIrish": "Irish",
    "KurdishCJafi": "Sorani-Kurdish",
    "NorwegianBokmal": "Norwegian",
    "PersianTehran": "Persian",
    "SerboCroatian": "Serbian",
    "Slovene": "Slovenian",
    "WelshNorth": "Welsh",
}

# Maps a geojson "name" to ALL metadata language names it should match.
# Single-entry lists handle simple renames; multi-entry lists expand one polygon
# into multiple rows so it can match both Speech and Cognate metadata files.
GEOJSON_EXPANSION = {
    "Belarusian (Belorussian)": ["Belarusian"],
    "Punjabi (Panjabi)":        ["Punjabi"],
    "Netherlandic":             ["Dutch"],
    "Slovene":                  ["Slovene", "Slovenian"],
    "Norwegian":                ["Norwegian", "NorwegianBokmal"],
    "Persian (Farsi)":          ["Persian", "PersianTehran"],
    "Armenian":                 ["Armenian", "ArmenianEastern"],
    "Kurdish":                  ["KurdishCJafi", "Sorani-Kurdish"],
    "Welsh":                    ["Welsh", "WelshNorth"],
    "Irish":                    ["Irish", "GaelicIrish"],
    "Serbian / Croatian / Bosnian": ["Serbian", "Croatian", "Bosnian", "SerboCroatian"],
}

# Languages to exclude from regression/plotting (non-IE or insufficient data)
EXCLUDE_LANGUAGES = {
    "Turkish", "Finnish", "Hungarian",
    "Breton", "Cornish",
    "Scottish Gaelic", "Gaelic", "Manx",
}
