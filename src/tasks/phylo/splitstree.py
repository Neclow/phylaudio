import pandas as pd


def extract_delta(filepath: str) -> pd.DataFrame:
    """Extract per-taxon delta scores from a SplitsTree6 .stree6 file.

    Parses the REPORT block produced by the Delta Score algorithm.
    """
    with open(filepath) as f:
        lines = f.readlines()

    capturing = False
    rows = []
    header = None

    for line in lines:
        text = line.strip().strip("'")
        if "Delta scores for individual taxa:" in text:
            capturing = True
            continue
        if capturing:
            if text in ("", ";"):
                break
            parts = text.split("\t")
            if header is None:
                header = parts
            else:
                rows.append(parts)

    df = pd.DataFrame(rows, columns=header)
    df = df.rename(columns={"delta score ": "delta.score", "Id": "id"})
    df["delta.score"] = df["delta.score"].astype(float)
    df["Q-residual"] = df["Q-residual"].astype(float)
    df = df.set_index("name")

    return df
