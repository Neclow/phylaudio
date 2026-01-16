import json
import os
import re
from getpass import getpass
from io import BytesIO, StringIO

import pandas as pd
import requests
from pypdf import PdfReader

from .._config import _FLEURS_SHORT_TO_LANG, DEFAULT_METADATA_DIR

DEFAULT_TIMEOUT = 30
FLEURS_PDF_URL = "https://arxiv.org/pdf/2205.12446"
LINGUAMETA_URL = "https://raw.githubusercontent.com/google-research/url-nlp/452a21ad3dae5668c06ceeac21ff073e1e40f9be/linguameta/linguameta.tsv"
WIKIMEDIA_AUTH_URL = "https://auth.enterprise.wikimedia.com/v1/login"
WIKIMEDIA_API_URL = "https://api.enterprise.wikimedia.com/v2/structured-contents"


class GetOutOfLoop(Exception):
    """A simple exception to get out of nested loops."""

    pass


def download_linguameta():
    """Download speaker population data from the LinguaMeta project

    Source: https://raw.githubusercontent.com/google-research/url-nlp/452a21ad3dae5668c06ceeac21ff073e1e40f9be/linguameta/linguameta.tsv

    Returns
    -------
    pd.DataFrame
        DataFrame with LinguaMeta data
        Columns:
            - bcp_47_code: BCP 47 language code
            - iso_639_3_code: ISO 639-3 language code
            - english_name: English name of the language
            - endonym: Endonym of the language
            - estimated_number_of_speakers: Estimated number of speakers
            - writing_systems: Writing systems used by the language
            - locales: Locales where the language is spoken
            - cldr_official_status: CLDR official status
            - endangerment_status: Endangerment status of the language
            - is_macrolanguage: Whether the language is a macrolanguage
            - macrolanguage_bcp_47_code: BCP 47 code of the macrolanguage
            - individual_language_bcp_47_code: BCP 47 code of the individual language
            - iso_639_2b_code: ISO 639-2/B code
            - deprecated_bcp_47_code: Deprecated BCP 47 code
            - glottocode: Glottocode of the language
            - wikidata_id: Wikidata ID of the language
            - wikidata_description: Description from Wikidata
    """
    linguameta_df = pd.read_csv(LINGUAMETA_URL, sep="\t")
    return linguameta_df


def download_fleurs():
    """Download speaker population data from the FLEURS paper

    Returns
    -------
    pd.DataFrame
        DataFrame with FLEURS data
        Columns:
            - Idx: Index
            - Language: Language name
            - ISO_639-3: ISO 639-3 language code
            - ISO_639-1: ISO 639-1 language code
            - Family: Language family
            - Group: Language group
            - #S: Number of speakers (in millions)
    """
    fleurs_response = requests.get(FLEURS_PDF_URL, timeout=DEFAULT_TIMEOUT)
    fleurs_response.raise_for_status()
    fleurs_reader = PdfReader(BytesIO(fleurs_response.content))
    pages = [8, 9]
    start_pages = [5, 0]
    data = []
    has_header = False
    for page, page_start in zip(pages, start_pages):
        page_text = fleurs_reader.pages[page].extract_text()
        for i, row in enumerate(page_text.split("\n")[page_start:]):
            if i == 0:
                if not has_header:
                    data.append(row.replace("ISO ", "ISO_").replace(" ", "\t"))
                    has_header = True
                continue
            columns = replace_ligatures(row).split(" ")
            last5 = columns[-5:]
            notlast5 = columns[:-5]
            new_row = "\t".join([notlast5[0], " ".join(notlast5[1:]), *last5])
            data.append(new_row)

    fleurs_df = pd.read_csv(StringIO("\n".join(data)), sep="\t").set_index("Idx")
    return fleurs_df


def download_wikimedia(languages, dataset, overwrite=False):
    """Download Wikimedia structured content for given languages

    Parameters
    ----------
    languages : iterable
        Languages to download articles for
        e.g., from FLEURS dataset
    dataset : str
        Dataset name (e.g., "fleurs-r" or "fleurs")
    overwrite : bool, optional
        Whether to overwrite existing files, by default False

    Returns
    -------
    speaker_data : dict
        Parsed speaker data with integer number of speakers
    """

    content_path = f"{DEFAULT_METADATA_DIR}/{dataset}/wikimedia.json"
    speakers_path = f"{DEFAULT_METADATA_DIR}/{dataset}/wikimedia_speakers.json"
    if not overwrite:
        if os.path.exists(speakers_path):
            print(f"Wikimedia speakers already exists at {speakers_path}, loading...")
            with open(speakers_path, "r", encoding="utf-8") as f:
                speaker_data = json.load(f)
            return speaker_data
        if os.path.exists(content_path):
            print(f"Wikimedia content already exists at {content_path}, loading...")
            with open(content_path, "r", encoding="utf-8") as f:
                content = json.load(f)

            speaker_data = parse_wikimedia_speakers(extract_wikimedia_speakers(content))
            return speaker_data

    wikimedia_access_token = get_wikimedia_access_token()

    headers = {"Authorization": f"Bearer {wikimedia_access_token}"}
    content = {}
    # Round 1
    content = download_wikimedia_loop(
        headers,
        languages,
        lambda language: f"{language}_language",
        content,
    )

    # Round 2: without "_language"
    content = download_wikimedia_loop(
        headers,
        languages,
        lambda language: f"{language.split('(')[0].strip().replace(' ', '_')}",
        content,
    )

    # Round 3: manual imputation of missing languages
    missing_languages = {
        "Cantonese Chinese": "Yue_Chinese",
        "Filipino (Tagalog)": "Filipino_language",
        "Sorani Kurdish": "Central_Kurdish",
    }
    # Round 4: solve "may refer to" disambiguation pages manually
    ambiguous_languages = {
        "Luo": "Dhuluo",
        "Ganda": "Luganda",
        "Oriya": "Odia_language",
        "Portguese (Brazil)": "Portuguese_language",
    }
    # Round 5: get only English Wikipedia articles
    non_english_languages = {
        "Afrikaans": "Afrikaans",
        "Amharic": "Amharic",
        "Arabic": "Modern_Standard_Arabic",
        "Kabuverdianu": "Cape_Verdean_Creole",
        "Kannada": "Kannada",
        "Lingala": "Lingala",
        "Luxembourgish": "Luxembourgish",
        "Maori": "Māori_language",
        "Northern Sotho": "Northern_Sotho",
        "Slovenian": "Slovene_language",
        "Swahili": "Swahili",
    }
    error_languages = {
        **missing_languages,
        **ambiguous_languages,
        **non_english_languages,
    }
    for key_language, target_article in error_languages.items():
        article_response = requests.get(
            f"{WIKIMEDIA_API_URL}/{target_article}",
            timeout=DEFAULT_TIMEOUT,
            headers=headers,
        )
        article_response.raise_for_status()
        content[key_language] = article_response.json()
        print(f"✔️ Got article for {key_language} (manual URL imputation)")

    save_wikimedia_content(speakers_path, content)

    # Check that each language article as an infobox
    assert len([k for k in content.keys() if "infoboxes" not in content[k][0]]) == 0

    speaker_data = parse_wikimedia_speakers(extract_wikimedia_speakers(content))
    # Manual imputation for missing/erroneous values
    speaker_data["Asturian"]["parsed_value"] = 100_000
    speaker_data["Irish"]["parsed_value"] = 195_029
    speaker_data["Nyanja"]["parsed_value"] = float("nan")
    speaker_data["Welsh"]["parsed_value"] = 538_000

    save_wikimedia_content(
        f"{DEFAULT_METADATA_DIR}/{dataset}/wikimedia_speakers.json", speaker_data
    )

    return speaker_data


def get_wikimedia_access_token():
    """
    Get Wikimedia Enterprise access token for API calls

    Returns
    -------
    access_token : str
        Wikimedia Enterprise access token for API calls
    """
    login_data = {
        "username": input("Wikimedia Enterprise Username: "),
        "password": getpass("Wikimedia Enterprise Password: "),
    }

    response = requests.post(
        WIKIMEDIA_AUTH_URL, data=login_data, timeout=DEFAULT_TIMEOUT
    )
    response.raise_for_status()
    access_token = response.json().get("access_token")
    return access_token


def download_wikimedia_loop(headers, languages, name_fn, content):
    """For loop to download Wikimedia structured content

    Parameters
    ----------
    headers : dict
        HTTP headers for requests
    languages : iterable
        Languages to download articles for
    name_fn : callable
        Function to generate article name from language
    content : dict
        Dictionary to store downloaded content

    Returns
    -------
    dict
        Updated dictionary with downloaded content
    """
    print("Downloading Wikimedia articles...")
    for i, language in enumerate(languages):
        name = name_fn(language)
        print(f"({i+1}/{len(languages)}) {language}...", end=" ")
        try:
            article_response = requests.get(
                f"{WIKIMEDIA_API_URL}/{name}", timeout=30, headers=headers
            )
            article_response.raise_for_status()
            print("✔️")
            content[language] = article_response.json()
        except requests.HTTPError:
            print("❌ Failed to get article")
        except requests.ReadTimeout:
            print("❌ Timeout")

    return content


def save_wikimedia_content(output_path, content):
    """Save Wikimedia content to a JSON file

    Parameters
    ----------
    output_path : str
        Output file path
    content : dict
        Wikimedia content to save
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=4)


def extract_wikimedia_speakers(content):
    """Extract number of speakers from Wikimedia infoboxes

    Parameters
    ----------
    content : dict
        Wikimedia content
        key = language
        value = Wikimedia structured content JSON

    Returns
    -------
    speaker_data : dict
        Extracted speaker data
    """
    print("Extracting number of speakers from infoboxes...")
    speaker_data = {}
    for language, language_content in content.items():
        infobox = language_content[0]["infoboxes"][0]
        parts = infobox["has_parts"]
        print(f"{language}...", end=" ")
        try:
            for part in parts:
                if "has_parts" in part:
                    for subpart in part["has_parts"]:
                        if not "name" in subpart:
                            continue
                        if (
                            "speakers" in subpart["name"].lower()
                            or "revival" in subpart["name"].lower()  # Hebrew
                            or "users"
                            in subpart["name"].lower()  # Modern Standard Arabic
                        ):
                            speaker_data[language] = {
                                "value": subpart.get("value", None),
                                "values": subpart.get("values", None),
                            }
                            raise GetOutOfLoop
            raise ValueError("No speakers found")
        except GetOutOfLoop:
            print("✔️")
        except ValueError:
            speaker_data[language] = {
                "value": None,
                "values": None,
            }
            print("❌ No speakers info")
        except KeyError:
            print("❌ Error parsing infobox")

    return speaker_data


def parse_wikimedia_speakers(speaker_data):
    """Parse number of speakers (as numbers) from Wikimedia infoboxes

    Parameters
    ----------
    speaker_data : dict
        Extracted speaker data

    Returns
    -------
    speaker_data : dict
        Parsed speaker data with integer number of speakers
    """
    pattern = re.compile(
        r"(?<![\d\w/+\-])"  # Negative lookbehind: not preceded by a digit, letter, /, +, or -
        r"(\d{1,3}(?:,\d{3})*|\d+)"  # Number with optional commas
        r"(?:\.\d+)?"  # Optional decimal
        r"(?:\s*(million|billion))?"  # Optional million/billion
        r"(?![\w/+])",  # Negative lookahead: not followed by letter, /, or +
        re.IGNORECASE,
    )
    for language, speakers_info in speaker_data.items():
        print(f"Getting speakers for {language}...", end=" ")
        try:
            found = False
            if speakers_info["value"] is not None:
                match = pattern.search(
                    speakers_info["value"].split("L1: ")[-1].split("Native: ")[-1]
                )
                if match:
                    number_str = match.group(1).replace(",", "")
                    number = float(number_str)
                    if match.group(2) is not None:
                        multiplier_str = match.group(2)
                        if multiplier_str is not None:
                            if multiplier_str.lower() == "million":
                                number *= 1e6
                            elif multiplier_str.lower() == "billion":
                                number *= 1e9
                    if number < 10000:
                        raise ValueError("Unrealistically low number of speakers")
                    speaker_data[language]["parsed_value"] = int(number)
                    print(f"✔️ ({int(number):_})")
                    found = True
            if not found:
                raise ValueError
        except ValueError:
            print("❌ Error parsing number")

    return speaker_data


def replace_ligatures(text: str) -> str:
    """Replace ligatures in PDFs read by PyPDF

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    str
        Output text with ligatures replaced
    """
    ligatures = {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "ft",
        "ﬆ": "st",
        "ꜳ": "aa",
    }
    for search, replace in ligatures.items():
        text = text.replace(search, replace)
    return text


def download_speakerpop(dataset, overwrite=False):
    linguameta_data = download_linguameta()

    fleurs_data = download_fleurs()

    merged_df = fleurs_data.merge(
        linguameta_data,
        left_on="ISO_639-3",
        right_on="iso_639_3_code",
    )

    wikimedia_data = download_wikimedia(
        languages=merged_df.Language,
        dataset=dataset,
        overwrite=overwrite,
    )

    speakerpop_data = merged_df.rename(
        columns={
            "#S": "speakers_fleurs",
            "estimated_number_of_speakers": "speakers_linguameta",
        }
    ).loc[
        :,
        [
            "Language",
            "english_name",
            "ISO_639-1",
            "ISO_639-3",
            "glottocode",
            "speakers_fleurs",
            "speakers_linguameta",
        ],
    ]
    speakerpop_data["speakers_linguameta"] /= 1e6
    speakerpop_data["speakers_wikimedia"] = speakerpop_data["Language"].map(
        lambda x: (
            wikimedia_data[x]["parsed_value"] / 1e6
            if x in wikimedia_data
            else float("nan")
        )
    )

    # Append language dir names
    speakerpop_data["fleurs_dir"] = speakerpop_data["ISO_639-1"].map(
        lambda x: _FLEURS_SHORT_TO_LANG.get(x, None)
    )

    print("Done")

    return speakerpop_data
