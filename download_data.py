"""
Download official BAAC road-accident CSV files from data.gouv.fr (2021-2024).

Each year has 4 tables: caracteristiques, usagers, lieux, vehicules.
Files are saved to <year>/<filename>.csv beside this script.

Usage:
    python download_data.py            # downloads all years
    python download_data.py --year 2024
"""

import argparse
import sys
from pathlib import Path

import requests

DATASET_PAGE = (
    "https://www.data.gouv.fr/fr/datasets/"
    "bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-"
    "routiere-annees-de-2005-a-2024/"
)

# Direct static URLs sourced from the data.gouv.fr API (resource list).
# These are stable CDN links; check DATASET_PAGE for updates if a download fails.
URLS = {
    2021: {
        "carcteristiques-2021.csv": (
            "https://static.data.gouv.fr/resources/"
            "bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-"
            "routiere-annees-de-2005-a-2020/20221024-113743/carcteristiques-2021.csv"
        ),
        "usagers-2021.csv": (
            "https://static.data.gouv.fr/resources/"
            "bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-"
            "routiere-annees-de-2005-a-2022/20231009-140337/usagers-2021.csv"
        ),
        "lieux-2021.csv": (
            "https://static.data.gouv.fr/resources/"
            "bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-"
            "routiere-annees-de-2005-a-2020/20221024-113743/lieux-2021.csv"
        ),
        "vehicules-2021.csv": (
            "https://static.data.gouv.fr/resources/"
            "bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-"
            "routiere-annees-de-2005-a-2020/20221024-113743/vehicules-2021.csv"
        ),
    },
    2022: {
        "carcteristiques-2022.csv": (
            "https://static.data.gouv.fr/resources/"
            "bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-"
            "routiere-annees-de-2005-a-2021/20231005-093927/carcteristiques-2022.csv"
        ),
        "usagers-2022.csv": (
            "https://static.data.gouv.fr/resources/"
            "bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-"
            "routiere-annees-de-2005-a-2021/20231005-094229/usagers-2022.csv"
        ),
        "lieux-2022.csv": (
            "https://static.data.gouv.fr/resources/"
            "bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-"
            "routiere-annees-de-2005-a-2021/20231005-094124/lieux-2022.csv"
        ),
        "vehicules-2022.csv": (
            "https://static.data.gouv.fr/resources/"
            "bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-"
            "routiere-annees-de-2005-a-2021/20231005-094053/vehicules-2022.csv"
        ),
    },
    2023: {
        "caract-2023.csv": (
            "https://static.data.gouv.fr/resources/"
            "bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-"
            "routiere-annees-de-2005-a-2023/20241028-103125/caract-2023.csv"
        ),
        "usagers-2023.csv": (
            "https://static.data.gouv.fr/resources/"
            "bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-"
            "routiere-annees-de-2005-a-2023/20241023-153328/usagers-2023.csv"
        ),
        "lieux-2023.csv": (
            "https://static.data.gouv.fr/resources/"
            "bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-"
            "routiere-annees-de-2005-a-2023/20241023-153328/lieux-2023.csv"
        ),
        "vehicules-2023.csv": (
            "https://static.data.gouv.fr/resources/"
            "bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-"
            "routiere-annees-de-2005-a-2023/20241023-153328/vehicules-2023.csv"
        ),
    },
    2024: {
        "caract-2024.csv": (
            "https://static.data.gouv.fr/resources/"
            "bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-"
            "routiere-annees-de-2005-a-2024/20251021-115900/caract-2024.csv"
        ),
        "usagers-2024.csv": (
            "https://static.data.gouv.fr/resources/"
            "bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-"
            "routiere-annees-de-2005-a-2024/20251021-115506/usagers-2024.csv"
        ),
        "lieux-2024.csv": (
            "https://static.data.gouv.fr/resources/"
            "bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-"
            "routiere-annees-de-2005-a-2024/20251021-115506/lieux-2024.csv"
        ),
        "vehicules-2024.csv": (
            "https://static.data.gouv.fr/resources/"
            "bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-"
            "routiere-annees-de-2005-a-2024/20251021-115506/vehicules-2024.csv"
        ),
    },
}


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  skip  {dest.name} (already exists)")
        return
    print(f"  fetch {dest.name} … ", end="", flush=True)
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size):
            f.write(chunk)
    size_mb = dest.stat().st_size / 1e6
    print(f"{size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download BAAC accident data 2021-2024")
    parser.add_argument("--year", type=int, choices=list(URLS), help="Download a single year")
    args = parser.parse_args()

    base = Path(__file__).parent
    years = [args.year] if args.year else sorted(URLS)

    print(f"Dataset page: {DATASET_PAGE}\n")
    failed = []
    for year in years:
        print(f"── {year} ──")
        for filename, url in URLS[year].items():
            dest = base / str(year) / filename
            try:
                download_file(url, dest)
            except Exception as e:
                print(f"FAILED: {e}")
                failed.append((year, filename, str(e)))

    if failed:
        print("\nFailed downloads:")
        for year, fname, err in failed:
            print(f"  {year}/{fname}: {err}")
        print(f"\nIf URLs are stale, check: {DATASET_PAGE}")
        sys.exit(1)
    else:
        print("\nAll files downloaded.")
        print("Delete data/sample/ after downloading full data if you want to save disk space.")


if __name__ == "__main__":
    main()
