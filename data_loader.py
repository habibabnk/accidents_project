import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import streamlit as st
    cache_data = st.cache_data
except Exception:
    def cache_data(func=None, **_):
        return func if func else lambda f: f


class AccidentDataLoader:
    def __init__(self, data_dir=None):
        self.data_dir    = Path(data_dir) if data_dir else self._find_data_dir()
        self.loading_errors = []

    def _find_data_dir(self):
        for candidate in [
            Path(__file__).parent,
            Path(__file__).parent / "accidents_project",
            Path.cwd(),
            Path.cwd() / "accidents_project",
        ]:
            if any((candidate / str(y)).exists() for y in range(2015, 2025)):
                return candidate
        return Path(__file__).parent

    # ── CSV reader ────────────────────────────────────────────────────────────
    def smart_read_csv(self, fp):
        fp = Path(fp)
        for sep in [";", ","]:
            for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(fp, sep=sep, encoding=enc,
                                     low_memory=False, on_bad_lines="skip")
                    if df.shape[1] > 1:
                        return df
                except Exception:
                    continue
        raise RuntimeError(f"Cannot read {fp}")

    # Expected accident columns — at least one must be present to accept a file
    _ACCIDENT_COLS = {"Num_Acc", "num_acc", "Accident_Id", "accident_id", "jour", "hrmn"}

    def _is_accident_file(self, df):
        """Return True if df looks like a caracteristiques accident file."""
        return bool(self._ACCIDENT_COLS & set(df.columns))

    # ── Yearly caracteristiques ───────────────────────────────────────────────
    def load_yearly(self, data_dir=None):
        if data_dir:
            self.data_dir = Path(data_dir)
        self.loading_errors = []
        all_dfs = []

        for year in range(2015, 2025):
            yr_dir = self.data_dir / str(year)
            if not yr_dir.exists():
                continue

            # Specific accident-data patterns first; generic {year}.csv last
            patterns = [
                f"carcteristiques-{year}.csv",   # actual spelling used in 2021-2022
                f"caracteristiques-{year}.csv",  # standard spelling (future-proof)
                f"caract-{year}.csv",            # abbreviated (2023-2024)
                f"carct-{year}.csv",
                f"caracteristiques_{year}.csv",
                f"{year}.csv",                   # last resort (may be a different file)
            ]

            loaded = False
            for pat in patterns:
                fp = yr_dir / pat
                if fp.exists():
                    try:
                        df = self.smart_read_csv(fp)
                        if df.shape[1] > 1 and self._is_accident_file(df):
                            df["year"] = year
                            all_dfs.append(df)
                            loaded = True
                            break
                    except Exception as e:
                        self.loading_errors.append({"file": str(fp), "error": str(e)})

            if not loaded:
                for fp in sorted(yr_dir.glob("*.csv")):
                    try:
                        df = self.smart_read_csv(fp)
                        if df.shape[1] > 3 and self._is_accident_file(df):
                            df["year"] = year
                            all_dfs.append(df)
                            loaded = True
                            break
                    except Exception as e:
                        self.loading_errors.append({"file": str(fp), "error": str(e)})

        return pd.concat(all_dfs, ignore_index=True) if all_dfs else None

    # ── Usagers (gravity per accident) ───────────────────────────────────────
    def load_usagers(self, data_dir=None):
        if data_dir:
            self.data_dir = Path(data_dir)
        all_dfs = []
        for year in range(2015, 2025):          # <-- extended to all years
            yr_dir = self.data_dir / str(year)
            if not yr_dir.exists():
                continue
            for pat in [f"usagers-{year}.csv", f"usagers_{year}.csv", f"usagers{year}.csv"]:
                fp = yr_dir / pat
                if fp.exists():
                    try:
                        df = self.smart_read_csv(fp)
                        df["year"] = year
                        all_dfs.append(df)
                        break
                    except Exception as e:
                        self.loading_errors.append({"file": str(fp), "error": str(e)})
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else None

    # ── Preprocess ────────────────────────────────────────────────────────────
    def preprocess(self, df, usagers=None):
        if df is None:
            return None
        df = df.copy()

        # Standardise column names
        rename = {
            "Num_Acc": "accident_id", "num_acc": "accident_id",
            "Accident_Id": "accident_id",           # 2022 caracteristiques format
            "jour": "day", "mois": "month", "an": "year_src",
            "hrmn": "time", "lum": "lighting", "dep": "department",
            "com": "commune", "agg": "localization", "int": "intersection",
            "atm": "weather", "col": "collision_type",
            "lat": "latitude", "long": "longitude",
        }
        for old, new in rename.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]

        # ── Merge usagers gravity ─────────────────────────────────────────────
        if usagers is not None:
            # Normalise the join key in usagers
            uid = next((c for c in ["Num_Acc", "num_acc", "accident_id"]
                        if c in usagers.columns), None)
            if uid and "accident_id" in df.columns:
                u = usagers.rename(columns={uid: "accident_id"}).copy()
                if "grav" in u.columns:
                    u["grav"] = pd.to_numeric(u["grav"], errors="coerce")
                    # BAAC format since 2018: grav 1=indemne, 2=tué, 3=blessé hospitalisé, 4=blessé léger
                    # (older format was: 1=indemne, 2=blessé léger, 3=blessé hospitalisé, 4=tué)
                    # We detect which format is in use by checking whether grav=2 or grav=4 is rarer
                    # (fatalities are always fewer than minor injuries)
                    g_valid = u["grav"].dropna()
                    new_format = g_valid[g_valid > 0].pipe(
                        lambda s: s[s == 2].count() < s[s == 4].count()
                    )
                    if new_format:
                        fatal_val, serious_val = 2, 3
                    else:
                        fatal_val, serious_val = 4, 3
                    sev = u.groupby("accident_id").agg(
                        fatalities       =("grav", lambda x: int((x == fatal_val).sum())),
                        serious_injuries =("grav", lambda x: int((x == serious_val).sum())),
                        grav_max         =("grav", "max"),
                    ).reset_index()
                    df = df.merge(sev, on="accident_id", how="left")

        # ── Build is_serious target ───────────────────────────────────────────
        #  Priority order:
        #  1. merged fatalities / serious_injuries columns
        #  2. grav column in the caracteristiques file itself
        #  3. fallback 0 (model will warn user)

        if "fatalities" in df.columns and "serious_injuries" in df.columns:
            df["is_serious"] = (
                (df["fatalities"].fillna(0) > 0) |
                (df["serious_injuries"].fillna(0) > 0)
            ).astype(int)
            df["fatalities"]       = df["fatalities"].fillna(0).astype(int)
            df["serious_injuries"] = df["serious_injuries"].fillna(0).astype(int)

        elif "grav" in df.columns:
            # grav is in the caracteristiques file (older format or already joined)
            g = pd.to_numeric(df["grav"], errors="coerce")
            new_fmt = (g == 2).sum() < (g == 4).sum()
            if new_fmt:  # 2018+ format: 2=tué, 3=hospitalisé, 4=léger
                df["is_serious"]       = ((g == 2) | (g == 3)).astype(int)
                df["fatalities"]       = (g == 2).astype(int)
                df["serious_injuries"] = (g == 3).astype(int)
            else:        # old format:   3=hospitalisé, 4=tué
                df["is_serious"]       = (g >= 3).astype(int)
                df["fatalities"]       = (g == 4).astype(int)
                df["serious_injuries"] = (g == 3).astype(int)

        elif "grav_max" in df.columns:
            g = pd.to_numeric(df["grav_max"], errors="coerce")
            new_fmt = (g == 2).sum() < (g == 4).sum()
            if new_fmt:
                df["is_serious"]       = ((g == 2) | (g == 3)).astype(int)
                df["fatalities"]       = (g == 2).astype(int)
                df["serious_injuries"] = (g == 3).astype(int)
            else:
                df["is_serious"]       = (g >= 3).astype(int)
                df["fatalities"]       = (g == 4).astype(int)
                df["serious_injuries"] = (g == 3).astype(int)

        else:
            # No gravity column found at all
            df["is_serious"]       = 0
            df["fatalities"]       = 0
            df["serious_injuries"] = 0

        # ── Parse hour ────────────────────────────────────────────────────────
        if "time" in df.columns:
            t = df["time"].astype(str).str.strip().str.replace(":", "", regex=False)
            df["hour"] = pd.to_numeric(t.str[:2], errors="coerce").fillna(0).astype(int)
            df["hour"] = df["hour"].clip(0, 23)

        # ── Date / day-of-week ────────────────────────────────────────────────
        if all(c in df.columns for c in ["year", "month", "day"]):
            df["date"] = pd.to_datetime(
                dict(
                    year =pd.to_numeric(df["year"],  errors="coerce"),
                    month=pd.to_numeric(df["month"], errors="coerce"),
                    day  =pd.to_numeric(df["day"],   errors="coerce"),
                ),
                errors="coerce",
            )
            df["day_of_week"] = df["date"].dt.dayofweek

        # ── Clean department ──────────────────────────────────────────────────
        if "department" in df.columns:
            df["department"] = (
                df["department"].astype(str).str.strip()
                .str.replace(r"\.0$", "", regex=True)
                .str.zfill(2)
            )

        return df

    # ── Public entry point ────────────────────────────────────────────────────
    def get_data(self, data_dir=None):
        raw     = self.load_yearly(data_dir)
        usagers = self.load_usagers(data_dir)
        if raw is None:
            return None, self.loading_errors
        processed = self.preprocess(raw, usagers)
        return processed, self.loading_errors


@cache_data
def load_accident_data(data_dir: str):
    loader = AccidentDataLoader(data_dir)
    return loader.get_data()


def debug_data_directory(data_dir: str):
    try:
        p = Path(data_dir)
        if not p.exists():
            return 0, [f"Directory not found: {p}"]
        csvs   = sorted(p.rglob("*.csv"))
        preview = [
            f"{f.relative_to(p)}  ({f.stat().st_size / 1e6:.1f} MB)"
            for f in csvs[:30]
        ]
        return len(csvs), preview
    except Exception as e:
        return 0, [f"Scan error: {e}"]