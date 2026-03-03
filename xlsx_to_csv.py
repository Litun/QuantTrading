"""
xlsx_to_csv.py
--------------
Converts DOL LCA Disclosure Data xlsx files to clean CSVs.

Cleaning applied (no value transformations):
  - Keep only columns relevant to investment/wage analysis
  - Original column names preserved; known rename variants normalised to
    canonical names (e.g. H-1B_DEPENDENT / H1B_DEPENDENT → H_1B_DEPENDENT)
  - String columns: strip leading/trailing whitespace
  - Date columns: normalise to ISO-8601 (YYYY-MM-DD)
  - Drop rows where CASE_NUMBER is null

Dropped column categories:
  - PII / contact: EMPLOYER_POC_*, AGENT_*, LAWFIRM_*, PREPARER_*
  - Boilerplate: AGREE_TO_LC_STATEMENT, PUBLIC_DISCLOSURE, STATUTORY_BASIS
  - Street addresses (city / state / zip retained)
  - PW survey provenance: PW_OTHER_*, PW_SURVEY_*

Usage:
  python xlsx_to_csv.py --data-dir dol_2026Q1
  python xlsx_to_csv.py --data-dir dol_2020-2025 dol_2026Q1 --out-dir csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Column configuration
# ---------------------------------------------------------------------------

KEEP_COLS = [
    "CASE_NUMBER",
    "CASE_STATUS",
    "RECEIVED_DATE",
    "DECISION_DATE",
    "ORIGINAL_CERT_DATE",
    "VISA_CLASS",
    "JOB_TITLE",
    "SOC_CODE",
    "SOC_TITLE",
    "FULL_TIME_POSITION",
    "BEGIN_DATE",
    "END_DATE",
    "TOTAL_WORKER_POSITIONS",
    "NEW_EMPLOYMENT",
    "CONTINUED_EMPLOYMENT",
    "CHANGE_PREVIOUS_EMPLOYMENT",
    "NEW_CONCURRENT_EMPLOYMENT",
    "CHANGE_EMPLOYER",
    "AMENDED_PETITION",
    "EMPLOYER_NAME",
    "EMPLOYER_CITY",
    "EMPLOYER_STATE",
    "EMPLOYER_POSTAL_CODE",
    "NAICS_CODE",
    "WORKSITE_WORKERS",
    "SECONDARY_ENTITY",
    "SECONDARY_ENTITY_BUSINESS_NAME",
    "WORKSITE_CITY",
    "WORKSITE_STATE",
    "WORKSITE_POSTAL_CODE",
    "TOTAL_WORKSITE_LOCATIONS",
    "WAGE_RATE_OF_PAY_FROM",
    "WAGE_RATE_OF_PAY_TO",
    "WAGE_UNIT_OF_PAY",
    "PREVAILING_WAGE",
    "PW_UNIT_OF_PAY",
    "PW_TRACKING_NUMBER",
    "PW_WAGE_LEVEL",
    "PW_OES_YEAR",
    "H_1B_DEPENDENT",
    "WILLFUL_VIOLATOR",
    "APPENDIX_A_ATTACHED",
]

DATE_COLS = ["RECEIVED_DATE", "DECISION_DATE", "ORIGINAL_CERT_DATE", "BEGIN_DATE", "END_DATE"]

# Column names that changed across DOL fiscal-year releases → normalise to
# the canonical name used in KEEP_COLS before any further processing.
COLUMN_ALIASES = {
    "H-1B_DEPENDENT":       "H_1B_DEPENDENT",   # FY2020, FY2025
    "H1B_DEPENDENT":        "H_1B_DEPENDENT",   # FY2021
    "EMPLOYER_POC_ADDRESS_1": "EMPLOYER_POC_ADDRESS1",  # FY2021 (dropped anyway)
    "EMPLOYER_POC_ADDRESS_2": "EMPLOYER_POC_ADDRESS2",  # FY2021 (dropped anyway)
}

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def convert(src: Path, out: Path) -> int:
    """Convert one xlsx to CSV. Returns row count written."""
    available = pd.read_excel(src, nrows=0, engine="openpyxl").columns.tolist()

    # Normalise known aliases so KEEP_COLS lookup works regardless of year
    rename_map = {old: new for old, new in COLUMN_ALIASES.items() if old in available}
    normalised = [rename_map.get(c, c) for c in available]

    # Columns to read: original names that map to something in KEEP_COLS
    use_original = [
        orig for orig, norm in zip(available, normalised) if norm in KEEP_COLS
    ]

    df = pd.read_excel(src, usecols=use_original, engine="openpyxl",
                       dtype_backend="numpy_nullable")

    # Apply renames to canonical names
    df = df.rename(columns=rename_map)

    # Drop rows with no case identifier
    df = df.dropna(subset=["CASE_NUMBER"])

    # Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # Normalise dates to ISO-8601; missing → empty string
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
            df[col] = df[col].where(df[col] != "NaT", other="")

    # Ensure consistent column order (KEEP_COLS order, only present cols)
    ordered = [c for c in KEEP_COLS if c in df.columns]
    df = df[ordered]

    df.to_csv(out, index=False, encoding="utf-8")
    return len(df)


def verify(path: Path) -> bool:
    df = pd.read_csv(path, nrows=500, low_memory=False)
    errors = []
    if df["CASE_NUMBER"].isna().any():
        errors.append("null CASE_NUMBER values present")
    if errors:
        print(f"  [FAIL] {path.name}: {'; '.join(errors)}")
        return False
    print(f"  [OK]   {path.name}  ({df.shape[1]} cols)")
    return True

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", nargs="+", default=["dol_2026Q1"],
                        help="One or more directories containing LCA_Disclosure_Data_*.xlsx")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for CSVs (default: same as first --data-dir)")
    args = parser.parse_args()

    data_dirs = [Path(d) for d in args.data_dir]
    out_dir   = Path(args.out_dir) if args.out_dir else data_dirs[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all xlsx files across all specified directories
    sources = []
    for d in data_dirs:
        sources.extend(sorted(d.glob("LCA_Disclosure_Data_*.xlsx")))

    if not sources:
        print(f"[error] no LCA_Disclosure_Data_*.xlsx found in: {[str(d) for d in data_dirs]}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(sources)} file(s) — output → {out_dir.resolve()}\n")

    outputs = []
    for src in sources:
        quarter = src.stem.split("_FY")[-1].lower()   # "2026_q1"
        out = out_dir / f"lca_disclosure_fy{quarter}.csv"
        n = convert(src, out)
        print(f"  {src.name:<45}  {n:>7,} rows  →  {out.name}")
        outputs.append(out)

    print("\nVerifying ...")
    ok = all(verify(o) for o in outputs)
    print(f"\n{'All OK' if ok else 'ERRORS — check output above'}. {len(outputs)} CSV(s) written.")


if __name__ == "__main__":
    main()
