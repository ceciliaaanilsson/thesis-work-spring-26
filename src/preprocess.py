#!/usr/bin/env python3
"""
Aggregera lektionsrader till elevnivå för EDM / klustring.

Endast rader med report_status == REPORTED används (UNREPORTED slängs).

Administrativa etiketter (absence_type) vs orsak (cause_ext):
- VALID: befogad frånvaro i systemet; många rader har cause_ext NOCAUSE men är ändå VALID.
- INVALID: ogiltig/obefogad frånvaro (administrativ ”skolk”).
- invalid_ratio (validerings-/etikettvariabel, ej i CLUSTERING_FEATURES): andel INVALID-minuter
  bland rader med is_true_absence, relativt total absence_minutes_total per elev; 0.0 om total
  frånvaro är 0.

Frånvaro för beteendefeatures definieras som is_true_absence: present == 0 och
cause_ext inte i (OTHERACTIVITY, WORKBASEDLEARNING) — sanktionerad närvaro räknas inte som frånvaro.

Elever exkluderas om rapporteringsgraden (REPORTED / schemalagda lektioner med schema_minutes > 0)
per termin understiger en tröskel i HT eller VT, eller om en termin saknas helt (se build_student_features).

Features för klustring listas i CLUSTERING_FEATURES (övriga kolumner är metadata eller reserverade).

morning_absence / afternoon_absence: andel lektioner med is_true_absence inom tidsfönster
(morgon: start före 09:00; eftermiddag: start efter 13:00, lokal tid).

trend_score: (VT frånvaroandel - HT frånvaroandel) där andel = sum(true_absence_minutes) / sum(schema_minutes) per termin.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project_paths import DEFAULT_RAW_PARQUET, DEFAULT_STUDENT_FEATURES

CLUSTERING_FEATURES = [
    "punctuality_score",
    "morning_absence",
    "afternoon_absence",
    "subject_variance",
    "fragmentation_index",
    "weekday_variance",
    "trend_score",
]

LOCAL_TZ = "Europe/Stockholm"
MORNING_CUTOFF_MIN = 9 * 60
# Eftermiddag: lektioner som startar strikt efter 13:00 lokal tid.
AFTERNOON_CUTOFF_MIN = 13 * 60

SANCTIONED_CAUSES = frozenset({"OTHERACTIVITY", "WORKBASEDLEARNING"})
KNOWN_ABSENCE_TYPES_FOR_GLOBALS = frozenset({"VALID", "INVALID", "NONE"})

DEFAULT_FULL_DAY_THRESHOLD = 0.9
DEFAULT_FRAGMENTATION_MIN_ABSENCE_DAYS = 3
DEFAULT_REPORTING_RATE_THRESHOLD = 0.5


def _minutes_since_midnight_local(series_local: pd.Series) -> pd.Series:
    return (
        series_local.dt.hour.astype("int64") * 60
        + series_local.dt.minute.astype("int64")
        + series_local.dt.second.astype("float64") / 60.0
    )


def _subject_variance_from_rates(rates: pd.Series) -> float:
    if len(rates) <= 1:
        return 0.0
    return float(rates.var(ddof=1))


def _safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    numer_f = numer.astype("float64")
    denom_f = denom.astype("float64")
    out = pd.Series(0.0, index=numer_f.index)
    mask = denom_f > 0
    out.loc[mask] = (numer_f.loc[mask] / denom_f.loc[mask]).to_numpy(dtype=float)
    return out


def _weekday_variance_from_rates(rates: pd.Series) -> float:
    # Varians över veckodagar; 0 om för få dagar för varians.
    if len(rates) <= 1:
        return 0.0
    return float(rates.var(ddof=1))


def _normalize_absence_type(series: pd.Series) -> pd.Series:
    """
    Normalisera absence_type till versaler; saknat/tom/'nan' → MISSING (för logg/invalid_ratio).
    """
    upper = series.astype(str).str.strip().str.upper()
    upper = upper.replace({"NAN": ""})
    return upper.mask(series.isna() | (upper == ""), "MISSING")


def _term_reporting_filter_stats(
    df: pd.DataFrame,
    id_col: str,
    reporting_rate_threshold: float,
) -> tuple[np.ndarray, int, int, int]:
    """
    Beräkna vilka elever som klarar terminsvis rapporteringsgrad.

    Exkludera om HT eller VT saknas (inga schemalagda rader med schema_minutes > 0),
    eller om rate < threshold i någon termin. NaN i rate efter pivot = saknad termin.

    Bortfallsorsak per elev (disjunkt, prioritet): (1) saknad termin, (2) låg HT, (3) låg VT.
    Returnerar (eligible_mask på df[id_col].unique(), Z, X, Y).
    """
    sched = pd.to_numeric(df["schema_minutes"], errors="coerce").fillna(0.0)
    df_s = df.loc[sched > 0].copy()
    all_ids = df[id_col].unique()
    if df_s.empty:
        # Inga schemalagda lektioner: saknad termin för alla med rader i df.
        return np.zeros(len(all_ids), dtype=bool), int(len(all_ids)), 0, 0

    n_sched = df_s.groupby([id_col, "termin"]).size()
    n_rep = (
        df_s.loc[df_s["report_status"].astype(str).eq("REPORTED")]
        .groupby([id_col, "termin"])
        .size()
    )

    all_ids = df[id_col].unique()
    n_sched_wide = n_sched.unstack(fill_value=0)
    for c in ("HT", "VT"):
        if c not in n_sched_wide.columns:
            n_sched_wide[c] = 0
    n_sched_wide = n_sched_wide.reindex(all_ids, fill_value=0)

    n_rep_wide = n_rep.unstack(fill_value=0)
    for c in ("HT", "VT"):
        if c not in n_rep_wide.columns:
            n_rep_wide[c] = 0
    n_rep_wide = n_rep_wide.reindex(all_ids, fill_value=0)

    n_ht = n_sched_wide["HT"].to_numpy(dtype=float)
    n_vt = n_sched_wide["VT"].to_numpy(dtype=float)
    rep_ht = n_rep_wide["HT"].to_numpy(dtype=float)
    rep_vt = n_rep_wide["VT"].to_numpy(dtype=float)

    missing_term = (n_ht == 0) | (n_vt == 0)
    # Undvik division med noll: np.where utvärderar båda grenarna och ger RuntimeWarning.
    rate_ht = np.full(n_ht.shape, np.nan, dtype=float)
    rate_vt = np.full(n_vt.shape, np.nan, dtype=float)
    mask_ht = n_ht > 0
    mask_vt = n_vt > 0
    rate_ht[mask_ht] = rep_ht[mask_ht] / n_ht[mask_ht]
    rate_vt[mask_vt] = rep_vt[mask_vt] / n_vt[mask_vt]

    thr = reporting_rate_threshold
    eligible_mask = (
        ~missing_term
        & np.isfinite(rate_ht)
        & np.isfinite(rate_vt)
        & (rate_ht >= thr)
        & (rate_vt >= thr)
    )

    # Disjunkt: (1) saknad termin, (2) låg HT, (3) låg VT.
    z_count = x_count = y_count = 0
    for i in range(len(all_ids)):
        if eligible_mask[i]:
            continue
        if missing_term[i]:
            z_count += 1
        elif rate_ht[i] < thr:
            x_count += 1
        elif rate_vt[i] < thr:
            y_count += 1

    return eligible_mask, z_count, x_count, y_count


def build_student_features(
    df: pd.DataFrame,
    reporting_rate_threshold: float = DEFAULT_REPORTING_RATE_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    id_col = "anon_student_id"

    if "report_status" not in df.columns:
        raise ValueError("Saknad kolumn: report_status")
    if "termin" not in df.columns:
        raise ValueError("Saknad kolumn: termin")
    if "schema_minutes" not in df.columns:
        raise ValueError("Saknad kolumn: schema_minutes")
    if "absence_minutes_total" not in df.columns:
        raise ValueError("Saknad kolumn: absence_minutes_total")

    # Säkerhetskontroller på rådata (innan terminsfilter och övrig aggregering).
    n_before_dedup = len(df)
    df = df.drop_duplicates()
    rows_removed_duplicates = int(n_before_dedup - len(df))

    schema_m = pd.to_numeric(df["schema_minutes"], errors="coerce")
    abs_m = pd.to_numeric(df["absence_minutes_total"], errors="coerce")
    both_finite = schema_m.notna() & abs_m.notna()
    abs_exceeds_schema = both_finite & (abs_m > schema_m)
    rows_capped_absence_to_schema = int(abs_exceeds_schema.sum())
    if rows_capped_absence_to_schema:
        df = df.copy()
        df.loc[abs_exceeds_schema, "absence_minutes_total"] = schema_m.loc[
            abs_exceeds_schema
        ].to_numpy(dtype=float)

    eligible_mask, students_excluded_missing_term, students_excluded_low_ht, students_excluded_low_vt = (
        _term_reporting_filter_stats(df, id_col, reporting_rate_threshold)
    )
    all_ids = df[id_col].unique()
    eligible_ids = all_ids[eligible_mask]
    df = df.loc[df[id_col].isin(eligible_ids)].copy()

    if df.empty:
        stats: dict[str, Any] = {
            "rows_removed_duplicates": rows_removed_duplicates,
            "rows_capped_absence_to_schema": rows_capped_absence_to_schema,
            "rows_dropped_unreported": 0,
            "students_after_reported": 0,
            "students_excluded_missing_term": students_excluded_missing_term,
            "students_excluded_low_reporting_ht": students_excluded_low_ht,
            "students_excluded_low_reporting_vt": students_excluded_low_vt,
            "reporting_rate_threshold": reporting_rate_threshold,
            "global_minutes_valid": 0.0,
            "global_minutes_invalid": 0.0,
            "global_minutes_other_or_missing": 0.0,
            "students_in_output": 0,
        }
        return pd.DataFrame(), stats

    rows_before_reported = len(df)
    df = df.loc[df["report_status"].astype(str).eq("REPORTED")].copy()
    rows_dropped_unreported = rows_before_reported - len(df)
    n_students_after_reported = df[id_col].nunique()

    if not pd.api.types.is_datetime64_any_dtype(df["lesson_start"]):
        df["lesson_start"] = pd.to_datetime(df["lesson_start"], utc=True)

    local = df["lesson_start"].dt.tz_convert(LOCAL_TZ)
    mins = _minutes_since_midnight_local(local)
    is_morning = mins < MORNING_CUTOFF_MIN
    is_afternoon = mins > AFTERNOON_CUTOFF_MIN

    df["_date_local"] = local.dt.date
    df["_weekday_local"] = local.dt.weekday.astype("int64")  # 0=Mon..6=Sun

    df["is_true_absence"] = df["present"].eq(0) & ~df["cause_ext"].isin(
        SANCTIONED_CAUSES
    )
    df["_true_abs"] = df["is_true_absence"]
    df["_late_arrival"] = df["cause_ext"].eq("LATEARRIVAL")

    df["_true_absence_minutes"] = np.where(
        df["is_true_absence"], df["absence_minutes_total"].to_numpy(), 0.0
    )

    abs_num = pd.to_numeric(df["absence_minutes_total"], errors="coerce").fillna(0.0)
    at_norm = _normalize_absence_type(df["absence_type"])
    global_minutes_valid = float(abs_num[at_norm == "VALID"].sum())
    global_minutes_invalid = float(abs_num[at_norm == "INVALID"].sum())
    global_minutes_other_or_missing = float(
        abs_num[~at_norm.isin(KNOWN_ABSENCE_TYPES_FOR_GLOBALS)].sum()
    )

    ids = df[id_col].unique()

    n_lessons = df.groupby(id_col).size()
    punctuality_score = df.groupby(id_col)["_late_arrival"].sum() / n_lessons

    morning_absence = (
        df.loc[is_morning].groupby(id_col)["_true_abs"].mean().reindex(ids)
    )
    afternoon_absence = (
        df.loc[is_afternoon].groupby(id_col)["_true_abs"].mean().reindex(ids)
    )

    sub_rates = df.groupby([id_col, "subject"], sort=False)["_true_abs"].mean()
    subject_variance = sub_rates.groupby(level=0, sort=False).agg(
        _subject_variance_from_rates
    )

    true_abs_by_term = df.pivot_table(
        index=id_col,
        columns="termin",
        values="_true_absence_minutes",
        aggfunc="sum",
        fill_value=0,
    )
    sched_by_term = df.pivot_table(
        index=id_col,
        columns="termin",
        values="schema_minutes",
        aggfunc="sum",
        fill_value=0,
    )

    def _term_series(pt: pd.DataFrame, term: str) -> pd.Series:
        if term in pt.columns:
            return pt[term].reindex(ids, fill_value=0)
        return pd.Series(0.0, index=ids)

    ht_abs = _term_series(true_abs_by_term, "HT")
    vt_abs = _term_series(true_abs_by_term, "VT")
    ht_sched = _term_series(sched_by_term, "HT")
    vt_sched = _term_series(sched_by_term, "VT")

    rate_ht = np.where(ht_sched > 0, ht_abs / ht_sched, 0.0)
    rate_vt = np.where(vt_sched > 0, vt_abs / vt_sched, 0.0)
    trend_score = pd.Series(rate_vt - rate_ht, index=ids)

    # invalid_ratio: administrativ INVALID / total frånvarominuter per elev; klippt [0, 1].
    # Täljare: absence_minutes_total på rader med is_true_absence och absence_type == INVALID.
    # Nämnare 0 ⇒ invalid_ratio 0.0 (ingen NaN i korrelationsanalyser).
    is_inv_true = df["is_true_absence"].to_numpy() & (at_norm == "INVALID").to_numpy()
    invalid_row_minutes = np.where(is_inv_true, abs_num.to_numpy(dtype=float), 0.0)
    invalid_minutes_sum = (
        pd.Series(invalid_row_minutes, index=df.index)
        .groupby(df[id_col])
        .sum()
        .reindex(ids, fill_value=0.0)
    )
    total_abs_min = abs_num.groupby(df[id_col]).sum().reindex(ids, fill_value=0.0)
    denom_ok = total_abs_min.to_numpy(dtype=float) > 0
    ratio_raw = np.zeros(len(ids), dtype=float)
    ratio_raw[denom_ok] = (
        invalid_minutes_sum.to_numpy(dtype=float)[denom_ok]
        / total_abs_min.to_numpy(dtype=float)[denom_ok]
    )
    invalid_ratio = pd.Series(np.clip(ratio_raw, 0.0, 1.0), index=ids, name="invalid_ratio")

    # fragmentation_index: partial-day vs full-day på dagsnivå (elev+datum).
    day = (
        df.groupby([id_col, "_date_local"], sort=False)
        .agg(day_abs_min=("absence_minutes_total", "sum"), day_sched_min=("schema_minutes", "sum"))
        .reset_index()
    )
    day["day_abs_rate"] = np.where(
        day["day_sched_min"].to_numpy(dtype=float) > 0,
        day["day_abs_min"].to_numpy(dtype=float) / day["day_sched_min"].to_numpy(dtype=float),
        0.0,
    )
    full_day_threshold = DEFAULT_FULL_DAY_THRESHOLD
    day["_is_full_day"] = day["day_abs_rate"] >= full_day_threshold
    day["_is_partial_day"] = (day["day_abs_rate"] > 0) & (day["day_abs_rate"] < full_day_threshold)
    day_counts = (
        day.groupby(id_col, sort=False)
        .agg(
            n_absence_days=("day_abs_rate", lambda s: int(np.sum(np.asarray(s) > 0))),
            n_full_days=("_is_full_day", "sum"),
            n_partial_days=("_is_partial_day", "sum"),
        )
        .reindex(ids, fill_value=0)
    )
    denom_days = (day_counts["n_partial_days"] + day_counts["n_full_days"]).astype("float64")
    raw_fragmentation = _safe_ratio(day_counts["n_partial_days"], denom_days)
    # Edge-case: beräkna endast om minst X frånvarodagar, annars 0 för stabilt KMeans-input.
    fragmentation_index = raw_fragmentation.where(
        day_counts["n_absence_days"] >= DEFAULT_FRAGMENTATION_MIN_ABSENCE_DAYS, 0.0
    ).rename("fragmentation_index")

    # weekday_variance: varians i daglig frånvaroandel över veckodagar (Friday-effect).
    # Vi använder dagsnivå-rate och tar medel per weekday, sedan varians över weekday (0..6).
    day_week = day.merge(
        df[[id_col, "_date_local", "_weekday_local"]].drop_duplicates(),
        on=[id_col, "_date_local"],
        how="left",
        validate="many_to_one",
    )
    weekday_rates = (
        day_week.groupby([id_col, "_weekday_local"], sort=False)["day_abs_rate"]
        .mean()
        .groupby(level=0, sort=False)
        .agg(_weekday_variance_from_rates)
        .reindex(ids, fill_value=0.0)
        .rename("weekday_variance")
    )

    reserved_total = df.groupby(id_col)["absence_minutes_total"].sum()

    at = df["absence_type"].str.upper()
    type_counts = (
        df.assign(_at=at).groupby([id_col, "_at"]).size().unstack(fill_value=0)
    )
    for name in ["NONE", "VALID", "INVALID"]:
        if name not in type_counts.columns:
            type_counts[name] = 0
    reserved_none = type_counts["NONE"]
    reserved_valid = type_counts["VALID"]
    reserved_invalid = type_counts["INVALID"]

    meta = df.groupby(id_col).agg(
        school_name=("school_name", "first"),
        grade=("grade", "first"),
        gender=("gender", "first"),
    )

    out = pd.concat(
        [
            meta,
            punctuality_score.rename("punctuality_score"),
            morning_absence.rename("morning_absence"),
            afternoon_absence.rename("afternoon_absence"),
            subject_variance.rename("subject_variance"),
            invalid_ratio,
            fragmentation_index,
            weekday_rates,
            trend_score.rename("trend_score"),
            reserved_total.rename("reserved_absence_minutes_total"),
            reserved_none.rename("reserved_absence_type_none"),
            reserved_valid.rename("reserved_absence_type_valid"),
            reserved_invalid.rename("reserved_absence_type_invalid"),
        ],
        axis=1,
    )
    out.index.name = id_col
    out = out.reset_index()

    out.loc[:, CLUSTERING_FEATURES] = out.loc[:, CLUSTERING_FEATURES].fillna(0.0)

    stats = {
        "rows_removed_duplicates": rows_removed_duplicates,
        "rows_capped_absence_to_schema": rows_capped_absence_to_schema,
        "rows_dropped_unreported": rows_dropped_unreported,
        "students_after_reported": n_students_after_reported,
        "students_excluded_missing_term": students_excluded_missing_term,
        "students_excluded_low_reporting_ht": students_excluded_low_ht,
        "students_excluded_low_reporting_vt": students_excluded_low_vt,
        "reporting_rate_threshold": reporting_rate_threshold,
        "global_minutes_valid": global_minutes_valid,
        "global_minutes_invalid": global_minutes_invalid,
        "global_minutes_other_or_missing": global_minutes_other_or_missing,
        "students_in_output": len(out),
    }
    return out, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregera lektions-parquet till elevnivå (student_features)."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RAW_PARQUET,
        help=f"Parquet indata (default {DEFAULT_RAW_PARQUET})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_STUDENT_FEATURES,
        help="Sparas i data/processed/ (default student_features.parquet)",
    )
    p.add_argument(
        "--reporting-rate-threshold",
        type=float,
        default=DEFAULT_REPORTING_RATE_THRESHOLD,
        help=(
            "Minsta rapporteringsgrad per termin (HT och VT), "
            f"andel REPORTED bland rader med schema_minutes > 0 (default {DEFAULT_REPORTING_RATE_THRESHOLD})."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inp = args.input.resolve()
    if not inp.is_file():
        raise SystemExit(
            f"Saknas indatafil: {inp}\n\n"
            f"Lägg din .parquet i data/raw/ som {DEFAULT_RAW_PARQUET.name}, "
            "eller kör t.ex.:\n"
            f"  PARQUET=\"/full/sökväg/din_fil.parquet\" ./scripts/run_project.sh\n"
            "eller:\n"
            f"  python3 src/preprocess.py --input /sökväg/din_fil.parquet\n"
        )
    df = pd.read_parquet(inp)
    result, stats = build_student_features(
        df, reporting_rate_threshold=args.reporting_rate_threshold
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)

    n = len(result)
    print(f"     Antal elever processade: {n}")
    print(f"     Utdatafil: {args.output.resolve()}")
    print()
    print("  --- Datakvalitet (logg) ---")
    print(
        f"     Rader borttagna (exakta dubletter): {stats['rows_removed_duplicates']}"
    )
    print(
        f"     Rader korrigerade (absence_minutes_total > schema_minutes → kapad): "
        f"{stats['rows_capped_absence_to_schema']}"
    )
    pct = int(round(stats["reporting_rate_threshold"] * 100))
    print(
        f"     Elever exkluderade pga låg HT-rapportering (< {pct}%): "
        f"{stats['students_excluded_low_reporting_ht']}"
    )
    print(
        f"     Elever exkluderade pga låg VT-rapportering (< {pct}%): "
        f"{stats['students_excluded_low_reporting_vt']}"
    )
    print(
        f"     Elever exkluderade pga saknad termin: "
        f"{stats['students_excluded_missing_term']}"
    )
    print(f"     Rader borttagna (UNREPORTED): {stats['rows_dropped_unreported']}")
    print(
        f"     Elever kvar efter terminsfilter + REPORTED: "
        f"{stats['students_after_reported']}"
    )
    gv = stats.get("global_minutes_valid", 0.0)
    gi = stats.get("global_minutes_invalid", 0.0)
    go = stats.get("global_minutes_other_or_missing", 0.0)
    gtot = gv + gi + go
    print()
    print(
        "     Frånvarominuter (globalt, REPORTED-kohort): "
        f"VALID={gv:,.0f}  INVALID={gi:,.0f}  Övrigt/Saknat={go:,.0f}"
    )
    if gtot > 0:
        print(
            f"     Andel INVALID av (VALID+INVALID+Övrigt): {100.0 * gi / gtot:.2f}%"
        )
    print()


if __name__ == "__main__":
    main()
