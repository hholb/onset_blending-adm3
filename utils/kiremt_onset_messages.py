"""
Kiremt Geba onset forecast message generator.

For each adm3 district:
 - Merges blend forecast probabilities with climatology onset DOY
 - Applies weekly thresholds to categorize onset likelihood
 - Prints a plain-language message per district

Usage (run from repo root)
--------------------------
python kiremt_onset_messages.py --district "Ada'a"
python kiremt_onset_messages.py -d "Adama town" --input_file /path/to/blend_output_summary_20260518.csv
"""

import argparse
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
# --- MODIFICATION FOR SUBFOLDER SUPPORT ---
# Identify the repository root (one level up from this script's directory)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# ── helpers ──────────────────────────────────────────────────────────────────

def doy_to_mmm_dd(doy, date_filter_year=2026):
    """Convert day of year to 'MMM DD' format."""
    date = pd.to_datetime(f"{date_filter_year}-{int(doy):03d}", format="%Y-%j")
    return date.strftime("%b %d")


def forecast_week_range(issue_date_str, week_number):
    """
    Return (start_date, end_date) strings like 'DD Month YYYY'
    for a given forecast week (1-indexed, each week = 7 days).

    week1 → issue_date to issue_date+6
    week2 → issue_date+7 to issue_date+13
    etc.
    """
    issue = datetime.strptime(issue_date_str, "%m/%d/%y")
    start = issue + timedelta(days=(week_number - 1) * 7)
    end   = issue + timedelta(days=(week_number - 1) * 7 + 6)
    fmt = "%-d %B %Y"         # Linux; change to "%#d %B %Y" on Windows
    return start.strftime(fmt), end.strftime(fmt)


def categorize_onset(w1, w2, w3, w4, thresholds):
    """
    Assign one of three categories based on cumulative weekly probabilities.

    thresholds : list of 4 floats [t1, t2, t3, t4]

    Category 1 – "onset within X weeks":
        week1 >= t1
        OR  week1+week2 >= t1
        OR  week2+week3 >= t2

    Category 2 – "onset after X weeks":
        (category 1 NOT met) AND:
        week2+week3+week4 >= t2
        OR  week3+week4 >= t3

    Category 3 – "onset is uncertain":
        neither category 1 nor category 2 met
    """
    t1, t2, t3, t4 = thresholds

    # Category 1
    if (w1 >= t1) or (w1 + w2 >= t1) or (w2 + w3 >= t2):
        return 1

    # Category 2
    if (w2 + w3 + w4 >= t2) or (w3 + w4 >= t3):
        return 2

    # Category 3
    return 3


def best_week_info(category, w1, w2, w3, w4, thresholds, issue_date_str):
    """
    Return (week_number, prob, date_start, date_end) for the triggering
    condition inside the assigned category, so the message can reference
    the correct window and probability.
    """
    t1, t2, t3, t4 = thresholds

    if category == 1:
        if w1 >= t1:
            wk = 1; prob = w1
        elif w1 + w2 >= t1:
            # spans week 1 + 2
            wk = (1, 2); prob = w1 + w2
        else:   # w2+w3 >= t2
            wk = (2, 3); prob = w2 + w3
    elif category == 2:
        if w2 + w3 + w4 >= t2:
            wk = (2, 4); prob = w2 + w3 + w4
        else:           # w3+w4 >= t3
            wk = (3, 4); prob = w3 + w4
    else:
        # uncertain – pick the highest single-week probability for reporting
        best_idx = [w1, w2, w3, w4].index(max(w1, w2, w3, w4)) + 1
        wk = best_idx
        prob = max(w1, w2, w3, w4)

    # Resolve date range
    if isinstance(wk, tuple):
        start, _ = forecast_week_range(issue_date_str, wk[0])
        _, end   = forecast_week_range(issue_date_str, wk[1])
    else:
        start, end = forecast_week_range(issue_date_str, wk)

    return wk, prob, start, end


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # ── CLI arguments ──
    parser = argparse.ArgumentParser(description="Generate Kiremt onset forecast messages.")
    parser.add_argument(
        "--district", "-d",
        type=str,
        default=None,
        help="Name of a specific adm3 district to process (case-insensitive). "
             "If omitted, all districts are processed."
    )
    parser.add_argument(
        "--input_file",
        default=None,
        help="path to the blend output summary csv file from export_blend_output.py "
    )

    parser.add_argument(
        "--output_file",
        default="predict/output/2026/kiremt_onset_categories.csv",
        help="path to the output message conversion file" 
    )

    args = parser.parse_args()

    # ── configuration ──
    THRESHOLDS = [0.70, 0.65, 0.60, 0.55]   # [t1, t2, t3, t4]
    SEASON_NAME = "Kiremt Geba"

    # ── load data ──
    if args.input_file:
        blend = pd.read_csv(args.input_file)
    else:
        blend = pd.read_csv("Monsoon_Data/Processed_Data/2026/blend_output_summary_20260518.csv")
    clim  = pd.read_csv("Monsoon_Data/Processed_Data/Models/mr_onset_idx_median_by_id.csv")

    # ── merge on district name ──
    df = blend.merge(
        clim,
        left_on="id",
        right_on="adm3_name",
        how="left"
    )

    # ── derive climatology onset date  (median_onset_idx + 120 = DOY) ──
    df["clim_doy"]  = df["median_onset_idx"] + 120
    df["clim_date"] = df["clim_doy"].apply(
        lambda d: doy_to_mmm_dd(d) if pd.notna(d) else "unknown"
    )

    # ── optional district filter ──
    if args.district:
        mask = df["id"].str.lower() == args.district.lower()
        df = df[mask]
        if df.empty:
            print(f"District '{args.district}' not found. Available names (sample):")
            print("\n".join(f"  {n}" for n in blend["id"].sort_values().head(20)))
            return

    issue_date = df["time"].iloc[0]   # e.g. "6/9/26"

    messages = []

    for _, row in df.iterrows():
        district    = row["id"]
        w1, w2, w3, w4 = row["week1"], row["week2"], row["week3"], row["week4"]
        clim_date   = row["clim_date"]

        category = categorize_onset(w1, w2, w3, w4, THRESHOLDS)
        wk, prob, date_start, date_end = best_week_info(
            category, w1, w2, w3, w4, THRESHOLDS, issue_date
        )

        pct = round(prob * 100)

        if category == 1:
            chance_word = "High"
        elif category == 2:
            chance_word = "Moderate"
        else:
            chance_word = "Uncertain"

        #msg = (
        #    f"In your woreda {district}, the chance of {SEASON_NAME} between "
        #    f"{date_start} and {date_end} is {chance_word}.\n\n"
        #    f" The chance of {SEASON_NAME} between {date_start} and {date_end} "
        #    f"is {pct} percent, meaning {pct} out of 100.\n\n"
        #    f"This is the likelihood the {SEASON_NAME} will occur between "
        #    f"{date_start} and {date_end}, not the amount of rainfall or the "
        #    f"areas of rainfall.\n\n"
        #    f"The climatology onset date is {clim_date}."
        #)

        if category == 1:
            msg = (
                f"In your woreda {district}, the chance of {SEASON_NAME} between "
                f"{date_start} and {date_end} is {chance_word}.\n\n"
                f" The chance of {SEASON_NAME} between {date_start} and {date_end} "
                f"is {pct} percent, meaning {pct} out of 100.\n\n"
                f"This is the likelihood the {SEASON_NAME} will occur between "
                f"{date_start} and {date_end}, not the amount of rainfall or the "
                f"areas of rainfall.\n\n"
                f"The climatology onset date is {clim_date}."
            )
        elif category == 2:
            msg = (
                f"In your woreda {district}, the {SEASON_NAME} is expected to onset "
                f"after {date_end}.\n\n"
                f"The chance of {SEASON_NAME} onset after {date_end} is {pct} percent, "
                f"meaning {pct} out of 100.\n\n"
                f"This is the likelihood the {SEASON_NAME} will occur after "
                f"{date_end}, not the amount of rainfall or the areas of rainfall.\n\n"
                f"The climatology onset date is {clim_date}."
            )
        else:  # category == 3
            msg = (
                f"In your woreda {district}, the onset timing of {SEASON_NAME} is uncertain.\n\n"
                f"No sufficiently high probability of onset was detected in any forecast window.\n\n"
                f"The climatology onset date is {clim_date}."
            )

        messages.append({
            "district": district,
            "category": category,
            "chance": chance_word,
            "prob_pct": pct,
            "date_start": date_start,
            "date_end": date_end,
            "clim_date": clim_date,
            "message": msg,
        })

    # ── print all messages ──
    if args.district:
        for m in messages:
            print("=" * 70)
            print(m["message"])
            print()

    # ── also save to CSV for downstream use ──
    out = pd.DataFrame(messages).drop(columns=["message"])
    #out.to_csv("predict/output/2026/kiremt_onset_categories.csv", index=False)
    out.to_csv(args.output_file, index=False)
    print(f"\n✓ Category summary saved to kiremt_onset_categories.csv")
    print(f"  Total districts: {len(messages)}")
    for cat, label in [(1, "Onset within weeks (High)"),
                       (2, "Onset after weeks (Moderate)"),
                       (3, "Uncertain")]:
        n = sum(1 for m in messages if m["category"] == cat)
        print(f"  Category {cat} – {label}: {n}")


if __name__ == "__main__":
    main()
