"""run_operational_pipeline.py
===========================
Runs the full operational blending pipeline for a given forecast year/issue date
in a single command. Each step is executed sequentially; the script verifies that
expected output files exist before proceeding to the next step.

Steps
-----
1. Build climatology.
2. Process one raw rainfall specification per configured forecast.
3. Combine datasets.
4. Connect daily inputs into blending predictors.
5. Apply the saved blending model.
6. Export predictions.
7. Generate maps.

With two forecasts, the historical eight-step numbering is preserved.

The repeated --forecast interface uses paths from each raw-data spec. The
legacy AIFS interface additionally accepts file/folder overrides. Runtime paths
are written to temporary specs (suffixed _op) and cleaned up on exit.

Usage (run from repo root)
--------------------------
    python predict/run_operational_pipeline.py \\
        --year            2026 \\
        --issue_date      2026-06-09 \\
        --forecast        aifs aifs_2026 \\
        --forecast        ngcm ngcm_2026 \\
        --clim_spec       ref_rain_fixed_cutoff_2026 \\
        --combine_spec    combine_template_fixed_cutoff_2026_ngcm \\
        --connect_spec    connect_fixed_cutoff_2026_ngcm \\
        --blend_spec      cv_models_fixed_cutoff_2026_ngcm \\
        --coef_dir        Monsoon_Data/results/training \\
        --coef_tag        final \\
        --work_dir        Monsoon_Data/Processed_Data/2026 \\
        [--blend_input     Monsoon_Data/Processed_Data/2026/cv_data_fixed_cutoff_new_pipeline_2026_ngcm.pkl] \\
        [--gt_path            Monsoon_Data/Processed_Data/Models/wet_spell/ref_rain_fixed_cutoff_wide.pkl] \\
        [--skip_to STEP] \\
        [--stop_at STEP] \\
        [--dry_run]

Notes
-----
--forecast MODEL SPEC_ID
    Repeat once per forecast source. MODEL must match the selected combine,
    connect, and blend specs. The existing two-slot AIFS arguments remain
    available as a backward-compatible alternative.

--aifs_nc_file / --aifs_ens_nc_file
    Legacy-interface file overrides. They cannot be combined with --forecast.

--aifs_nc_folder / --aifs_ens_nc_folder
    Override the input.nc_folder field in the aifs / aifs_ens yml specs.

--gt_path
    Path to the historical ground truth wide pkl. Simultaneously overrides:
      - input.gt_path  in the clim spec     (ref_rain_fixed_cutoff_2026.yml)
      - ground_truth_wide_rds in the combine spec (combine_template_*_2026.yml)
    Both fields must point to the same file, so a single arg controls both.

--skip_to N
    Skip steps 1..N-1 and start from step N (1-indexed). Useful for resuming
    after a failure without rerunning expensive earlier steps.

--stop_at N
    Stop after completing step N (1-indexed). Useful for running only the first
    N steps of the pipeline.

--dry_run
    Print the commands that would be run without executing them.
"""

import os
import sys
import argparse
import subprocess
import textwrap
import atexit
import pickle
from datetime import datetime

import yaml
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.chdir(REPO_ROOT)

from python.run_training import (  # noqa: E402
    configure_blend_spec_models,
    configure_forecast_source,
    load_yaml_spec,
    normalize_forecast_jobs,
    reconcile_forecast_horizon,
    resolve_model_entries,
    validate_blend_spec_models,
)

# ---------------------------------------------------------------------------
# Temp spec management
# ---------------------------------------------------------------------------

_temp_spec_files = []   # paths to clean up on exit

def _cleanup_temp_specs():
    for path in _temp_spec_files:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

atexit.register(_cleanup_temp_specs)


def write_patched_spec(base_spec_id, spec_type, patches):
    """
    Load specs/<spec_type>/<base_spec_id>.yml, apply nested key patches,
    write to specs/<spec_type>/<base_spec_id>_op.yml, and return the new spec_id.

    patches is a list of (dotted_key, value) tuples, e.g.:
        [("input.nc_folder", "/new/path"), ("input.gt_path", "/other/path")]
    """
    src_path = os.path.join("specs", spec_type, f"{base_spec_id}.yml")
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Base spec not found: {src_path}")

    with open(src_path) as f:
        spec = yaml.safe_load(f)

    for dotted_key, value in patches:
        keys = dotted_key.split(".")
        node = spec
        for k in keys[:-1]:
            if k not in node:
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value

    new_spec_id = f"{base_spec_id}_op"
    dst_path = os.path.join("specs", spec_type, f"{new_spec_id}.yml")
    with open(dst_path, "w") as f:
        yaml.dump(spec, f, default_flow_style=False, allow_unicode=True)

    _temp_spec_files.append(dst_path)
    log(f"Patched spec written: {dst_path}")
    return new_spec_id


def write_temp_spec(base_spec_id, spec_type, spec):
    """Write an already-resolved specification using the existing _op convention."""
    new_spec_id = f"{base_spec_id}_op"
    path = os.path.join("specs", spec_type, f"{new_spec_id}.yml")
    with open(path, "w") as f:
        yaml.dump(spec, f, default_flow_style=False, allow_unicode=True)
    _temp_spec_files.append(path)
    log(f"Patched spec written: {path}")
    return new_spec_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def abort(msg):
    log(msg, level="ERROR")
    sys.exit(1)


def check_output_exists(path, step_name):
    if not os.path.exists(path):
        abort(
            f"Step '{step_name}' completed but expected output not found:\n"
            f"  {path}\n"
            f"Check the step's logs above for errors."
        )
    log(f"Output verified: {path}")


def required_probability_prefixes_from_coefs(coef_path, connect_spec):
    """Return connector probability series used by the saved feature schema."""
    with open(coef_path, "rb") as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict):
        features = bundle.get("features")
    elif hasattr(bundle, "columns") and "feature" in bundle.columns:
        features = bundle["feature"].dropna().astype(str).unique().tolist()
    else:
        features = None
    if not features:
        raise ValueError(
            f"Coefficient artifact has no saved feature schema: {coef_path}"
        )

    configured = {
        f"{model['name']}_p_onset"
        + ("" if variant is None else f"_{variant}")
        for model in connect_spec.get("forecast_models", [])
        for variant in [None] + list(model.get("variants") or [])
    }
    required = set()
    for model in connect_spec.get("forecast_models", []):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_])"
            rf"({re.escape(model['name'])}_p_onset(?:_[A-Za-z0-9_]+)?)"
            rf"_week\d+"
        )
        for feature in features:
            required.update(pattern.findall(str(feature)))
    unknown = sorted(required - configured)
    if unknown:
        raise ValueError(
            "Coefficient feature schema requires forecast probability series "
            "not declared by connect forecast_models: " + ", ".join(unknown)
        )
    return sorted(required)


def run_step(step_num, total, name, cmd, expected_output, dry_run):
    log(f"-- Step {step_num}/{total}: {name} ------------------------------")
    cmd_str = " ".join(cmd)
    log(f"Command: {cmd_str}")

    if dry_run:
        log("(dry run - skipping execution)")
        return

    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        abort(f"Step '{name}' failed with exit code {result.returncode}.")

    if expected_output:
        check_output_exists(expected_output, name)

    log(f"Step {step_num}/{total} complete.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the full operational blending pipeline in one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Example
            -------
            python predict/run_operational_pipeline.py \\
                --year 2026 \\
                --issue_date 2026-06-09 \\
                --forecast aifs aifs_2026 \\
                --forecast ngcm ngcm_2026 \\
                --clim_spec ref_rain_fixed_cutoff_2026 \\
                --combine_spec combine_template_fixed_cutoff_2026_ngcm \\
                --connect_spec connect_fixed_cutoff_2026_ngcm \\
                --blend_spec cv_models_fixed_cutoff_2026_ngcm \\
                --coef_dir Monsoon_Data/results/training \\
                --coef_tag final \\
                --work_dir Monsoon_Data/Processed_Data/2026 \\
                --gt_path Monsoon_Data/Processed_Data/Models/wet_spell/ref_rain_fixed_cutoff_wide.pkl
        """),
    )

    # -- Forecast sources --------------------------------------------------
    parser.add_argument(
        "--forecast",
        action="append",
        nargs=2,
        metavar=("MODEL", "SPEC_ID"),
        help=(
            "Forecast model name and raw-data spec ID. Repeat once per source, "
            "for example: --forecast aifs aifs_2026 --forecast ngcm ngcm_2026"
        ),
    )
    parser.add_argument("--model_single", default=None,
                        help="Legacy deterministic forecast model name.")
    parser.add_argument("--model_ens", default=None,
                        help="Legacy ensemble forecast model name.")
    parser.add_argument("--aifs_spec", default=None,
                        help="Legacy deterministic raw-data spec ID.")
    parser.add_argument("--aifs_ens_spec", default=None,
                        help="Legacy ensemble raw-data spec ID.")

    # -- Required pipeline inputs -----------------------------------------
    parser.add_argument("--year",          required=True)
    parser.add_argument("--issue_date",    required=True,
                        help="Forecast issue date, e.g. 2026-06-09")
    parser.add_argument("--clim_spec",     required=True,
                        help="Base spec ID for 2_build_climatology, e.g. ref_rain_fixed_cutoff_2026")
    parser.add_argument("--combine_spec",  required=True,
                        help="Base spec ID for 3_combine_datasets, e.g. combine_template_fixed_cutoff_2026")
    parser.add_argument("--connect_spec",  required=True,
                        help="Spec ID for 0_connect_prepare_data_to_2025_pipeline")
    parser.add_argument("--blend_spec",    required=True,
                        help="Spec ID for apply_blend_model")
    parser.add_argument("--coef_dir",      required=True,
                        help="Directory containing the blending model coef pkl")
    parser.add_argument("--coef_tag",      required=True,
                        help="Coef tag passed to apply_blend_model --coef_tag")
    parser.add_argument("--work_dir",      required=True,
                        help="Working output directory for intermediate and final files")

    # -- yml field overrides -----------------------------------------------
    parser.add_argument("--aifs_nc_folder", default=None,
                        help="Override input.nc_folder in the aifs spec yml")
    parser.add_argument("--aifs_ens_nc_folder", default=None,
                        help="Override input.nc_folder in the aifs_ens spec yml")
    parser.add_argument("--aifs_nc_file", default=None,
                        help="Path to a specific aifs NetCDF file to process. "
                             "Overrides both input.nc_folder and input.file_regex "
                             "in the aifs spec yml. Takes priority over --aifs_nc_folder.")
    parser.add_argument("--aifs_ens_nc_file", default=None,
                        help="Path to a specific aifs_ens NetCDF file to process. "
                             "Overrides both input.nc_folder and input.file_regex "
                             "in the aifs_ens spec yml. Takes priority over --aifs_ens_nc_folder.")
    parser.add_argument("--gt_path", default=None,
                        help="Path to historical ground truth wide pkl. Overrides "
                             "input.gt_path in the clim spec AND ground_truth_wide_rds "
                             "in the combine spec (both must point to the same file)")
    parser.add_argument("--blend_input", default=None,
                    help="Exact connector output and blend-application input path. "
                         "Defaults to the selected connect spec's output basename "
                         "inside <work_dir>.")

    # -- Optional ---------------------------------------------------------
    parser.add_argument("--map_output_path", default=None,
                        help="Reserved for compatibility; maps currently use work_dir.")
    parser.add_argument("--blend_model",   default="blended_model",
                        help="Blended model name (default: blended_model)")
    parser.add_argument("--region",        default="Ethiopia",
                        help="Region passed to run_maps (default: Ethiopia)")
    parser.add_argument("--skip_to",       type=int, default=1,
                        help="Skip to step N (1-indexed). Default: 1 (run all steps)")
    parser.add_argument("--stop_at",       type=int, default=None,
                        help="Stop after step N (1-indexed). Default: None (run all steps)")
    parser.add_argument("--dry_run",       action="store_true",
                        help="Print commands without executing them")
    parser.add_argument("--regrid_spec",   default=None,
                        help="Optional. Regrid spec id (specs/regrid/<id>.yml). When set, gridded "
                             "rainfall is regridded onto the shapefile's admin units before step 1 "
                             "(use when the target boundaries are political). Omit to run on the "
                             "existing grid. The step-1 specs must then read the *_adm3.nc outputs.")
    args = parser.parse_args()

    try:
        forecast_jobs = normalize_forecast_jobs(args)
    except ValueError as exc:
        parser.error(str(exc))

    # -- Derived paths -----------------------------------------------------
    year       = args.year
    issue_date = args.issue_date
    work_dir   = args.work_dir
    date_compact = issue_date.replace("-", "")

    os.makedirs(work_dir, exist_ok=True)

    # -- Resolve connector model entries and rainfall horizons -------------
    cs_connect = load_yaml_spec(args.connect_spec, "2025_blend")
    connect_entries = resolve_model_entries(
        cs_connect.get("forecast_models"),
        forecast_jobs,
        "connect forecast_models",
    )
    probability_day_max = cs_connect.get("day_max")

    # -- Patch one raw-data spec per forecast source -----------------------
    for job in forecast_jobs:
        raw_spec = load_yaml_spec(job.spec_id, "raw_data")
        job.rain_day_max, job.rain_horizon_policy = reconcile_forecast_horizon(
            raw_spec,
            connect_entries[job.model],
            job.model,
            probability_day_max,
        )

        output = raw_spec.setdefault("output", {})
        output["basename"] = job.spec_id
        output["out_dir"] = work_dir

        if job.nc_file:
            nc_file = os.path.abspath(job.nc_file)
            input_spec = raw_spec.setdefault("input", {})
            input_spec["nc_folder"] = os.path.dirname(nc_file)
            input_spec["file_regex"] = (
                f"^{re.escape(os.path.basename(nc_file))}$"
            )
        elif job.nc_folder:
            raw_spec.setdefault("input", {})["nc_folder"] = job.nc_folder

        job.runtime_spec_id = write_temp_spec(
            job.spec_id, "raw_data", raw_spec
        )
        job.artifact_path = os.path.join(
            work_dir, f"{job.spec_id}_wide.pkl"
        )

    # -- Patch climatology spec --------------------------------------------
    clim_spec = write_patched_spec(
        args.clim_spec, "raw_data",
        [("output.basename",           args.clim_spec),
         ("paths.climatology_out_dir", work_dir),
         ("input.gt_path",             args.gt_path or os.path.join(work_dir, f"{args.clim_spec}_wide.pkl")),]
    )

    # -- Patch combine spec ------------------------------------------------
    clim_spec_raw = load_yaml_spec(args.clim_spec, "raw_data")
    clim_rds     = os.path.join(work_dir, clim_spec_raw["climatologies"]["clim"]["out_stem"]     + ".pkl")
    clim_unc_rds = os.path.join(work_dir, clim_spec_raw["climatologies"]["clim_unc"]["out_stem"] + ".pkl")

    cs = load_yaml_spec(args.combine_spec, "combine")
    cs["output"]["out_dir"]                           = work_dir
    cs["output"]["basename"]                          = args.combine_spec
    cs["input"]["climatologies"]["clim"]["rds"]       = clim_rds
    cs["input"]["climatologies"]["clim_unc"]["rds"]   = clim_unc_rds
    for job in forecast_jobs:
        configure_forecast_source(
            cs.get("forecasts"),
            job.model,
            job.template_name,
            job.artifact_path,
        )
    if args.gt_path:
        cs["input"]["ground_truth_wide_rds"] = args.gt_path
    combine_spec = write_temp_spec(args.combine_spec, "combine", cs)

    # -- Patch connect spec ------------------------------------------------
    combine_basename = f"{args.combine_spec}_combined_wide.pkl"
    connect_input_rds = os.path.join(work_dir, combine_basename)
    configured_connect_output = cs_connect.get("output_rds")
    if not configured_connect_output:
        raise ValueError(
            f"Connect spec '{args.connect_spec}' must define output_rds."
        )
    connect_output_rds = args.blend_input or os.path.join(
        work_dir, os.path.basename(configured_connect_output)
    )
    cs_connect["input_rds"]  = connect_input_rds
    cs_connect["output_rds"] = connect_output_rds
    connect_spec = write_temp_spec(
        args.connect_spec, "2025_blend", cs_connect
    )

    coef_path = os.path.join(
        args.coef_dir,
        f"coefs_{args.blend_model}_global_{args.coef_tag}.pkl",
    )
    required_probability_prefixes = required_probability_prefixes_from_coefs(
        coef_path, cs_connect
    )
    log(
        "Required forecast probability series: "
        + (", ".join(required_probability_prefixes) or "none (rainfall-only)")
    )

    # -- Patch blend spec --------------------------------------------------
    cs_blend = load_yaml_spec(args.blend_spec, "2025_blend")
    configure_blend_spec_models(cs_blend, forecast_jobs)
    validate_blend_spec_models(cs_blend, forecast_jobs)
    blend_spec = write_temp_spec(args.blend_spec, "2025_blend", cs_blend)

    # -- Expected output paths ---------------------------------------------
    connect_pkl  = connect_output_rds
    preds_pkl    = os.path.join(work_dir, f"{args.blend_model}_global_year{year}_preds.pkl")
    export_csv   = os.path.join(work_dir, f"blend_output_summary_{date_compact}.csv")

    # -- Build steps list --------------------------------------------------
    forecast_steps = []
    for step_num, job in enumerate(forecast_jobs, start=2):
        if job.runtime_spec_id is None or job.artifact_path is None:
            raise RuntimeError(
                f"Forecast job {job.model!r} has not been fully configured."
            )
        forecast_steps.append((
            step_num,
            f"Process {job.model} nc files",
            [
                sys.executable,
                "python/pipelines/prepare_data/1_process_raw_nc_files.py",
                "--spec_id", job.runtime_spec_id,
            ],
            job.artifact_path,
        ))

    combine_step = len(forecast_jobs) + 2
    connect_step = combine_step + 1
    apply_step = connect_step + 1
    export_step = apply_step + 1
    map_step = export_step + 1
    total_steps = map_step

    steps = [
        (1, "Build climatology", [
            sys.executable,
            "python/pipelines/prepare_data/2_build_climatology.py",
            "--spec_id", clim_spec,
        ], None),

        *forecast_steps,

        (combine_step, "Combine datasets", [
            sys.executable,
            "python/pipelines/prepare_data/3_combine_datasets.py",
            "--spec_id", combine_spec,
        ], None),

        (connect_step, "Connect/prepare pipeline input", [
            sys.executable,
            "python/pipelines/blending_process/0_connect_prepare_data_to_2025_pipeline.py",
            "--spec_id", connect_spec,
            "--required_probability_prefixes",
            *required_probability_prefixes,
        ], connect_pkl),

        (apply_step, "Apply blend model", [
            sys.executable,
            "predict/apply_blend_model.py",
            "--spec_id",    blend_spec,
            "--model",      args.blend_model,
            "--year",       year,
            "--coef_dir",   args.coef_dir,
            "--coef_tag",   args.coef_tag,
            "--input_path", connect_pkl,
            "--out_dir",    work_dir,
        ], preds_pkl),

        (export_step, "Export blend output", [
            sys.executable,
            "predict/export_blend_output.py",
            "--issue_date", issue_date,
            "--spec_id",    blend_spec,
            "--preds_file", preds_pkl,
            "--out_dir",    work_dir,
        ], export_csv),

        (map_step, "Generate maps", [
            sys.executable,
            "predict/run_maps.py",
            "--input_file",  export_csv,
            "--output_path", work_dir,
            "--region",      args.region,
        ], None),
    ]

    # -- Run ---------------------------------------------------------------
    log(f"Starting operational pipeline for year={year}, issue_date={issue_date}")
    log(f"Work dir    : {work_dir}")
    log("Forecast sources:")
    for job in forecast_jobs:
        if job.rain_day_max is None:
            horizon = "legacy rain horizon"
        else:
            horizon = (
                f"rain_day_max={job.rain_day_max} "
                f"({job.rain_horizon_policy})"
            )
        log(f"  {job.model}: spec={job.spec_id}, {horizon}")
        if job.nc_file:
            log(f"    nc_file override: {job.nc_file}")
        elif job.nc_folder:
            log(f"    nc_folder override: {job.nc_folder}")
    log(f"Connector -> application artifact: {connect_output_rds}")
    if args.gt_path:
        log(f"gt_path override (clim+combine): {args.gt_path}")
    if args.skip_to > 1:
        log(f"Skipping steps 1-{args.skip_to - 1} (--skip_to {args.skip_to})")
    if args.dry_run:
        log("DRY RUN - commands will be printed but not executed")
    print()

    if args.stop_at is not None:
        log(f"Stopping after step {args.stop_at} (--stop_at {args.stop_at})")

    # -- Optional pre-step 0: regrid gridded rainfall onto political boundaries --
    if args.regrid_spec:
        if args.skip_to > 1:
            log(f"-- Step 0/{total_steps}: Regrid to shapefile [SKIPPED - skip_to={args.skip_to}] --")
        else:
            run_step("0", total_steps, "Regrid rainfall to shapefile (political boundaries)", [
                sys.executable,
                "python/pipelines/prepare_data/0_regrid_to_shapefile.py",
                "--spec_id", args.regrid_spec,
            ], None, args.dry_run)
    else:
        log("No --regrid_spec given; running on the existing grid (no regridding).")

    for step_num, name, cmd, expected_output in steps:
        if step_num < args.skip_to:
            log(f"-- Step {step_num}/{total_steps}: {name} [SKIPPED] --")
            continue

        if args.stop_at is not None and step_num > args.stop_at:
            log(f"-- Step {step_num}/{total_steps}: {name} [SKIPPED - stop_at={args.stop_at}] --")
            continue

        run_step(step_num, total_steps, name, cmd, expected_output, args.dry_run)


    if not args.dry_run:
        log(f"Pipeline complete. Outputs in : {work_dir}")
    else:
        log("Dry run complete - no files were created.")


if __name__ == "__main__":
    main()
