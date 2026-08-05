"""
run_training.py
===============
Runs the full blending training pipeline in a single command. Each step is
executed sequentially; the script verifies that expected output files exist
before proceeding to the next step.

Steps
-----
1. 1_process_raw_nc_files.py  --spec_id <clim_spec>
2. 1_process_raw_nc_files.py  --spec_id <aifs_spec>
3. 1_process_raw_nc_files.py  --spec_id <aifs_ens_spec>
4. 2_build_climatology.py     --spec_id <clim_spec>
5. 3_combine_datasets.py      --spec_id <combine_spec>
6. 0_connect_prepare_data_to_2025_pipeline.py --spec_id <connect_spec>
7. 1_blend_evaluation.py      --spec_id <blend_spec>

When --aifs_nc_folder, --aifs_ens_nc_folder, --gt_path are supplied, the script
patches the relevant yml fields before running each step by writing temporary
spec files (suffixed _op) into the specs/ directories. These are cleaned up on
exit.

Usage (run from repo root)
--------------------------
    python python/run_training.py \\
        --model_single    aifs \\
        --model_ens       aifs_ens \\
        --aifs_spec       aifs_2026 \\
        --aifs_ens_spec   aifs_ens_2026 \\
        --clim_spec       ref_rain_fixed_cutoff_2026 \\
        --combine_spec    combine_template_fixed_cutoff_2026 \\
        --connect_spec    connect_fixed_cutoff_2026 \\
        --blend_spec      cv_models_fixed_cutoff_2026 \\
        --work_dir        Monsoon_Data/Processed_Data/training \\
        --results_dir     Monsoon_Data/results/dry_spell_aifs_aifs_ens \\
        [--aifs_nc_file       /path/to/aifs.nc] \\
        [--aifs_ens_nc_file   /path/to/aifs_ens.nc] \\
        [--aifs_nc_folder     /path/to/aifs/nc/files] \\
        [--aifs_ens_nc_folder /path/to/aifs_ens/nc/files] \\
        [--gt_path            Monsoon_Data/Processed_Data/Models/dry_spell/imd_clim_mok_date_wide.pkl] \\
        [--blend_input        Monsoon_Data/Processed_Data/training/cv_data_fixed_cutoff_new_pipeline.pkl] \\
        [--skip_to STEP] \\
        [--stop_at STEP] \\
        [--dry_run]

Notes
-----
--aifs_nc_file / --aifs_ens_nc_file
    Point to a specific NetCDF file. Overrides both input.nc_folder (set to the
    file's parent directory) and input.file_regex (set to match the exact filename)
    in the aifs / aifs_ens yml specs. Takes priority over --aifs_nc_folder /
    --aifs_ens_nc_folder.

--aifs_nc_folder / --aifs_ens_nc_folder
    Override the input.nc_folder field in the aifs / aifs_ens yml specs.

--gt_path
    Path to the historical ground truth wide pkl. Simultaneously overrides:
      - input.gt_path            in the clim spec
      - ground_truth_wide_rds    in the combine spec
    Both fields must point to the same file, so a single arg controls both.

--work_dir
    Working directory for all intermediate files (processed nc outputs,
    climatology pkls, combined wide pkl, connect output pkl). Patched into
    output.out_dir in aifs/aifs_ens/clim specs, output.out_dir in combine spec,
    and input_rds/output_rds in connect spec.

--results_dir
    Directory where 1_blend_evaluation.py writes its outputs (coefs, cv_preds,
    summary csvs). Overrides run.pipeline_output_dir in the blend spec.

--skip_to N
    Skip steps 1..N-1 and start from step N (1-indexed).

--stop_at N
    Stop after completing step N (1-indexed).

--dry_run
    Print the commands that would be run without executing them.
"""

import os
import sys
import argparse
import subprocess
import atexit
from datetime import datetime

import yaml
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONNECTOR_ARTIFACT = "cv_data_fixed_cutoff_new_pipeline.pkl"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.chdir(REPO_ROOT)

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def abort(msg):
    log(msg, level="ERROR")
    sys.exit(1)


def resolve_connector_artifact(work_dir, blend_input=None):
    """Return the exact connector-output / blend-evaluator-input path."""
    return blend_input or os.path.join(work_dir, DEFAULT_CONNECTOR_ARTIFACT)


def configure_forecast_source(forecasts, model_name, template_name, source_path):
    """
    Point a combine-spec forecast entry at its processed artifact.

    Prefer the model key already declared by the selected spec (for example,
    ``ngcm``). If the spec is a generic template, rename its role key (for
    example, ``aifs_ens``) to the requested model before updating the source.
    """
    if not isinstance(forecasts, dict):
        raise ValueError("Combine spec 'forecasts' must be a mapping.")

    if model_name in forecasts:
        selected_name = model_name
    elif template_name in forecasts:
        forecasts[model_name] = forecasts.pop(template_name)
        selected_name = model_name
    else:
        available = ", ".join(sorted(forecasts)) or "<none>"
        raise KeyError(
            f"Combine spec has no forecast entry for model '{model_name}' "
            f"or template role '{template_name}'. Available keys: {available}"
        )

    sources = forecasts[selected_name].get("sources")
    if (
        not isinstance(sources, list)
        or not sources
        or not isinstance(sources[0], dict)
    ):
        raise ValueError(
            f"Combine forecast '{selected_name}' must define at least one "
            "'sources' mapping."
        )

    sources[0]["file"] = source_path
    return selected_name


def build_blend_eval_cmd(spec_id, work_dir, input_path, results_dir, cores=None):
    """Build Step 7 with an explicit connector artifact handoff."""
    cmd = [
        sys.executable,
        "python/pipelines/blending_process/1_blend_evaluation.py",
        "--spec_id", spec_id,
        "--work_dir", work_dir,
        "--input_path", input_path,
        "--results_dir", results_dir,
    ]
    if cores is not None:
        cmd += ["--cores", str(cores)]
    return cmd


def check_output_exists(path, step_name):
    if not os.path.exists(path):
        abort(
            f"Step '{step_name}' completed but expected output not found:\n"
            f"  {path}\n"
            f"Check the step's logs above for errors."
        )
    log(f"Output verified: {path}")


def run_step(step_num, total, name, cmd, expected_output, dry_run):
    log(f"── Step {step_num}/{total}: {name} ──────────────────────────────")
    cmd_str = " ".join(str(c) for c in cmd)
    log(f"Command: {cmd_str}")

    if dry_run:
        log("(dry run — skipping execution)")
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
        description="Run the full blending training pipeline in one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Required ─────────────────────────────────────────────────────────
    parser.add_argument("--model_single",   required=True,
                        help="Name of the deterministic forecast model, e.g. 'aifs'. "
                             "Overrides 'aifs' key in combine, connect, and blend specs.")
    parser.add_argument("--model_ens",      required=True,
                        help="Name of the ensemble forecast model, e.g. 'aifs_ens'. "
                             "Overrides 'aifs_ens' key in combine, connect, and blend specs.")

    parser.add_argument("--aifs_spec",      required=True,
                        help="Spec ID for step 2: process aifs nc files, e.g. aifs")
    parser.add_argument("--aifs_ens_spec",  required=True,
                        help="Spec ID for step 3: process aifs_ens nc files, e.g. aifs_ens")
    parser.add_argument("--clim_spec",      required=True,
                        help="Spec ID for step 4: build climatology, "
                             "e.g. ref_rain_fixed_cutoff")
    parser.add_argument("--combine_spec",   required=True,
                        help="Spec ID for step 5: combine datasets, "
                             "e.g. combine_template_fixed_cutoff")
    parser.add_argument("--connect_spec",   required=True,
                        help="Spec ID for step 6: connect/prepare pipeline input, "
                             "e.g. connect_fixed_cutoff")
    parser.add_argument("--blend_spec",     required=True,
                        help="Spec ID for step 7: blend evaluation, "
                             "e.g. cv_models_fixed_cutoff")
    parser.add_argument("--work_dir",       required=True,
                        help="Working directory for all intermediate files")
    parser.add_argument("--results_dir",    required=True,
                        help="Output directory for blend evaluation results "
                             "(coefs, cv_preds, summary csvs). Overrides "
                             "run.pipeline_output_dir in the blend spec.")

    # ── yml field overrides ───────────────────────────────────────────────
    parser.add_argument("--aifs_nc_file",   default=None,
                        help="Path to a specific aifs NetCDF file. Overrides both "
                             "input.nc_folder and input.file_regex in the aifs spec. "
                             "Takes priority over --aifs_nc_folder.")
    parser.add_argument("--aifs_ens_nc_file", default=None,
                        help="Path to a specific aifs_ens NetCDF file. Overrides both "
                             "input.nc_folder and input.file_regex in the aifs_ens spec. "
                             "Takes priority over --aifs_ens_nc_folder.")
    parser.add_argument("--aifs_nc_folder", default=None,
                        help="Override input.nc_folder in the aifs spec yml")
    parser.add_argument("--aifs_ens_nc_folder", default=None,
                        help="Override input.nc_folder in the aifs_ens spec yml")
    parser.add_argument("--gt_path",        default=None,
                        help="Path to historical ground truth wide pkl. Overrides "
                             "input.gt_path in the clim spec AND ground_truth_wide_rds "
                             "in the combine spec.")
    parser.add_argument("--blend_input",    default=None,
                        help="Exact connector output / blend-evaluator input pickle path. "
                             "Defaults to <work_dir>/cv_data_fixed_cutoff_new_pipeline.pkl")

    # ── Optional ─────────────────────────────────────────────────────────
    parser.add_argument("--cores",          type=int, default=None,
                        help="Number of cores passed to 1_blend_evaluation.py --cores")
    parser.add_argument("--skip_to",        type=int, default=1,
                        help="Skip to step N (1-indexed). Default: 1 (run all steps)")
    parser.add_argument("--stop_at",        type=int, default=None,
                        help="Stop after step N (1-indexed). Default: None (run all steps)")
    parser.add_argument("--dry_run",        action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

    # ── Derived paths ─────────────────────────────────────────────────────
    work_dir = args.work_dir

    os.makedirs(work_dir,      exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Patch aifs spec ───────────────────────────────────────────────────
    aifs_spec = args.aifs_spec
    if args.aifs_nc_file:
        nc_file = os.path.abspath(args.aifs_nc_file)
        aifs_spec = write_patched_spec(
            args.aifs_spec, "raw_data",
            [("input.nc_folder",  os.path.dirname(nc_file)),
             ("input.file_regex", f"^{re.escape(os.path.basename(nc_file))}$"),
             ("output.basename",  args.aifs_spec),
             ("output.out_dir",   work_dir),]
        )
    elif args.aifs_nc_folder:
        aifs_spec = write_patched_spec(
            args.aifs_spec, "raw_data",
            [("input.nc_folder",  args.aifs_nc_folder),
             ("output.basename",  args.aifs_spec),
             ("output.out_dir",   work_dir),]
        )

    # ── Patch aifs_ens spec ───────────────────────────────────────────────
    aifs_ens_spec = args.aifs_ens_spec
    if args.aifs_ens_nc_file:
        nc_file = os.path.abspath(args.aifs_ens_nc_file)
        aifs_ens_spec = write_patched_spec(
            args.aifs_ens_spec, "raw_data",
            [("input.nc_folder",  os.path.dirname(nc_file)),
             ("input.file_regex", f"^{re.escape(os.path.basename(nc_file))}$"),
             ("output.basename",  args.aifs_ens_spec),
             ("output.out_dir",   work_dir),]
        )
    elif args.aifs_ens_nc_folder:
        aifs_ens_spec = write_patched_spec(
            args.aifs_ens_spec, "raw_data",
            [("input.nc_folder",  args.aifs_ens_nc_folder),
             ("output.basename",  args.aifs_ens_spec),
             ("output.out_dir",   work_dir),]
        )

    # ── Patch clim spec (also used for step 1: process IMD nc files) ─────
    clim_spec = args.clim_spec

    clim_patches = [
        ("output.out_dir",             work_dir),
        ("paths.climatology_out_dir",  work_dir),
        ("output.basename",            args.clim_spec),
        ("input.gt_path",              args.gt_path or os.path.join(work_dir, f"{args.clim_spec}_wide.pkl")),
    ]
    clim_spec = write_patched_spec(args.clim_spec, "raw_data", clim_patches)

    # ── Patch combine spec ────────────────────────────────────────────────
    # Read clim spec for climatology output filenames
    aifs_pkl     = os.path.join(work_dir, f"{args.aifs_spec}_wide.pkl")
    aifs_ens_pkl = os.path.join(work_dir, f"{args.aifs_ens_spec}_wide.pkl")

    clim_spec_raw_path = os.path.join("specs", "raw_data", f"{args.clim_spec}.yml")
    with open(clim_spec_raw_path) as f:
        clim_spec_raw = yaml.safe_load(f)
    clim_rds     = os.path.join(work_dir, clim_spec_raw["climatologies"]["clim"]["out_stem"]     + ".pkl")
    clim_unc_rds = os.path.join(work_dir, clim_spec_raw["climatologies"]["clim_unc"]["out_stem"] + ".pkl")

    combine_spec_path = os.path.join("specs", "combine", f"{args.combine_spec}.yml")
    with open(combine_spec_path) as f:
        cs = yaml.safe_load(f)

    cs["output"]["out_dir"]                           = work_dir
    cs["output"]["basename"]                          = args.combine_spec
    cs["input"]["climatologies"]["clim"]["rds"]       = clim_rds
    cs["input"]["climatologies"]["clim_unc"]["rds"]   = clim_unc_rds
    configure_forecast_source(
        cs["forecasts"], args.model_ens, "aifs_ens", aifs_ens_pkl
    )
    configure_forecast_source(
        cs["forecasts"], args.model_single, "aifs", aifs_pkl
    )

    if args.gt_path:
        cs["input"]["ground_truth_wide_rds"] = args.gt_path

    combine_spec_op      = f"{args.combine_spec}_op"
    combine_spec_op_path = os.path.join("specs", "combine", f"{combine_spec_op}.yml")
    with open(combine_spec_op_path, "w") as f:
        yaml.dump(cs, f, default_flow_style=False, allow_unicode=True)
    _temp_spec_files.append(combine_spec_op_path)
    combine_spec = combine_spec_op
    log(f"Patched spec written: {combine_spec_op_path}")

    # ── Patch connect spec ────────────────────────────────────────────────
    combine_basename   = f"{args.combine_spec}_combined_wide.pkl"
    connect_input_rds  = os.path.join(work_dir, combine_basename)
    connect_output_rds = resolve_connector_artifact(work_dir, args.blend_input)

    connect_spec_path = os.path.join("specs", "2025_blend", f"{args.connect_spec}.yml")
    with open(connect_spec_path) as f:
        cs_connect = yaml.safe_load(f)

    cs_connect["input_rds"]  = connect_input_rds
    cs_connect["output_rds"] = connect_output_rds

    for entry in cs_connect["forecast_models"]:
        if entry["name"] == "aifs_ens":
            entry["name"] = args.model_ens
        elif entry["name"] == "aifs":
            entry["name"] = args.model_single

    connect_spec_op      = f"{args.connect_spec}_op"
    connect_spec_op_path = os.path.join("specs", "2025_blend", f"{connect_spec_op}.yml")
    with open(connect_spec_op_path, "w") as f:
        yaml.dump(cs_connect, f, default_flow_style=False, allow_unicode=True)
    _temp_spec_files.append(connect_spec_op_path)
    connect_spec = connect_spec_op
    log(f"Patched spec written: {connect_spec_op_path}")

    # ── Patch blend spec ──────────────────────────────────────────────────
    blend_spec_path = os.path.join("specs", "2025_blend", f"{args.blend_spec}.yml")
    with open(blend_spec_path) as f:
        cs_blend = yaml.safe_load(f)

    # pipeline_input_dir and pipeline_output_dir are used by 1_blend_evaluation.py
#    cs_blend["run"]["pipeline_input_dir"]  = work_dir
#    cs_blend["run"]["pipeline_output_dir"] = args.results_dir

    # rename in mme.blend_models list
    for entry in cs_blend.get("mme", {}).get("blend_models", []):
        if entry["name"] == "aifs_ens":
            entry["name"] = args.model_ens
        elif entry["name"] == "aifs":
            entry["name"] = args.model_single

    # rename in extras.forecasts list
    for entry in cs_blend.get("extras", {}).get("forecasts", []):
        if entry["name"] == "aifs_ens":
            entry["name"] = args.model_ens
        elif entry["name"] == "aifs":
            entry["name"] = args.model_single

    # substitute model names in formula text
    for model_name, formula in cs_blend["models"]["formulas"].items():
        if formula.get("enabled"):
            formula["text"] = (
                formula["text"]
                .replace("diff_aifs_ens_qx", f"diff_{args.model_ens}_qx")
                .replace("diff_aifs_qx",     f"diff_{args.model_single}_qx")
            )

    blend_spec_op      = f"{args.blend_spec}_op"
    blend_spec_op_path = os.path.join("specs", "2025_blend", f"{blend_spec_op}.yml")
    with open(blend_spec_op_path, "w") as f:
        yaml.dump(cs_blend, f, default_flow_style=False, allow_unicode=True)
    _temp_spec_files.append(blend_spec_op_path)
    blend_spec = blend_spec_op
    log(f"Patched spec written: {blend_spec_op_path}")

    # ── Expected output paths ─────────────────────────────────────────────
    ref_rain_pkl     = os.path.join(work_dir, f"{args.clim_spec}_wide.pkl")
    connect_pkl = connect_output_rds

    TOTAL = 7

    # ── Build steps list ──────────────────────────────────────────────────
    blend_eval_cmd = build_blend_eval_cmd(
        blend_spec,
        work_dir,
        connect_output_rds,
        args.results_dir,
        cores=args.cores,
    )

    steps = [
        (1, "Process IMD ground truth nc files", [
            sys.executable,
            "python/pipelines/prepare_data/1_process_raw_nc_files.py",
            "--spec_id", clim_spec,
        ], ref_rain_pkl),

        (2, f"Process {args.model_single} nc files", [
            sys.executable,
            "python/pipelines/prepare_data/1_process_raw_nc_files.py",
            "--spec_id", aifs_spec,
        ], aifs_pkl),

        (3, f"Process {args.model_ens} nc files", [
            sys.executable,
            "python/pipelines/prepare_data/1_process_raw_nc_files.py",
            "--spec_id", aifs_ens_spec,
        ], aifs_ens_pkl),

        (4, "Build climatology", [
            sys.executable,
            "python/pipelines/prepare_data/2_build_climatology.py",
            "--spec_id", clim_spec,
        ], clim_rds),

        (5, "Combine datasets", [
            sys.executable,
            "python/pipelines/prepare_data/3_combine_datasets.py",
            "--spec_id", combine_spec,
        ], None),

        (6, "Connect/prepare pipeline input", [
            sys.executable,
            "python/pipelines/blending_process/0_connect_prepare_data_to_2025_pipeline.py",
            "--spec_id", connect_spec,
        ], connect_pkl),

        (7, "Blend evaluation", blend_eval_cmd, None),
    ]

    # ── Run ───────────────────────────────────────────────────────────────
    log(f"Starting training pipeline")
    log(f"Work dir    : {work_dir}")
    log(f"Results dir : {args.results_dir}")
    log(f"Connector → evaluator artifact: {connect_output_rds}")
    if args.aifs_nc_file:
        log(f"aifs nc_file override        : {args.aifs_nc_file}")
    elif args.aifs_nc_folder:
        log(f"aifs nc_folder override      : {args.aifs_nc_folder}")
    if args.aifs_ens_nc_file:
        log(f"aifs_ens nc_file override    : {args.aifs_ens_nc_file}")
    elif args.aifs_ens_nc_folder:
        log(f"aifs_ens nc_folder override  : {args.aifs_ens_nc_folder}")
    if args.gt_path:
        log(f"gt_path override (clim+combine): {args.gt_path}")
    if args.skip_to > 1:
        log(f"Skipping steps 1–{args.skip_to - 1} (--skip_to {args.skip_to})")
    if args.stop_at is not None:
        log(f"Stopping after step {args.stop_at} (--stop_at {args.stop_at})")
    if args.dry_run:
        log("DRY RUN — commands will be printed but not executed")
    print()

    for step_num, name, cmd, expected_output in steps:
        if step_num < args.skip_to:
            log(f"── Step {step_num}/{TOTAL}: {name} [SKIPPED] ──")
            continue

        if args.stop_at is not None and step_num > args.stop_at:
            log(f"── Step {step_num}/{TOTAL}: {name} [SKIPPED — stop_at={args.stop_at}] ──")
            continue

        run_step(step_num, TOTAL, name, cmd, expected_output, args.dry_run)

    if not args.dry_run:
        log(f"Pipeline complete.")
        log(f"Intermediate files : {work_dir}")
        log(f"Training outputs   : {args.results_dir}")
    else:
        log("Dry run complete — no files were created.")


if __name__ == "__main__":
    main()
