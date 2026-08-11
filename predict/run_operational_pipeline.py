"""--issue_date      2026-06-09
run_operational_pipeline.py
===========================
Runs the full operational blending pipeline for a given forecast year/issue date
in a single command. Each step is executed sequentially; the script verifies that
expected output files exist before proceeding to the next step.

Steps
-----
1. 1_process_raw_nc_files.py  --spec_id <aifs_spec>
2. 1_process_raw_nc_files.py  --spec_id <aifs_ens_spec>
3. 2_build_climatology.py     --spec_id <clim_spec>
4. 3_combine_datasets.py      --spec_id <combine_spec>
5. 0_connect_prepare_data_to_2025_pipeline.py --spec_id <connect_spec>
6. apply_blend_model.py       (with user-supplied coef args)
7. export_blend_output.py     (with --preds_file)
8. run_maps.py                (with --input_file)

When --aifs_nc_folder, --aifs_ens_nc_folder, --gt_path are supplied, the script
patches the relevant yml fields before running each step by writing temporary
spec files (suffixed _op) into the specs/ directories. These are cleaned up on
exit.

Usage (run from repo root)
--------------------------
    python predict/run_operational_pipeline.py \\
        --year            2026 \\
        --issue_date      2026-06-09 \\
        --model_single    aifs \\
        --model_ens       aifs_ens \\
        --aifs_spec       aifs_2026 \\
        --aifs_ens_spec   aifs_ens_2026 \\
        --clim_spec       ref_rain_fixed_cutoff_2026 \\
        --combine_spec    combine_template_fixed_cutoff_2026 \\
        --connect_spec    connect_fixed_cutoff_2026 \\
        --blend_spec      cv_models_fixed_cutoff_2026 \\
        --coef_dir        Monsoon_Data/results/wet_spell_aifs_aifs_ens \\
        --coef_tag        fixed_cutoff_2022_year2022 \\
        --work_dir        Monsoon_Data/Processed_Data/2026 \\
        [--aifs_nc_file       /path/to/aifs/2026.nc] \
        [--aifs_ens_nc_file   /path/to/aifs_ens/2026.nc] \
        [--aifs_nc_folder     /path/to/aifs/nc/files] \\
        [--aifs_ens_nc_folder /path/to/aifs_ens/nc/files] \\
        [--blend_input     Monsoon_Data/Processed_Data/2026/cv_data_fixed_cutoff_new_pipeline_2026.pkl] \\
        [--gt_path            Monsoon_Data/Processed_Data/Models/wet_spell/ref_rain_fixed_cutoff_wide.pkl] \\
        [--map_output_path    predict/output/2026/] \\
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
import shutil
import atexit
import pickle
from datetime import datetime

import yaml
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
                --aifs_spec aifs_2026 \\
                --aifs_ens_spec aifs_ens_2026 \\
                --clim_spec ref_rain_fixed_cutoff_2026 \\
                --combine_spec combine_template_fixed_cutoff_2026 \\
                --connect_spec connect_fixed_cutoff_2026 \\
                --blend_spec cv_models_fixed_cutoff_2026 \\
                --coef_dir Monsoon_Data/results/wet_spell_aifs_aifs_ens \\
                --coef_tag fixed_cutoff_2022_year2022 \\
                --blend_input Monsoon_Data/Processed_Data/2026/cv_data_fixed_cutoff_new_pipeline_2026.pkl \\
                --work_dir Monsoon_Data/Processed_Data/2026 \\
                --aifs_nc_folder /data/forecasts/aifs/2026 \\
                --aifs_ens_nc_folder /data/forecasts/aifs_ens/2026 \\
                --gt_path Monsoon_Data/Processed_Data/Models/wet_spell/ref_rain_fixed_cutoff_wide.pkl
        """),
    )

    # -- Required ---------------------------------------------------------
    parser.add_argument("--year",          required=True)
    parser.add_argument("--issue_date",    required=True,
                        help="Forecast issue date, e.g. 2026-06-09")
    parser.add_argument("--aifs_spec",     required=True,
                        help="Base spec ID for aifs 1_process_raw_nc_files, e.g. aifs_2026")
    parser.add_argument("--aifs_ens_spec", required=True,
                        help="Base spec ID for aifs_ens 1_process_raw_nc_files, e.g. aifs_ens_2026")
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
    #parser.add_argument("--blend_input",   required=True,
    #                    help="Path to the wide pipeline pkl for apply_blend_model --input_path")
    parser.add_argument("--work_dir",      required=True,
                        help="Working output directory for intermediate and final files")
    parser.add_argument("--model_single", required=True,
                    help="Name of the single deterministic forecast model, e.g. 'aifs'. "
                         "Overrides 'aifs' key in combine, connect, and cv_models specs.")
    parser.add_argument("--model_ens",    required=True,
                    help="Name of the ensemble forecast model, e.g. 'aifs_ens'. "
                         "Overrides 'aifs_ens' key in combine, connect, and cv_models specs.")

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
                    help="Path to the wide pipeline pkl for apply_blend_model --input_path. "
                         "Defaults to <work_dir>/cv_data_fixed_cutoff_new_pipeline_<year>.pkl")

    # -- Optional ---------------------------------------------------------
    parser.add_argument("--map_output_path", default=None,
                        help="Output directory for maps. Default: predict/output/{year}/")
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

    # -- Derived paths -----------------------------------------------------
    year       = args.year
    issue_date = args.issue_date
    work_dir   = args.work_dir
    #map_out    = args.map_output_path or os.path.join("predict", "output", year)
    date_compact = issue_date.replace("-", "")

    os.makedirs(work_dir, exist_ok=True)
    #os.makedirs(map_out,  exist_ok=True)

    # -- Patch specs where overrides are provided --------------------------
    aifs_spec     = args.aifs_spec
    aifs_ens_spec = args.aifs_ens_spec
    clim_spec     = args.clim_spec
    combine_spec  = args.combine_spec

    if args.aifs_nc_file:
        nc_file = os.path.abspath(args.aifs_nc_file)
        aifs_spec = write_patched_spec(
            args.aifs_spec, "raw_data",
            [("input.nc_folder",   os.path.dirname(nc_file)),
             ("input.file_regex",  f"^{re.escape(os.path.basename(nc_file))}$"),
             ("output.basename",   args.aifs_spec),
             ("output.out_dir",    work_dir),]
        )
        _temp_spec_files.append(os.path.join(work_dir, f"{args.aifs_spec}_op_wide.pkl"))
    elif args.aifs_nc_folder:
        aifs_spec = write_patched_spec(
            args.aifs_spec, "raw_data",
            [("input.nc_folder",   args.aifs_nc_folder),
             ("output.basename",   args.aifs_spec),
             ("output.out_dir",    work_dir),]
        )
        _temp_spec_files.append(os.path.join(work_dir, f"{args.aifs_spec}_op_wide.pkl"))

    if args.aifs_ens_nc_file:
        nc_file = os.path.abspath(args.aifs_ens_nc_file)
        aifs_ens_spec = write_patched_spec(
            args.aifs_ens_spec, "raw_data",
            [("input.nc_folder",   os.path.dirname(nc_file)),
             ("input.file_regex",  f"^{re.escape(os.path.basename(nc_file))}$"),
             ("output.basename",   args.aifs_ens_spec),
             ("output.out_dir",    work_dir),]
        )
        _temp_spec_files.append(os.path.join(work_dir, f"{args.aifs_ens_spec}_op_wide.pkl"))
    elif args.aifs_ens_nc_folder:
        aifs_ens_spec = write_patched_spec(
            args.aifs_ens_spec, "raw_data",
            [("input.nc_folder",   args.aifs_ens_nc_folder),
             ("output.basename",   args.aifs_ens_spec),
             ("output.out_dir",    work_dir),]
        )
        _temp_spec_files.append(os.path.join(work_dir, f"{args.aifs_ens_spec}_op_wide.pkl"))

    #if args.gt_path:
    #    clim_spec = write_patched_spec(
    #        args.clim_spec, "raw_data",
    #        [("input.gt_path",          args.gt_path),
    #         ("output.basename",        args.clim_spec),
    #         ("paths.climatology_out_dir", work_dir),]
    #    )
    
    clim_spec = write_patched_spec(
        args.clim_spec, "raw_data",
        [("output.basename",           args.clim_spec),
         ("paths.climatology_out_dir", work_dir),
         ("input.gt_path",             args.gt_path or os.path.join(work_dir, f"{args.clim_spec}_wide.pkl")),]
    )

#        combine_spec = write_patched_spec(
#            args.combine_spec, "combine",
#            [("ground_truth_wide_rds",  args.gt_path),
#             ("output.basename",        args.combine_spec)]
#        )
#        _temp_spec_files.append(os.path.join(work_dir, f"{args.combine_spec}_op_combined_wide.pkl"))


    # ---------------------------------------------------------------------
    # patch combine spec
    # Read clim spec for climatology output filenames (always, not just when gt_path set)
    #aifs_pkl     = os.path.join(work_dir, f"aifs_{year}_wide.pkl")
    #aifs_ens_pkl = os.path.join(work_dir, f"aifs_ens_{year}_wide.pkl")
    aifs_pkl     = os.path.join(work_dir, f"{args.aifs_spec}_wide.pkl")
    aifs_ens_pkl = os.path.join(work_dir, f"{args.aifs_ens_spec}_wide.pkl")

    clim_spec_path = os.path.join("specs", "raw_data", f"{args.clim_spec}.yml")
    with open(clim_spec_path) as f:
        clim_spec_raw = yaml.safe_load(f)
    clim_rds     = os.path.join(work_dir, clim_spec_raw["climatologies"]["clim"]["out_stem"]     + ".pkl")
    clim_unc_rds = os.path.join(work_dir, clim_spec_raw["climatologies"]["clim_unc"]["out_stem"] + ".pkl")

    # Patch combine_spec with work_dir paths (always)
    combine_spec_path = os.path.join("specs", "combine", f"{combine_spec}.yml")
    with open(combine_spec_path) as f:
        cs = yaml.safe_load(f)
    cs["output"]["out_dir"]                           = work_dir
    cs["input"]["climatologies"]["clim"]["rds"]       = clim_rds
    cs["input"]["climatologies"]["clim_unc"]["rds"]   = clim_unc_rds
    cs["forecasts"]["aifs_ens"]["sources"][0]["file"] = aifs_ens_pkl
    cs["forecasts"]["aifs"]["sources"][0]["file"]     = aifs_pkl

    # rename forecast keys if model names differ from defaults
    if args.model_single != "aifs":
        cs["forecasts"][args.model_single] = cs["forecasts"].pop("aifs")
    if args.model_ens != "aifs_ens":
        cs["forecasts"][args.model_ens] = cs["forecasts"].pop("aifs_ens")

    cs["output"]["basename"] = args.combine_spec   # pins output filename to original name
    if args.gt_path:
        cs["input"]["ground_truth_wide_rds"] = args.gt_path
    combine_spec_op      = f"{args.combine_spec}_op"
    combine_spec_op_path = os.path.join("specs", "combine", f"{combine_spec_op}.yml")
    with open(combine_spec_op_path, "w") as f:
        yaml.dump(cs, f, default_flow_style=False, allow_unicode=True)
    _temp_spec_files.append(combine_spec_op_path)
    #_temp_spec_files.append(os.path.join(work_dir, f"{args.combine_spec}_op_combined_wide.pkl"))
    combine_spec = combine_spec_op
    log(f"Patched spec written: {combine_spec_op_path}")


    # ---------------------------------------------------------------------
    # patch connect spec
    # Patch connect_spec input_rds to match the _op combine output basename.
    # write_patched_spec appends _op to combine_spec, so the combine output is
    # named <combine_spec>_op_combined_wide.pkl - the connect spec must match.
    #combine_basename = f"{combine_spec}_combined_wide.pkl"

#    # NEW - reads output.out_dir from the combine spec itself, so subdirs like dry_spell_strict/ are respected
#    def _read_combine_out_dir(spec_id):
#        """Return output.out_dir from specs/combine/<spec_id>.yml, falling back to work_dir."""
#        path = os.path.join("specs", "combine", f"{spec_id}.yml")
#        if os.path.exists(path):
#            with open(path) as f:
#                spec = yaml.safe_load(f)
#            return spec.get("output", {}).get("out_dir", work_dir)
#        return work_dir
#    
#    # Use the base (un-patched) combine_spec to find where 3_combine_datasets.py will write
#    combine_out_dir   = _read_combine_out_dir(args.combine_spec)
#    combine_basename  = f"{args.combine_spec}_combined_wide.pkl"
#    connect_input_rds = os.path.join(combine_out_dir, combine_basename)
#    connect_spec = write_patched_spec(
#        args.connect_spec, "2025_blend",
#        [("input_rds", connect_input_rds)]
#    )

    # replace write_patched_spec with a manual read/mutate/write (same pattern as combine) 
    connect_spec_path = os.path.join("specs", "2025_blend", f"{args.connect_spec}.yml")
    combine_basename = f"{args.combine_spec}_combined_wide.pkl"
    connect_input_rds = os.path.join(work_dir, combine_basename)
    connect_output_rds = os.path.join(work_dir, f"cv_data_fixed_cutoff_new_pipeline_{year}.pkl")
    if args.blend_input:
        connect_output_rds = args.blend_input   # honour explicit override
    #connect_spec = write_patched_spec(
    #    args.connect_spec, "2025_blend",
    #    [("input_rds", connect_input_rds),
    #     ("output_rds", connect_output_rds),]
    #)

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


    # ---------------------------------------------------------------------
    # patch blend spec
    # blend spec patch, also manual read/mutate/write since we need list mutation and formula string substitution
    blend_spec_path = os.path.join("specs", "2025_blend", f"{args.blend_spec}.yml")
    with open(blend_spec_path) as f:
        cs_blend = yaml.safe_load(f)
    
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
                .replace(f"diff_aifs_ens_qx", f"diff_{args.model_ens}_qx")
                .replace(f"diff_aifs_qx",     f"diff_{args.model_single}_qx")
            )
    
    blend_spec_op      = f"{args.blend_spec}_op"
    blend_spec_op_path = os.path.join("specs", "2025_blend", f"{blend_spec_op}.yml")
    with open(blend_spec_op_path, "w") as f:
        yaml.dump(cs_blend, f, default_flow_style=False, allow_unicode=True)
    _temp_spec_files.append(blend_spec_op_path)
    blend_spec = blend_spec_op
    log(f"Patched spec written: {blend_spec_op_path}")


    # -- Expected output paths ---------------------------------------------
#    aifs_pkl     = os.path.join(work_dir, f"aifs_{year}_wide.pkl")
#    aifs_ens_pkl = os.path.join(work_dir, f"aifs_ens_{year}_wide.pkl")
    #connect_pkl  = args.blend_input
    connect_pkl  = connect_output_rds
    preds_pkl    = os.path.join(work_dir, f"{args.blend_model}_global_year{year}_preds.pkl")
    export_csv   = os.path.join(work_dir, f"blend_output_summary_{date_compact}.csv")

    TOTAL = 8

    steps = [
        (1, "Build climatology", [
            sys.executable,
            "python/pipelines/prepare_data/2_build_climatology.py",
            "--spec_id", clim_spec,
        ], None),

        (2, "Process aifs nc files", [
            sys.executable,
            "python/pipelines/prepare_data/1_process_raw_nc_files.py",
            "--spec_id", aifs_spec,
        ], aifs_pkl),

        (3, "Process aifs_ens nc files", [
            sys.executable,
            "python/pipelines/prepare_data/1_process_raw_nc_files.py",
            "--spec_id", aifs_ens_spec,
        ], aifs_ens_pkl),

        (4, "Combine datasets", [
            sys.executable,
            "python/pipelines/prepare_data/3_combine_datasets.py",
            "--spec_id", combine_spec,
        ], None),

        (5, "Connect/prepare pipeline input", [
            sys.executable,
            "python/pipelines/blending_process/0_connect_prepare_data_to_2025_pipeline.py",
            #"--spec_id", args.connect_spec,
            "--spec_id", connect_spec,
            "--required_probability_prefixes",
            *required_probability_prefixes,
        ], connect_pkl),

        (6, "Apply blend model", [
            sys.executable,
            "predict/apply_blend_model.py",
            #"--spec_id",    args.blend_spec,
            "--spec_id",    blend_spec,
            "--model",      args.blend_model,
            "--year",       year,
            "--coef_dir",   args.coef_dir,
            "--coef_tag",   args.coef_tag,
            #"--input_path", args.blend_input,
            "--input_path", connect_pkl,
            "--out_dir",    work_dir,
        ], preds_pkl),

        (7, "Export blend output", [
            sys.executable,
            "predict/export_blend_output.py",
            "--issue_date", issue_date,
            #"--spec_id",    args.blend_spec,
            "--spec_id",    blend_spec,
            "--preds_file", preds_pkl,
            "--out_dir",    work_dir,
        ], export_csv),

        (8, "Generate maps", [
            sys.executable,
            "predict/run_maps.py",
            "--input_file",  export_csv,
            #"--output_path", map_out,
            "--output_path", work_dir,
            "--region",      args.region,
        ], None),
    ]

    # -- Run ---------------------------------------------------------------
    log(f"Starting operational pipeline for year={year}, issue_date={issue_date}")
    log(f"Work dir    : {work_dir}")
    #log(f"Map output  : {map_out}")
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
        log(f"Skipping steps 1-{args.skip_to - 1} (--skip_to {args.skip_to})")
    if args.dry_run:
        log("DRY RUN - commands will be printed but not executed")
    print()

    if args.stop_at is not None:
        log(f"Stopping after step {args.stop_at} (--stop_at {args.stop_at})")

    # -- Optional pre-step 0: regrid gridded rainfall onto political boundaries --
    if args.regrid_spec:
        if args.skip_to > 1:
            log(f"-- Step 0/{TOTAL}: Regrid to shapefile [SKIPPED - skip_to={args.skip_to}] --")
        else:
            run_step("0", TOTAL, "Regrid rainfall to shapefile (political boundaries)", [
                sys.executable,
                "python/pipelines/prepare_data/0_regrid_to_shapefile.py",
                "--spec_id", args.regrid_spec,
            ], None, args.dry_run)
    else:
        log("No --regrid_spec given; running on the existing grid (no regridding).")

    for step_num, name, cmd, expected_output in steps:
        if step_num < args.skip_to:
            log(f"-- Step {step_num}/{TOTAL}: {name} [SKIPPED] --")
            continue

        if args.stop_at is not None and step_num > args.stop_at:
            log(f"-- Step {step_num}/{TOTAL}: {name} [SKIPPED - stop_at={args.stop_at}] --")
            continue

        run_step(step_num, TOTAL, name, cmd, expected_output, args.dry_run)


    if not args.dry_run:
        log(f"Pipeline complete. Outputs in : {work_dir}")
        #log(f"Maps in                       : {map_out}")
    else:
        log("Dry run complete - no files were created.")


if __name__ == "__main__":
    main()
