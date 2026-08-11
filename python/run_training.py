"""
run_training.py
===============
Runs the full blending training pipeline in a single command. Each step is
executed sequentially; the script verifies that expected output files exist
before proceeding to the next step.

Steps
-----
1. Process the ground-truth rainfall specification.
2. Process one raw rainfall specification per configured forecast.
3. Build climatology.
4. Combine datasets.
5. Connect/prepare the blend input.
6. Evaluate the blend.

The script patches the relevant YAML fields before running each stage by
writing temporary specifications (suffixed _op) into the existing spec
directories. These are cleaned up on exit. With two forecasts, the historical
seven-step numbering is preserved.

Usage (run from repo root)
--------------------------
    python python/run_training.py \\
        --forecast        aifs aifs_2026 \\
        --forecast        ngcm ngcm_2026 \\
        --clim_spec       ref_rain_fixed_cutoff_2026 \\
        --combine_spec    combine_template_fixed_cutoff_2026_ngcm \\
        --connect_spec    connect_fixed_cutoff_2026_ngcm \\
        --blend_spec      cv_models_fixed_cutoff_2026_ngcm \\
        --work_dir        Monsoon_Data/Processed_Data/training \\
        --results_dir     Monsoon_Data/results/dry_spell_aifs_ngcm \\
        [--gt_path            Monsoon_Data/Processed_Data/Models/dry_spell/imd_clim_mok_date_wide.pkl] \\
        [--blend_input        Monsoon_Data/Processed_Data/training/cv_data_fixed_cutoff_new_pipeline.pkl] \\
        [--skip_to STEP] \\
        [--stop_at STEP] \\
        [--dry_run]

Notes
-----
--forecast MODEL SPEC_ID
    Repeat once per forecast source. MODEL is the name already used by the
    selected combine, connect, and blend specs. SPEC_ID identifies the raw-data
    YAML under specs/raw_data. The existing model_single/model_ens and
    aifs_spec/aifs_ens_spec arguments remain available as a legacy alternative.

--aifs_nc_file / --aifs_ens_nc_file
    Legacy-interface overrides. Point to a specific NetCDF file and override
    both input.nc_folder and input.file_regex. These cannot be combined with
    --forecast.

--aifs_nc_folder / --aifs_ens_nc_folder
    Legacy-interface input.nc_folder overrides. These cannot be combined with
    --forecast.

--gt_path
    Path to the historical ground truth wide pkl. Simultaneously overrides:
      - input.gt_path            in the clim spec
      - ground_truth_wide_rds    in the combine spec
    Both fields must point to the same file, so a single arg controls both.

--work_dir
    Working directory for all intermediate files (processed nc outputs,
    climatology pkls, combined wide pkl, connect output pkl). Patched into all
    raw forecast and climatology specs, the combine spec, and the connect
    input/output handoff.

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
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import yaml
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONNECTOR_ARTIFACT = "cv_data_fixed_cutoff_new_pipeline.pkl"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from python.rain_horizon_utils import resolve_rain_day_max  # noqa: E402

os.chdir(REPO_ROOT)

# ---------------------------------------------------------------------------
# Temp spec management
# ---------------------------------------------------------------------------

_temp_spec_files = []   # paths to clean up on exit


@dataclass
class ForecastJob:
    model: str
    spec_id: str
    template_name: Optional[str] = None
    nc_file: Optional[str] = None
    nc_folder: Optional[str] = None
    runtime_spec_id: Optional[str] = None
    artifact_path: Optional[str] = None
    rain_day_max: Optional[int] = None
    rain_horizon_policy: str = "legacy"


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


def load_yaml_spec(spec_id, spec_type):
    """Load one maintained YAML specification and require a mapping."""
    path = os.path.join("specs", spec_type, f"{spec_id}.yml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Base spec not found: {path}")
    with open(path) as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise ValueError(f"Spec must contain a YAML mapping: {path}")
    return spec


def write_temp_spec(base_spec_id, spec_type, spec):
    """Write an already-resolved specification using the existing _op convention."""
    new_spec_id = f"{base_spec_id}_op"
    path = os.path.join("specs", spec_type, f"{new_spec_id}.yml")
    with open(path, "w") as f:
        yaml.dump(spec, f, default_flow_style=False, allow_unicode=True)
    _temp_spec_files.append(path)
    log(f"Patched spec written: {path}")
    return new_spec_id


def normalize_forecast_jobs(args):
    """Normalize the new repeated interface or the complete legacy interface."""
    forecast_pairs = args.forecast or []
    legacy_names = [
        "model_single", "model_ens", "aifs_spec", "aifs_ens_spec",
        "aifs_nc_file", "aifs_ens_nc_file",
        "aifs_nc_folder", "aifs_ens_nc_folder",
    ]
    supplied_legacy = [name for name in legacy_names if getattr(args, name)]

    if forecast_pairs and supplied_legacy:
        raise ValueError(
            "--forecast cannot be combined with legacy forecast arguments."
        )

    if forecast_pairs:
        jobs = [
            ForecastJob(model=model, spec_id=spec_id)
            for model, spec_id in forecast_pairs
        ]
    else:
        required = [
            args.model_single, args.model_ens,
            args.aifs_spec, args.aifs_ens_spec,
        ]
        if not all(required):
            raise ValueError(
                "Provide --forecast MODEL SPEC_ID at least once, or provide the "
                "complete legacy model/spec argument set."
            )
        jobs = [
            ForecastJob(
                model=args.model_single,
                spec_id=args.aifs_spec,
                template_name="aifs",
                nc_file=args.aifs_nc_file,
                nc_folder=args.aifs_nc_folder,
            ),
            ForecastJob(
                model=args.model_ens,
                spec_id=args.aifs_ens_spec,
                template_name="aifs_ens",
                nc_file=args.aifs_ens_nc_file,
                nc_folder=args.aifs_ens_nc_folder,
            ),
        ]

    model_names = [job.model for job in jobs]
    if len(model_names) != len(set(model_names)):
        raise ValueError("Forecast model names must be unique.")

    return jobs


def resolve_model_entries(entries, jobs, context):
    """Apply legacy role aliases and return entries for the requested models."""
    names = [entry.get("name") for entry in entries]
    duplicates = sorted({
        name for name in names if name is not None and names.count(name) > 1
    })
    if duplicates:
        raise ValueError(
            f"{context} contains duplicate names: " + ", ".join(duplicates)
        )

    by_name = {entry["name"]: entry for entry in entries}
    for job in jobs:
        if job.model in by_name:
            continue
        if job.template_name and job.template_name in by_name:
            entry = by_name.pop(job.template_name)
            entry["name"] = job.model
            by_name[job.model] = entry

    missing = [job.model for job in jobs if job.model not in by_name]
    if missing:
        raise ValueError(
            f"{context} has no entry for: " + ", ".join(missing)
        )

    return by_name


def reconcile_forecast_horizon(
    raw_spec, connect_entry, model_name, probability_day_max
):
    """Resolve one explicit rainfall-horizon contract into raw and connect specs."""
    raw_options = raw_spec.setdefault("options", {})

    def explicit_contract(config):
        if config.get("rain_day_max") is None:
            return None
        return resolve_rain_day_max(config, probability_day_max)

    raw_contract = explicit_contract(raw_options)
    connect_contract = explicit_contract(connect_entry)

    if raw_contract and connect_contract and raw_contract != connect_contract:
        raise ValueError(
            f"Rainfall-horizon mismatch for forecast '{model_name}': raw spec "
            f"resolves to {raw_contract}, connect spec resolves to "
            f"{connect_contract}."
        )

    resolved = raw_contract or connect_contract
    if resolved is None:
        return None, "legacy"

    rain_day_max, policy = resolved
    raw_options["rain_day_max"] = rain_day_max
    raw_options["rain_horizon_policy"] = policy
    connect_entry["rain_day_max"] = rain_day_max
    connect_entry["rain_horizon_policy"] = policy
    return rain_day_max, policy


def configure_blend_spec_models(spec, jobs):
    """Apply legacy two-slot aliases inside the selected blend spec."""
    alias_map = {
        job.template_name: job.model
        for job in jobs
        if job.template_name and job.template_name != job.model
    }

    for entries in (
        spec.get("extras", {}).get("forecasts", []),
        spec.get("mme", {}).get("blend_models", []),
    ):
        for entry in entries:
            name = entry.get("name")
            if name in alias_map:
                entry["name"] = alias_map[name]

    formulas = spec.get("models", {}).get("formulas", {}).values()
    for formula in formulas:
        if not formula.get("enabled"):
            continue
        text = formula["text"]
        for template_name, model_name in sorted(
            alias_map.items(), key=lambda item: len(item[0]), reverse=True
        ):
            text = text.replace(f"_{template_name}_", f"_{model_name}_")
        formula["text"] = text


def required_probability_prefixes_from_blend_spec(blend_spec, connect_spec):
    """Return forecast probability series consumed by this training run."""
    configured = {
        f"{fm['name']}_p_onset"
        + ("" if variant is None else f"_{variant}")
        for fm in connect_spec.get("forecast_models", [])
        for variant in [None] + list(fm.get("variants") or [])
    }
    extras = blend_spec.get("extras") or {}
    variant_suffixes = extras.get("forecast_variants") or {
        "base": "",
        "ref_onset": "_ref",
        "fixed_cutoff": "_fixed_cutoff",
    }

    def probability_prefix(name, variant):
        if variant not in variant_suffixes:
            raise ValueError(f"Unknown forecast probability variant: {variant}")
        return f"{name}_p_onset{variant_suffixes[variant]}"

    required = set()
    enabled_formulas = [
        formula["text"]
        for formula in blend_spec.get("models", {}).get("formulas", {}).values()
        if formula.get("enabled")
    ]
    for fm in connect_spec.get("forecast_models", []):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_])"
            rf"({re.escape(fm['name'])}_p_onset(?:_[A-Za-z0-9_]+)?)"
            rf"_(?:qx|week\d+)"
        )
        for formula in enabled_formulas:
            required.update(pattern.findall(formula))

    for forecast in extras.get("forecasts") or []:
        if forecast.get("raw", False) or forecast.get("calibrated", False):
            required.add(
                probability_prefix(
                    forecast["name"], forecast.get("variant", "base")
                )
            )

    mme = blend_spec.get("mme") or {}
    if mme.get("enabled", False):
        for model in mme.get("blend_models") or []:
            if model.get("source") == "forecast":
                required.add(
                    probability_prefix(
                        model["name"], model.get("cal_variant", "base")
                    )
                )

    unknown = sorted(required - configured)
    if unknown:
        raise ValueError(
            "Blend spec requires forecast probability series not declared by "
            "connect forecast_models: " + ", ".join(unknown)
        )
    return sorted(required)


def validate_blend_spec_models(spec, jobs):
    """Require every requested forecast model to appear in the blend spec."""
    requested = [job.model for job in jobs]
    declared = {
        entry.get("name")
        for entries in (
            spec.get("extras", {}).get("forecasts", []),
            spec.get("mme", {}).get("blend_models", []),
        )
        for entry in entries
    }

    formula_models = set()
    formulas = spec.get("models", {}).get("formulas", {}).values()
    for formula in formulas:
        if not formula.get("enabled"):
            continue
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula["text"])
        for token in tokens:
            for model in sorted(requested, key=len, reverse=True):
                if re.search(rf"(?:^|_){re.escape(model)}(?:_|$)", token):
                    formula_models.add(model)
                    break

    missing = [
        model for model in requested
        if model not in declared and model not in formula_models
    ]
    if missing:
        raise ValueError(
            "blend spec has no enabled formula, extras forecast, or MME entry "
            "for: " + ", ".join(missing)
        )


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
    elif template_name and template_name in forecasts:
        forecasts[model_name] = forecasts.pop(template_name)
        selected_name = model_name
    else:
        available = ", ".join(sorted(forecasts)) or "<none>"
        if template_name:
            expected = (
                f"model '{model_name}' or legacy template role '{template_name}'"
            )
        else:
            expected = f"model '{model_name}'"
        raise KeyError(
            f"Combine spec has no forecast entry for {expected}. "
            f"Available keys: {available}"
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
    """Build the final evaluation command with an explicit artifact handoff."""
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

    # ── Forecast sources ─────────────────────────────────────────────────
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

    # Legacy two-slot interface. These remain supported but are mutually
    # exclusive with --forecast.
    parser.add_argument("--model_single",   default=None,
                        help="Legacy deterministic forecast model name.")
    parser.add_argument("--model_ens",      default=None,
                        help="Legacy ensemble/second forecast model name.")
    parser.add_argument("--aifs_spec",      default=None,
                        help="Legacy raw-data spec for --model_single.")
    parser.add_argument("--aifs_ens_spec",  default=None,
                        help="Legacy raw-data spec for --model_ens.")

    # ── Required pipeline specs and paths ────────────────────────────────
    parser.add_argument("--clim_spec",      required=True,
                        help="Raw-data/climatology spec ID.")
    parser.add_argument("--combine_spec",   required=True,
                        help="Combine-stage spec ID.")
    parser.add_argument("--connect_spec",   required=True,
                        help="Connect-stage spec ID.")
    parser.add_argument("--blend_spec",     required=True,
                        help="Blend-evaluation spec ID.")
    parser.add_argument("--work_dir",       required=True,
                        help="Working directory for all intermediate files")
    parser.add_argument("--results_dir",    required=True,
                        help="Output directory for blend evaluation results "
                             "(coefs, cv_preds, summary csvs). Overrides "
                             "run.pipeline_output_dir in the blend spec.")

    # ── yml field overrides ───────────────────────────────────────────────
    parser.add_argument("--aifs_nc_file",   default=None,
                        help="Legacy first-slot NetCDF-file override. Takes "
                             "priority over --aifs_nc_folder.")
    parser.add_argument("--aifs_ens_nc_file", default=None,
                        help="Legacy second-slot NetCDF-file override. Takes "
                             "priority over --aifs_ens_nc_folder.")
    parser.add_argument("--aifs_nc_folder", default=None,
                        help="Legacy first-slot input.nc_folder override.")
    parser.add_argument("--aifs_ens_nc_folder", default=None,
                        help="Legacy second-slot input.nc_folder override.")
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

    try:
        forecast_jobs = normalize_forecast_jobs(args)
    except ValueError as exc:
        parser.error(str(exc))

    # ── Derived paths ─────────────────────────────────────────────────────
    work_dir = args.work_dir

    os.makedirs(work_dir,      exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Resolve connect model entries and rainfall-horizon contracts ──────
    cs_connect = load_yaml_spec(args.connect_spec, "2025_blend")
    connect_entries = resolve_model_entries(
        cs_connect.get("forecast_models"),
        forecast_jobs,
        "connect forecast_models",
    )
    probability_day_max = cs_connect.get("day_max")

    # ── Patch one raw-data spec per forecast source ───────────────────────
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

    # ── Patch clim spec (also used for step 1: process IMD nc files) ─────
    clim_patches = [
        ("output.out_dir",             work_dir),
        ("paths.climatology_out_dir",  work_dir),
        ("output.basename",            args.clim_spec),
        (
            "input.gt_path",
            args.gt_path
            or os.path.join(work_dir, f"{args.clim_spec}_wide.pkl"),
        ),
    ]
    clim_spec = write_patched_spec(args.clim_spec, "raw_data", clim_patches)

    # ── Patch combine spec ────────────────────────────────────────────────
    # Read clim spec for climatology output filenames
    clim_spec_raw = load_yaml_spec(args.clim_spec, "raw_data")
    clim_rds = os.path.join(
        work_dir,
        clim_spec_raw["climatologies"]["clim"]["out_stem"] + ".pkl",
    )
    clim_unc_rds = os.path.join(
        work_dir,
        clim_spec_raw["climatologies"]["clim_unc"]["out_stem"] + ".pkl",
    )

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

    # ── Patch connect spec ────────────────────────────────────────────────
    combine_basename   = f"{args.combine_spec}_combined_wide.pkl"
    connect_input_rds  = os.path.join(work_dir, combine_basename)
    connect_output_rds = resolve_connector_artifact(work_dir, args.blend_input)

    cs_connect["input_rds"]  = connect_input_rds
    cs_connect["output_rds"] = connect_output_rds

    connect_spec = write_temp_spec(args.connect_spec, "2025_blend", cs_connect)

    # ── Patch blend spec ──────────────────────────────────────────────────
    cs_blend = load_yaml_spec(args.blend_spec, "2025_blend")
    configure_blend_spec_models(cs_blend, forecast_jobs)
    validate_blend_spec_models(cs_blend, forecast_jobs)
    required_probability_prefixes = required_probability_prefixes_from_blend_spec(
        cs_blend, cs_connect
    )
    blend_spec = write_temp_spec(args.blend_spec, "2025_blend", cs_blend)

    # ── Expected output paths ─────────────────────────────────────────────
    ref_rain_pkl     = os.path.join(work_dir, f"{args.clim_spec}_wide.pkl")
    connect_pkl = connect_output_rds

    # ── Build steps list ──────────────────────────────────────────────────
    blend_eval_cmd = build_blend_eval_cmd(
        blend_spec,
        work_dir,
        connect_output_rds,
        args.results_dir,
        cores=args.cores,
    )

    forecast_steps = []
    for step_num, job in enumerate(forecast_jobs, start=2):
        runtime_spec_id = job.runtime_spec_id
        artifact_path = job.artifact_path
        if runtime_spec_id is None or artifact_path is None:
            raise RuntimeError(
                f"Forecast job {job.model!r} has not been fully configured."
            )
        forecast_steps.append((
            step_num,
            f"Process {job.model} nc files",
            [
                sys.executable,
                "python/pipelines/prepare_data/1_process_raw_nc_files.py",
                "--spec_id", runtime_spec_id,
            ],
            artifact_path,
        ))

    climatology_step = len(forecast_jobs) + 2
    combine_step = climatology_step + 1
    connect_step = combine_step + 1
    blend_step = connect_step + 1
    total_steps = blend_step

    steps = [
        (1, "Process IMD ground truth nc files", [
            sys.executable,
            "python/pipelines/prepare_data/1_process_raw_nc_files.py",
            "--spec_id", clim_spec,
        ], ref_rain_pkl),

        *forecast_steps,

        (climatology_step, "Build climatology", [
            sys.executable,
            "python/pipelines/prepare_data/2_build_climatology.py",
            "--spec_id", clim_spec,
        ], clim_rds),

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

        (blend_step, "Blend evaluation", blend_eval_cmd, None),
    ]

    # ── Run ───────────────────────────────────────────────────────────────
    log(f"Starting training pipeline")
    log(f"Work dir    : {work_dir}")
    log(f"Results dir : {args.results_dir}")
    log(f"Connector → evaluator artifact: {connect_output_rds}")
    log(
        "Required forecast probability series: "
        + (", ".join(required_probability_prefixes) or "none (rainfall-only)")
    )
    log("Forecast sources:")
    for job in forecast_jobs:
        if job.rain_day_max is None:
            horizon = "legacy"
        else:
            horizon = (
                f"{job.rain_day_max} days ({job.rain_horizon_policy})"
            )
        log(
            f"  {job.model}: spec={job.spec_id}, artifact={job.artifact_path}, "
            f"rain_horizon={horizon}"
        )
        if job.nc_file:
            log(f"    nc_file override   : {job.nc_file}")
        elif job.nc_folder:
            log(f"    nc_folder override : {job.nc_folder}")
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
            log(f"── Step {step_num}/{total_steps}: {name} [SKIPPED] ──")
            continue

        if args.stop_at is not None and step_num > args.stop_at:
            log(
                f"── Step {step_num}/{total_steps}: {name} "
                f"[SKIPPED — stop_at={args.stop_at}] ──"
            )
            continue

        run_step(step_num, total_steps, name, cmd, expected_output, args.dry_run)

    if not args.dry_run:
        log(f"Pipeline complete.")
        log(f"Intermediate files : {work_dir}")
        log(f"Training outputs   : {args.results_dir}")
    else:
        log("Dry run complete — no files were created.")


if __name__ == "__main__":
    main()
