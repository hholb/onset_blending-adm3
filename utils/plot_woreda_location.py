#!/usr/bin/env python3
"""
Plot adm2 and adm3 boundaries on a map and highlight one or more woredas.

Usage:
  # Single woreda (no quotes needed unless the name has spaces)
  python plot_woreda_location.py \
      --woreda Adaba \
      --shapefile_dir /path/to/shapefiles \
      --output_dir /path/to/output

  # Multiple woredas
  python plot_woreda_location.py \
      --woreda Adaba Goro Sinana \
      --shapefile_dir /path/to/shapefiles \
      --output_dir /path/to/output

  # Name with spaces — quote only that token
  python plot_woreda_location.py \
      --woreda Adaba "Kore Woreda" Sinana \
      --shapefile_dir /path/to/shapefiles \
      --output_dir /path/to/output

  # Override individual shapefiles if needed:
  python plot_woreda_location.py \
      --woreda Adaba \
      --shapefile_dir /path/to/shapefiles \
      --output_dir /path/to/output \
      --adm3_shp /other/path/manual_zones_woredas.shp \
      --country_shp /other/path/Country_Boundary.shp
"""

import argparse
import sys
from pathlib import Path

import matplotlib
#matplotlib.use("Agg")
matplotlib.use("TkAgg")  # or "QtAgg"
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import geopandas as gpd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Loaders  (mirrors maps_new_zone.py conventions)
# ---------------------------------------------------------------------------

def load_country_boundary(shapefile_dir: Path, country_shp: Path | None) -> gpd.GeoDataFrame:
    path = country_shp or (shapefile_dir / "Country_Boundary.shp")
    if not path.is_file():
        logging.warning(f"Country boundary not found at {path}; skipping.")
        return None
    return gpd.read_file(path).to_crs("EPSG:4326")


def load_adm3(shapefile_dir: Path, adm3_shp: Path | None) -> gpd.GeoDataFrame:
    path = adm3_shp or (shapefile_dir / "manual_zones_woredas.shp")
    if not path.is_file():
        sys.exit(f"ERROR: adm3 shapefile not found: {path}")
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    if "adm3_name" not in gdf.columns:
        sys.exit(
            f"ERROR: adm3 shapefile has no 'adm3_name' column. "
            f"Found: {gdf.columns.tolist()}"
        )
    return gdf


def load_adm2(adm3_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve adm3 polygons to adm2 boundaries (same approach as maps_new_zone.py)."""
    if "adm2_name" not in adm3_gdf.columns:
        sys.exit(
            f"ERROR: shapefile has no 'adm2_name' column. "
            f"Found: {adm3_gdf.columns.tolist()}"
        )
    return adm3_gdf.dissolve(by="adm2_name", as_index=False)[["adm2_name", "geometry"]]


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot_woreda(woreda_names: list[str],          # ← was: woreda_name: str
                shapefile_dir: Path,
                output_dir: Path,
                adm3_shp: Path | None,
                country_shp: Path | None) -> None:

    # ── Load geometries ───────────────────────────────────────────────────
    adm3_gdf    = load_adm3(shapefile_dir, adm3_shp)
    adm2_gdf    = load_adm2(adm3_gdf)
    country_gdf = load_country_boundary(shapefile_dir, country_shp)

    # Validate every requested woreda                ← was: single check
    for wn in woreda_names:
        if wn not in adm3_gdf["adm3_name"].values:
            close = [n for n in adm3_gdf["adm3_name"] if wn.lower() in n.lower()]
            hint = f"  Did you mean: {close[:5]}" if close else ""
            sys.exit(
                f"ERROR: '{wn}' not found in adm3_name column.{hint}\n"
                f"  Total woredas: {len(adm3_gdf)}"
            )

    target = adm3_gdf[ adm3_gdf["adm3_name"].isin(woreda_names)]   # ← was: == woreda_name
    other  = adm3_gdf[~adm3_gdf["adm3_name"].isin(woreda_names)]   # ← was: != woreda_name

    # ── Map extent (country bounds if available, else adm3 total bounds) ─
    if country_gdf is not None:
        minx, miny, maxx, maxy = country_gdf.total_bounds
    else:
        minx, miny, maxx, maxy = adm3_gdf.total_bounds

    # ── Colors  (from maps_new_zone.py) ───────────────────────────────────
    # adm3 fill colours
    color_none    = '#d3d3d3'   # unselected woredas
    color_target  = '#e05c2a'   # highlighted woreda (warm orange-red, stands out)

    # adm2 boundary style  (mirrors maps_new_zone.py)
    adm2_edge_color  = 'dimgray'
    adm2_edge_lw     = 0.8
    adm2_label_color = 'darkgray'
    adm2_label_size  = 5

    # adm3 boundary style
    adm3_edge_color = 'white'
    adm3_edge_lw    = 0.15

    # country boundary style
    country_edge_color = 'black'
    country_edge_lw    = 0.8

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor('#f5f5f5')

    # 1. All adm3 polygons (grey background)
    other.plot(ax=ax, color=color_none,
               edgecolor=adm3_edge_color, linewidth=adm3_edge_lw, zorder=1)

    # 2. Highlighted woredas
    target.plot(ax=ax, color=color_target,
                edgecolor='white', linewidth=0.5, zorder=2)

    # 3. Country boundary
    if country_gdf is not None:
        country_gdf.boundary.plot(ax=ax, linewidth=country_edge_lw,
                                  edgecolor=country_edge_color, zorder=5)

    # 4. ADM2 boundaries overlay  (mirrors maps_new_zone.py)
    adm2_gdf.boundary.plot(ax=ax, linewidth=adm2_edge_lw,
                           edgecolor=adm2_edge_color, linestyle='-', zorder=3)

    # 5. ADM2 name labels  (mirrors maps_new_zone.py)
    for _, row in adm2_gdf.iterrows():
        cx = row.geometry.centroid.x
        cy = row.geometry.centroid.y
        ax.text(cx, cy, row["adm2_name"], fontsize=adm2_label_size,
                color=adm2_label_color, ha='center', va='center',
                zorder=4, clip_on=True)

    # 6. Woreda name label on each highlighted polygon   ← was: single label
    for _, trow in target.iterrows():
        tc = trow.geometry.centroid
        ax.text(tc.x, tc.y, trow["adm3_name"],
                fontsize=6, fontweight='bold', color='white',
                ha='center', va='center', zorder=6, clip_on=True,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color_target,
                          edgecolor='white', linewidth=0.5, alpha=0.85))

    # ── Axes styling  (mirrors maps_new_zone.py) ──────────────────────────
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.5)

    # Title — singular vs plural                        ← was: f"Woreda: {woreda_name}"
    title_str = ", ".join(woreda_names)
    ax.set_title(
        f"Woreda{'s' if len(woreda_names) > 1 else ''}: {title_str}",
        fontsize=12, fontweight='bold', pad=8
    )

    # ── Legend ────────────────────────────────────────────────────────────
    label_str = ", ".join(woreda_names)                # ← was: woreda_name
    handles = [
        Patch(facecolor=color_target,  edgecolor='white', label=label_str),
        Patch(facecolor=color_none,    edgecolor='white', label='Other woredas (adm3)'),
        Patch(facecolor='none', edgecolor=adm2_edge_color,
              linewidth=1.2, label='adm2 boundary'),
    ]
    ax.legend(handles=handles, loc='lower left', fontsize=8,
              handlelength=1.5, handleheight=1.5,
              borderpad=0.5, labelspacing=0.3, framealpha=0.8)

    # ── Save ──────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    # Join all names with "__" for the filename         ← was: single woreda_name
    safe = ("__".join(woreda_names)
            .replace(" ", "_").replace("/", "_")
            .replace("'", "").replace("(", "").replace(")", ""))
    out_path = output_dir / f"woreda_location_{safe}.png"
    plt.tight_layout()
    #plt.show(block=False)
    plt.show()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved: {out_path}")
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot adm2/adm3 boundaries and highlight one or more woredas."
    )
    parser.add_argument(
        "--woreda", required=True, nargs="+",     # ← added nargs="+"
        help=(
            "One or more adm3_name values to highlight (case-sensitive). "
            "Space-separated. Quote names that contain spaces, e.g.: "
            "--woreda Adaba Goro \"Kore Woreda\""
        ),
    )
    parser.add_argument(
        "--shapefile_dir", required=True,
        help="Directory containing manual_zones_woredas.shp (and optionally "
             "Country_Boundary.shp).",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory where the output PNG will be saved.",
    )
    parser.add_argument(
        "--adm3_shp", default=None,
        help="(Optional) Explicit path to the adm3 shapefile; "
             "overrides --shapefile_dir / manual_zones_woredas.shp.",
    )
    parser.add_argument(
        "--country_shp", default=None,
        help="(Optional) Explicit path to the country boundary shapefile; "
             "overrides --shapefile_dir / Country_Boundary.shp.",
    )
    args = parser.parse_args()

    plot_woreda(
        woreda_names  = args.woreda,              # ← was: woreda_name = args.woreda
        shapefile_dir = Path(args.shapefile_dir),
        output_dir    = Path(args.output_dir),
        adm3_shp      = Path(args.adm3_shp)    if args.adm3_shp    else None,
        country_shp   = Path(args.country_shp) if args.country_shp else None,
    )


if __name__ == "__main__":
    main()
