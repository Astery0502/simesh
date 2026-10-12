"""Assemble a new plotting-data release from validated author-side products."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RETAINED = {
    3: (3, ("slice.npz",)),
    5: (4, ("profiles.npz", "magnetic-profiles.npz", "schematic-geometry.npz",
            "summary.json", "magnetic-summary.json", "schematic-summary.json")),
    6: (5, ("diagnostics.npz", "sampling.npz", "amr-slice.npz", "lines.npz",
            "selection.json")),
    7: (6, ("lines.npz", "bottom.npz", "maps.npz", "projection.npz", "provenance.json")),
}
THERMAL = ("bands.npz", "diagnostics.npz", "configuration.json",
           "colormaps.npz", "plot-settings.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-data", type=Path, required=True,
                        help="Verified 2026-09-21 plotting-data root")
    parser.add_argument("--thermal-data", type=Path, required=True,
                        help="Validated absorbed/EM-weighted thermal plotting directory")
    parser.add_argument("--performance-results", type=Path, required=True,
                        help="Recorded endpoint-mode benchmark JSON directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="New or empty plotting-data directory")
    args = parser.parse_args()
    previous, thermal, performance, output = (
        path.resolve() for path in (args.previous_data, args.thermal_data,
                                    args.performance_results, args.output)
    )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("the output must be a new or empty directory")
    if any(output == source or source in output.parents
           for source in (previous, thermal, performance)):
        parser.error("the output must be outside the source directories")
    subprocess.run([sys.executable, str(ROOT / "verify.py"), "--data", str(previous)],
                   check=True)
    config = json.loads((thermal / "configuration.json").read_text())
    if not (config.get("absorption")
            and config.get("temperature_projection") == "emission-measure-weighted"
            and config.get("temperature_distribution") == "volume-emission-measure"):
        parser.error("thermal products must include absorption and both EM diagnostics")
    files = [(previous / f"figure{old}" / name, Path(f"figure{new}") / name)
             for new, (old, names) in RETAINED.items() for name in names]
    files += [(thermal / name, Path("figure8") / name) for name in THERMAL]
    for rep in ("amr", "l5", "l6"):
        for workers in (1, 4, 8, 16, 32):
            name = f"{rep}-endpoints-w{workers}.json"
            source = performance / name
            record = json.loads(source.read_text())
            if (record["representation"], record["workers"], record["mode"]) != (
                rep, workers, "endpoints"
            ):
                parser.error(f"unexpected performance record: {name}")
            files.append((source, Path("figure4/results") / name))
    files += [(previous / name, Path(name)) for name in ("LICENSE", "LICENSE-SunPy.txt")]
    files += [(ROOT / "DATA.md", Path("DATA.md"))]
    for source, _ in files:
        if not source.is_file():
            parser.error(f"missing release input: {source}")
    output.mkdir(parents=True, exist_ok=True)
    for source, relative in files:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    release = {
        "release": "2026-10-12",
        "manuscript_figures": 8,
        "numbering": "Current manuscript Figures 1–8",
        "previous_manifest_sha256": hashlib.sha256(
            (previous / "manifest.json").read_bytes()).hexdigest(),
        "scope": "Redraw retained scientific products; no simulation snapshots",
    }
    (output / "release.json").write_text(json.dumps(release, indent=2) + "\n")
    subprocess.run([sys.executable, str(ROOT / "verify.py"), "--data", str(output),
                    "--write-manifest"], check=True)
    print(f"Prepared plotting inputs: {output}")


if __name__ == "__main__":
    main()
