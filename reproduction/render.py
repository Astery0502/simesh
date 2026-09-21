"""Render manuscript figures with separately supplied numerical inputs."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def figure_number(value):
    number = int(value)
    if number not in range(1, 8):
        raise argparse.ArgumentTypeError("figure number must be between 1 and 7")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=os.environ.get("SIMESH_FIGURE_DATA", ROOT / "data"),
        help="Plotting input directory; default: data beside this file",
    )
    parser.add_argument(
        "figures",
        nargs="*",
        type=figure_number,
        metavar="N",
        help="Figure numbers (1–7); default: all",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output",
        help="Rendering destination; default: output beside this file",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    figures = list(dict.fromkeys(args.figures or range(1, 8)))
    inputs = args.data.resolve()
    if any(number >= 3 for number in figures):
        for number in figures:
            if number >= 3 and not (inputs / f"figure{number}").is_dir():
                parser.error(
                    f"missing figure{number}/: place the separately downloaded inputs "
                    "in reproduction/data/ or specify --data DIR"
                )
    if output == inputs or inputs in output.parents:
        parser.error("the output directory must be outside the data directory")
    environment = dict(os.environ, SIMESH_FIGURE_OUTPUT=str(output), MPLBACKEND="Agg")
    environment["SIMESH_FIGURE_DATA"] = str(inputs)
    for number in figures:
        print(f"Rendering Figure {number}", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / f"render_figure{number}.py")],
            env=environment,
            check=True,
        )
    print("Rendering complete.")


if __name__ == "__main__":
    main()
