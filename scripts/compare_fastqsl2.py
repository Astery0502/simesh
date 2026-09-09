"""Compare independent native results with an external FastQSL2 checkout.

The reference source/executable is never bundled or imported by simesh.
Run this script with the project's interpreter; --reference-python selects
an environment containing NumPy and Matplotlib for the upstream wrapper.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

import numpy as np


def reference(args):
    spec = importlib.util.spec_from_file_location("fastqsl_reference", args.reference/"fastqsl.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = subprocess.run
    original_directory = Path.cwd()
    try:
        with np.load(args.input) as data, patch.object(module.subprocess, "run",
                side_effect=lambda *a, **k: run([str(args.reference/"fastqsl.x")], check=True)):
            result = module.fastqsl(
                data["values"], xa=data["x"], ya=data["y"], za=data["z"],
                seed=data["seeds"], scottFlag=True, RK4Flag=True, step=.125,
                maxsteps=20000, nthreads=1, silent=True, length_out=True,
                twist_out=True, rF_out=True, tmp_dir=str(args.input.parent/"upstream-work"),
            )
            np.savez(args.output, **{name:result[name] for name in
                                    ("q","q_perp","twist","length","rFs","rFe")})
    finally:
        os.chdir(original_directory)


def compare(args):
    import simesh as sm
    cells = args.cells
    mesh = sm.mesh_from_forest((1,1,1), np.array([True]), lower=(-1,-1,0), upper=(1,1,1),
                              block_shape=(cells,cells,cells))
    axes = [lo+(np.arange(cells)+.5)*(hi-lo)/cells for lo,hi in zip(mesh.lower,mesh.upper)]
    x,y,z = np.meshgrid(*axes, indexing="ij")
    cases = {
        "hyperbolic": np.array([.5*x,-.5*y,np.ones_like(x)]),
        "helical": np.array([-.5*y,.5*x,np.ones_like(x)]),
    }
    charges = np.zeros_like(cases["helical"])
    for position, strength in zip((-1.5,-.5,.5,1.5), (1,-1,1,-1)):
        delta = np.array([x-position,y,z+.5])
        charges += strength*delta/np.sum(delta*delta,axis=0)**1.5
    cases["quadrupole"] = charges
    seeds = np.array([[a,.2,.3] for a in np.linspace(-.7,.7,15)])
    summaries = {}
    with tempfile.TemporaryDirectory(prefix="simesh-qsl-compare-") as directory:
        folder = Path(directory)
        for name, values in cases.items():
            input_path, output_path = folder/f"{name}-input.npz", folder/f"{name}-reference.npz"
            np.savez(input_path, values=values.transpose(3,2,1,0),
                     x=axes[0],y=axes[1],z=axes[2],seeds=seeds)
            subprocess.run([args.reference_python, str(Path(__file__).resolve()),
                            "--run-reference", "--reference", str(args.reference),
                            "--input",str(input_path),"--output",str(output_path)], check=True)
            with sm.source_from_arrays(mesh, values[None], ("b1","b2","b3")) as source:
                fields = sm.prepare(source, scheme="exact-phase")
            ours = sm.qsl(fields, seeds, bounds=([a[0] for a in axes],[a[-1] for a in axes]),
                          normalization="flux", method=args.method, step_fraction=.125, max_steps=20000)
            with np.load(output_path) as upstream:
                metrics = {"valid_native":int(ours.valid.sum()), "seeds":len(seeds)}
                for quantity in ("q","q_perp","twist","length"):
                    old = upstream[quantity].reshape(-1)
                    new = getattr(ours,quantity)
                    valid = np.isfinite(old) & np.isfinite(new)
                    metrics[quantity] = {
                        "compared":int(valid.sum()),
                        "max_absolute_error":float(np.max(np.abs(old[valid]-new[valid]))) if valid.any() else None,
                        "max_relative_error":float(np.max(np.abs(old[valid]-new[valid])/np.maximum(np.abs(old[valid]),1e-8))) if valid.any() else None,
                    }
                summaries[name] = metrics
    revision = subprocess.check_output(["git","-C",str(args.reference),"rev-parse","HEAD"], text=True).strip()
    report = {"reference":"https://github.com/el2718/FastQSL2", "revision":revision,
              "cells":cells, "native_scheme":"exact-phase", "native_method":args.method,
              "reference_method":"variational", "normalization":"flux", "cases":summaries}
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(text+"\n")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-python", default="python3")
    parser.add_argument("--cells", type=int, default=32)
    parser.add_argument("--method", choices=("finite-difference","variational"), default="finite-difference")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-reference", action="store_true")
    args = parser.parse_args()
    args.reference = args.reference.resolve()
    reference(args) if args.run_reference else compare(args)
