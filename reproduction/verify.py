"""Verify the separately stored figure data and its manifest."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

# Match home directories, mounted volumes, temporary paths, and file URLs.
PRIVATE_PATH = re.compile(
    r"/(?:Users|home|Volumes|private|tmp|var/folders)/[^\s\"'<>]+"
    r"|[A-Za-z]:\\(?:Users|Documents and Settings)\\[^\s\"'<>]+"
    r"|file:" + r"//[^\s\"'<>]+"
)


def check_text(text, label):
    if PRIVATE_PATH.search(text):
        raise ValueError(f"Machine-specific reference in {label}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=os.environ.get(
            "SIMESH_FIGURE_DATA", Path(__file__).resolve().parent / "data"
        ),
        help="Figure data and manifest directory; default: data beside this file",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Refresh checksums and schemas after intentional bundle changes",
    )
    args = parser.parse_args()
    root = args.data.resolve()
    if not root.is_dir():
        parser.error("the supplied data directory does not exist")
    if args.write_manifest:
        records = {}
        for path in sorted(root.rglob("*")):
            name = path.relative_to(root)
            if (
                not path.is_file()
                or "__pycache__" in name.parts
                or name.parts[0] == "output"
                or str(name) == "manifest.json"
            ):
                continue
            if path.is_symlink():
                raise ValueError(f"Symlinks are not portable release inputs: {name}")
            content = path.read_bytes()
            record = {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as data:
                    record["arrays"] = {}
                    for key in data.files:
                        value = data[key]
                        record["arrays"][key] = {
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                        }
                        if value.dtype.kind in "US":
                            check_text(str(value.tolist()), f"{name}:{key}")
            else:
                check_text(content.decode("utf-8"), str(name))
            records[name.as_posix()] = record
        (root / "manifest.json").write_text(
            json.dumps({"format": 1, "files": records}, indent=2) + "\n"
        )
    manifest = json.loads((root / "manifest.json").read_text())
    for name, record in manifest["files"].items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(
                "Manifest entries must refer to files inside the data directory"
            )
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
        if len(content) != record["bytes"]:
            raise ValueError(f"Size mismatch: {name}")
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != set(record["arrays"]):
                    raise ValueError(f"Array inventory mismatch: {name}")
                for key in data.files:
                    value = data[key]
                    expected = record["arrays"][key]
                    if (
                        list(value.shape) != expected["shape"]
                        or str(value.dtype) != expected["dtype"]
                    ):
                        raise ValueError(f"Array schema mismatch: {name}:{key}")
                    if value.dtype.kind in "US":
                        check_text(str(value.tolist()), f"{name}:{key}")
        else:
            check_text(content.decode("utf-8"), name)
    print(
        f"Verified {len(manifest['files'])} files, array schemas, and machine-path checks."
    )


if __name__ == "__main__":
    main()
