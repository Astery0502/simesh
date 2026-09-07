"""Complete consumer footprint and optimistic storage-saving bounds."""

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import lfe_001 as lfe
import chs_001 as chs
from m1_cache_scaling import array_inventory_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lfe", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local = json.loads(args.lfe.read_text())
    cache = json.loads(args.cache.read_text())
    records = []
    with tempfile.TemporaryDirectory(prefix="simesh-m1-resources-") as directory:
        synthetic = lfe.make_fixture()
        path = Path(directory) / "local.dat"
        lfe.write_fixture_dat(path, synthetic)
        for name, source, regions in (("synthetic", path, local["regions"]),
                ("tdm", Path(local["real_dat"]["path"]), local["real_dat"]["regions"])):
            fd = os.open(source, os.O_RDONLY)
            try:
                index = lfe.read_amrvac_v5_index(fd)
                binding = lfe.bind_amrvac_v5_forest(index)
                reader = lfe.make_amrvac_v5_block_reader(fd, index, binding)
                fixture = synthetic if name == "synthetic" else chs.case_from_v5(
                    index, binding, np.asarray(local["real_dat"]["magnetic_field_ids"], dtype=np.int64), name)
                metadata = array_inventory_bytes(index, binding, reader,
                    tuple(getattr(fixture, field.name) for field in fields(fixture)
                          if field.name != "backing"))
                block = index.block_cell_counts
                padding_per_slot = 3 * 8 * int(np.prod(block + 2) - np.prod(block))
                for region in regions:
                    for case in region.get("capacities", [region]):
                        result = case["native"]
                        # Traced call peak already includes workspace, backend scratch
                        # and transient Python allocations; do not add them again.
                        total = metadata + region["selection"]["plan_bytes"] + result["allocated_output_bytes"] + result["tracemalloc"]["peak_delta_bytes"]
                        saving = case["capacity"] * padding_per_slot
                        records.append({"fixture": name, "selection": region["label"],
                            "capacity": case["capacity"], "metadata_array_bytes": metadata,
                            "complete_native_peak_upper_bytes": total,
                            "full_source_payload_bytes": int(np.prod(reader.shape) * 8),
                            "all_padding_removal_optimistic_bytes": saving,
                            "optimistic_padding_memory_ratio": total / (total - saving),
                            "all_output_removal_optimistic_ratio": total / (total - result["allocated_output_bytes"])})
            finally:
                os.close(fd)
    cache_rows = []
    for record in cache["records"]:
        # Include cache and every workspace slot in this deliberately optimistic
        # bound, even though valid primary/entry halos cannot actually be dropped.
        saving = (record["capacity"] + 57) * 3 * 8 * (10**3 - 8**3)
        total = record["complete_controlled_array_peak_bytes"] + record["temporary_index_python_upper_bytes"] + record["native_transfer_scratch_upper_bytes"]
        cache_rows.append({"capacity": record["capacity"], "batch": record["batch"],
                           "complete_controlled_upper_bytes": total,
                           "optimistic_padding_memory_ratio": total / (total - saving)})
    args.output.write_text(json.dumps({"environment": lfe.environment_record(),
        "local": records, "cache": cache_rows,
        "scope": "Controlled arrays and measured/bounded transient storage. RSS, allocator retention and OS page cache are separate; no out-of-core claim."}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
