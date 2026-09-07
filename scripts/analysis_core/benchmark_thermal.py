"""WENO geometry/rho with explicitly manufactured temperature and unit scales."""
import gc
import json
from pathlib import Path
import resource
import time
import numpy as np
from simesh.analysis import (open_prepared, FieldDefinition, thermal_fields,
    integrate_thermal_los, orthographic_plane, emissivity_fields, integrate_los)
from simesh.analysis.thermal import PROTON_MASS_G
from simesh.amrvac.analysis import prepare_resident
from analysis_core.benchmark_prepared import measure


def main():
    result = {"fixture":"data/weno509_sub_0000.dat", "physical_validation":False,
        "density_unit_g_cm3":1.4*PROTON_MASS_G*1e9,"length_unit_cm":1e8,
        "temperature":"manufactured 0.45--1.65 MK sinusoid along normalized z",
        "interpretation":"demonstration scales, not inferred snapshot units",
        "profile":"8x8 full-domain axis and oblique views; subdivisions 1,4,16; identical nodes for both orders",
        "runs":[]}
    t = time.perf_counter()
    density = open_prepared(result["fixture"],field_names="rho")
    mesh = density.mesh
    result["file_to_density_seconds"] = time.perf_counter()-t
    raw = np.empty((mesh.leaf_count,1,*mesh.block_shape))
    for leaf in range(mesh.leaf_count):
        z = mesh.bounds[leaf,0,2]+(np.arange(mesh.block_shape[2])+.5)*mesh.spacing[leaf,2]
        z = (z-mesh.lower[2])/(mesh.upper[2]-mesh.lower[2])
        raw[leaf,0] = 1.05e6+6e5*np.sin(2*np.pi*z)
    t = time.perf_counter()
    temperature = prepare_resident(mesh,mesh.roots.shape,mesh.node_leaves>=0,raw,(FieldDefinition("external_T","K"),),
                                   budget_bytes=2*1024**3-density.nbytes)
    del raw
    state = thermal_fields(density,temperature,density_unit_g_cm3=result["density_unit_g_cm3"],
                           temperature_label=result["temperature"])
    result["temperature_and_state_seconds"] = time.perf_counter()-t
    result["state_bytes"] = state.nbytes
    result["thermal_admission"] = state.preparation_stats
    del density,temperature
    gc.collect()
    t = time.perf_counter()
    retained_emissivity = emissivity_fields(state)
    result['retained_emissivity_build_seconds'] = time.perf_counter()-t
    result['retained_emissivity_bytes'] = retained_emissivity.nbytes
    for direction in ([0.,0.,1.],[.3,.2,1.]):
        plane = orthographic_plane(mesh.lower,mesh.upper,direction,(8,8))
        images = {}
        rows = []
        for order,n in (("emissivity-first",1),("thermodynamics-first",1),("thermodynamics-first",4),("thermodynamics-first",16),("thermodynamics-first",64)):
            t = time.perf_counter()
            image = integrate_thermal_los(state,plane,direction,length_unit_cm=1e8,order=order,subdivisions=n)
            elapsed = time.perf_counter()-t
            if not image.complete:
                raise AssertionError(np.unique(image.status,return_counts=True))
            images[f"{order}-{n}"] = image.values
            rows.append({"order":order,"subdivisions":n,"seconds":elapsed,"samples":int(image.samples.sum())})
            print("thermal view",direction,order,n,elapsed,flush=True)
        reference = images["thermodynamics-first-64"]
        norm = np.linalg.norm(reference)
        for row in rows:
            error = images[f'{row["order"]}-{row["subdivisions"]}']-reference
            row.update(relative_l2_to_subdivision64=float(np.linalg.norm(error)/norm),max_absolute_difference=float(np.max(np.abs(error))))
        result["runs"].append({"direction":direction,"variants":rows,"image_range":[float(reference.min()),float(reference.max())],
            "retained_emissivity_los_repeat":measure(lambda:integrate_los(retained_emissivity,plane,direction),3),
            "thermodynamics_subdivision4_repeat":measure(lambda:integrate_thermal_los(state,plane,direction,length_unit_cm=1e8,subdivisions=4),3)})
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    Path('benchmark-results/analysis-core/thermal.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
