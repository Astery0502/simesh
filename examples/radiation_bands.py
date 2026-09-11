"""Synthesize EUV and radio images from an explicitly isothermal AMRVAC snapshot.

Run from the repository root, for example::

    .venv/bin/python examples/radiation_bands.py data/weno509_sub_0000.dat \
        --temperature-k 1e6 --output example-output/radiation-weno509

The default density and length scales are MHDUnits.solar(); override them for
other simulation normalizations. Temperature is prescribed, not inferred from
an energy column. Each band's native coefficient fields retain AMR resolution.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import simesh as sm
from simesh import applications as app


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with sm.open_amrvac(args.snapshot, fields=['rho']) as source:
        density = sm.prepare(source, scheme='exact-phase')
    mesh = density.mesh
    print(f"Prepared {mesh.leaf_count} AMR leaves", flush=True)
    report = dict(snapshot=str(args.snapshot.resolve()), leaves=mesh.leaf_count,
                  temperature_k=args.temperature_k, density_unit_g_cm3=args.density_unit,
                  length_unit_cm=args.length_unit, pixels=args.pixels, products=[])
    views = {}
    for name, direction in (('axial', [0,0,1]), ('oblique', [1,.4,1])):
        plane = sm.orthographic_plane(mesh.lower, mesh.upper, direction, (args.pixels,args.pixels))
        views[name] = sm.RaySet.from_plane(plane, direction)
    models = [(f'euv{wave}', sm.EUV(wavelength=wave)) for wave in args.wavelengths]
    models.append(('radio', sm.RadioFreeFree(frequency_hz=args.radio_frequency)))
    for band, model in models:
        print(f"Synthesizing {band}", flush=True)
        thermal = sm.thermal_fields(density, args.temperature_k, model=model,
            density_unit_g_cm3=args.density_unit, temperature_label='prescribed isothermal snapshot')
        coefficients = sm.radiation_fields(thermal, model=model,
            absorption=sm.HHeAbsorption() if isinstance(model,sm.EUV) else None)
        del thermal
        for view, rays in views.items():
            result = sm.radiative_los(coefficients, rays, length_unit_cm=args.length_unit,
                                      workers=args.workers)
            if not result.complete:
                raise RuntimeError(f"Incomplete {band}/{view} transfer")
            assert np.all(result.intensity.values >= 0)
            assert np.all(result.intensity.values <= result.thin_intensity.values*(1+1e-12))
            scalar = app.los(coefficients, rays, component='emissivity', workers=args.workers)
            expected_thin = scalar.values*args.length_unit
            scale = max(float(np.max(expected_thin)), np.finfo(float).tiny)
            thin_error = float(np.max(np.abs(expected_thin-result.thin_intensity.values))/scale)
            if thin_error > 5e-3:
                raise RuntimeError(f"Midpoint thin integral needs refinement: {thin_error:g}")
            complete_rows = np.flatnonzero(result.intensity.status == sm.LOSStatus.COMPLETE)
            selected = np.zeros(len(rays.origins), dtype=bool)
            selected[complete_rows[len(complete_rows)//2]] = True
            reference = sm.radiative_los(coefficients, rays.select(selected.reshape(rays.origins.shape)),
                                        length_unit_cm=args.length_unit, implementation='reference')
            np.testing.assert_allclose(result.intensity.values[selected], reference.intensity.values,
                                       rtol=2e-10, atol=1e-80)
            row = dict(band=band, view=view, units=result.intensity.units,
                       complete=result.complete, intensity_max=float(result.intensity.values.max()),
                       tau_max=float(result.optical_depth.values.max()), thin_relative_error=thin_error)
            if band in ('euv171','radio'):
                refined = sm.radiative_los(coefficients,rays,length_unit_cm=args.length_unit,
                                           subdivisions=8,workers=args.workers)
                difference = np.max(np.abs(result.intensity.values-refined.intensity.values))
                row['subdivision_relative_change'] = float(difference/max(float(refined.intensity.values.max()),
                                                                         np.finfo(float).tiny))
                assert refined.complete and row['subdivision_relative_change'] < 5e-3
            if isinstance(model,sm.RadioFreeFree):
                analytic = args.temperature_k*-np.expm1(-result.optical_depth.values)
                np.testing.assert_allclose(result.intensity.values,analytic,rtol=2e-10,atol=1e-12)
            for name in ('intensity','optical_depth','thin_intensity'):
                product = getattr(result,name)
                path = args.output/f'{band}-{view}-{name}.result.npz'
                sm.save_result(path,product,overwrite=True,source={'path':str(args.snapshot.resolve())})
                np.testing.assert_array_equal(sm.load_result(path).result.values,product.values)
            report['products'].append(row)
            print(json.dumps(row), flush=True)
        del coefficients
    (args.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path)
    parser.add_argument('--output',type=Path,default=Path('example-output/radiation-bands'))
    parser.add_argument('--temperature-k',type=float,default=1e6)
    parser.add_argument('--density-unit',type=float,default=sm.MHDUnits.solar().density_g_cm3)
    parser.add_argument('--length-unit',type=float,default=sm.MHDUnits.solar().length_cm)
    parser.add_argument('--pixels',type=int,default=32)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--wavelengths',type=int,nargs='+',default=[94,131,171,193,211,304,335,1354,192,255,263,264])
    parser.add_argument('--radio-frequency',type=float,default=17e9)
    run(parser.parse_args())
