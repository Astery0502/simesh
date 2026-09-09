"""Run standard magnetic, selected-line and LOS applications on a synthetic AMR arcade.

Usage: .venv/bin/python examples/standard_applications.py --output /tmp/simesh-demo --plot
Plotting is optional; install the plot extra to use --plot. Physical scales are
illustrative, not a calibration of an observed or simulated solar snapshot.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np


def make_fields():
    import simesh as sm
    mesh = sm.mesh_from_forest((2,1,1),np.array([False]+[True]*9),
                              lower=(-1,-1,0),upper=(1,1,1),block_shape=(16,16,16))
    local = np.indices(mesh.block_shape)+.5
    values = np.empty((mesh.leaf_count,4,*mesh.block_shape))
    k,alpha = 1.2,.6
    decay = np.sqrt(k*k-alpha*alpha)
    for leaf in range(mesh.leaf_count):
        x,y,z = mesh.bounds[leaf,0,:,None,None,None]+local*mesh.spacing[leaf,:,None,None,None]
        values[leaf,:3] = np.array([decay*np.cos(k*x),alpha*np.cos(k*x),-k*np.sin(k*x)])*np.exp(-decay*z)
        values[leaf,3] = 1.+.5*np.exp(-4*(x*x+y*y))*np.exp(-z)
    with sm.source_from_arrays(mesh,values,("b1","b2","b3","rho")) as source:
        magnetic = sm.prepare(source,("b1","b2","b3"),scheme="coordinate-phase")
        density = sm.prepare(source,("rho",),scheme="coordinate-phase")
    return magnetic,density


def run(output, plot):
    import simesh as sm
    from simesh import applications as app
    output.mkdir(parents=True,exist_ok=True)
    magnetic,density = make_fields()
    units = sm.MagneticUnits(field_tesla=1e-3,length_m=1e6)
    current = sm.current_density(magnetic,units=units,workers=4)
    current_norm = sm.magnitude(current)
    pressure = sm.magnetic_pressure(magnetic,units=units)
    div_b = sm.divergence(magnetic,workers=4)
    grid = app.uniform_grid(pressure,(40,40,24),workers=4)

    surface = sm.Plane([-.8,-.2,0],[1.6,0,0],[0,.4,0],(32,24))
    diagnostics = app.surface_diagnostics(magnetic,surface,quantities=("q","twist"),workers=4)
    current_map = app.field_map(current_norm,diagnostics.points,workers=4)
    div_map = app.field_map(div_b,diagnostics.points,workers=4)
    selected = diagnostics.threshold(q_min=4.,abs_twist_min=.09)
    # The default traces both branches; only selected seeds are reintegrated.
    lines = app.trace(magnetic,selected,step=float(magnetic.mesh.spacing.min())*.125,
                      max_steps=4000,workers=4)

    direction = (.3,.2,1.)
    camera = sm.orthographic_plane(density.mesh.lower,density.mesh.upper,direction,(48,48))
    rays = sm.RaySet.from_plane(camera,direction)
    column = app.los(density,rays,workers=4)
    thermal = sm.thermal_fields(density,1e6,density_unit_g_cm3=1e-15,
                                temperature_label="illustrative isothermal 1 MK")
    thermal_image = app.thermal_los(thermal,rays,length_unit_cm=1e8,workers=4)
    np.savez_compressed(output/"products.npz",seed_ids=diagnostics.points.ids,
        q=diagnostics.image("q"),q_valid=diagnostics.image("q_valid"),
        twist=diagnostics.image("twist"),twist_valid=diagnostics.image("twist_valid"),current=current_map.image,
        div_b=div_map.image,pressure=grid.values,pressure_valid=grid.valid,
        selected_ids=lines.seeds.ids,line_positions=lines.positions,line_offsets=lines.offsets,
        line_termination=lines.termination,column=column.image,thermal=thermal_image.image,
        ray_ids=rays.origins.ids,los_status=column.status,thermal_status=thermal_image.status)
    summary = {"field":"synthetic linear force-free AMR arcade", "normalization":"illustrative",
               "magnetic_units":asdict(units),"density_unit_g_cm3":1e-15,"temperature_K":1e6,"length_unit_cm":1e8,
               "surface":{"origin":surface.origin.tolist(),"u":surface.u.tolist(),"v":surface.v.tolist()},
               "camera":{"origin":camera.origin.tolist(),"u":camera.u.tolist(),"v":camera.v.tolist(),
                         "direction":list(direction)},
               "uniform":{"lower":grid.lower.tolist(),"upper":grid.upper.tolist()},
               "column_units":column.units,"thermal_units":thermal_image.units,
               "surface_seeds":len(diagnostics.points),"valid_q":int(diagnostics.data.valid.sum()),
               "selected_lines":len(lines.seeds),"stored_path_points":len(lines.positions),
               "column_valid_rays":int(column.valid.sum()),"thermal_valid_rays":int(thermal_image.valid.sum())}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    if plot:
        render(output)
    print(json.dumps(summary,indent=2))


def render(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    with np.load(output/"products.npz") as archive:
        data = dict(archive)
    summary = json.loads((output/"summary.json").read_text())
    fig = plt.figure(figsize=(12,9),layout="constrained")
    for index,(image,title,label) in enumerate((
        (np.where(data["q_valid"],np.log10(data["q"]),np.nan),"Bottom Q map","log10 Q"),
        (np.where(data["twist_valid"],data["twist"],np.nan),"Bottom twist map","Tw"),
        (data["current"][...,0],"Current magnitude","A / m²")),1):
        ax = fig.add_subplot(2,2,index)
        im = ax.imshow(image.T,origin="lower",extent=(-.8,.8,-.2,.2),aspect="auto",cmap="viridis")
        ax.set(title=title,xlabel="x [coordinate units]",ylabel="y [coordinate units]")
        fig.colorbar(im,ax=ax,label=label)
    ax = fig.add_subplot(2,2,4,projection="3d")
    lookup = {int(seed):row for row,seed in enumerate(data["seed_ids"])}
    color = Normalize(vmin=0.,vmax=float(np.nanmax(np.abs(data["twist"]))))
    for index,seed_id in enumerate(data["selected_ids"]):
        value = abs(data["twist"].ravel()[lookup[int(seed_id)]])
        for side in range(2):
            lo,hi = data["line_offsets"][2*index+side:2*index+side+2]
            points = data["line_positions"][lo:hi]
            ax.plot(*points.T,color=plt.cm.plasma(color(value)),lw=.8,alpha=.75)
    ax.set(title="Selected field lines (both branches)",xlabel="x",ylabel="y",zlabel="z")
    fig.suptitle("Synthetic AMR arcade | illustrative SI normalization")
    fig.savefig(output/"magnetic-applications.png",dpi=150)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    for ax,name,title in zip(axes,("column","thermal"),("Density column","Historical AIA171 | isothermal 1 MK")):
        flags = data["los_status" if name == "column" else "thermal_status"]
        image = np.where(np.isin(flags,(1,2)).reshape(data[name].shape),data[name],np.nan)
        im = ax.imshow(image.T,origin="lower",cmap="inferno")
        ax.set(title=title,xlabel="Image pixel u",ylabel="Image pixel v")
        fig.colorbar(im,ax=ax,label=summary[name+"_units"])
    fig.suptitle("Synthetic AMR LOS | illustrative physical inputs")
    fig.savefig(output/"los-applications.png",dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("application-output"))
    parser.add_argument("--plot",action="store_true")
    parser.add_argument("--render-only",action="store_true",help="Render saved products; only NumPy and Matplotlib are needed")
    args = parser.parse_args()
    render(args.output) if args.render_only else run(args.output,args.plot)
