"""Print small manufactured-field diagnostics without writing data products.

Run with ``.venv/bin/python examples/numerical_quality.py``. Each case uses at
most 4,608 native cells and one worker. Quantities use synthetic coordinate and
field units. Cell-average and cell-center inputs are distinct experiments.

Native errors exclude physical boundaries and distinguish refinement levels
and interface cells. Sampled-field divergence and normal jumps are separate
checks; a small centered div(curl B) is not a solenoidal interpolation guarantee.
The stored-curl comparison uses an analytic reference, not solver output.
"""

import argparse
import json
import numpy as np
import simesh as sm


def centers(mesh, leaf):
    return (mesh.bounds[leaf,0,:,None,None,None] +
            (np.indices(mesh.block_shape)+.5)*mesh.spacing[leaf,:,None,None,None])


def magnetic(xyz, spacing, representation):
    x,y,z = xyz
    values = np.array([x*x+y*z, -2*x*y+z*z, 1+x*x+y*y])
    if representation == "cell-average":
        dx,dy,dz = spacing
        values[0] += dx*dx/12
        values[1] += dz*dz/12
        values[2] += (dx*dx+dy*dy)/12
    return values


def analytic_curl(xyz):
    x,y,z = xyz
    return np.array([2*y-2*z, y-2*x, -2*y-z])


def sample(fields, points):
    values, _, valid = sm.sample(fields, points, workers=1)
    if not valid.all() or not np.isfinite(values).all():
        raise RuntimeError("Manufactured query lacks usable field support")
    return values


def native_errors(fields, current):
    mesh = fields.mesh
    divergence = sm.divergence(fields).interior()
    values = current.interior()
    groups = {}
    for leaf in fields.leaf_ids:
        xyz = centers(mesh, leaf)
        x,y,z = xyz
        interior = (x>.5)&(x<1.5)&(y>.25)&(y<.75)&(z>.25)&(z<.75)
        interface = np.abs(x-1)<mesh.spacing[leaf,0]
        error = np.linalg.norm(values[leaf]-np.moveaxis(analytic_curl(xyz),0,-1),axis=-1)
        level = int(mesh.forest.node_levels[mesh.leaf_nodes[leaf]])
        volume = float(np.prod(mesh.spacing[leaf]))
        for name, mask in (("interface",interior&interface),("away",interior&~interface)):
            count = int(mask.sum())
            if not count:
                continue
            group = groups.setdefault((level,name),dict(cells=0,volume=0.,curl_linf=0.,curl_integral=0.,div_linf=0.))
            group["cells"] += count
            group["volume"] += count*volume
            group["curl_linf"] = max(group["curl_linf"],float(error[mask].max()))
            group["curl_integral"] += float(error[mask].sum())*volume
            group["div_linf"] = max(group["div_linf"],float(np.abs(divergence[leaf,...,0][mask]).max()))
    rows = []
    for (level,part), row in sorted(groups.items()):
        row["curl_l1_volume"] = row.pop("curl_integral")/row["volume"]
        rows.append(dict(level=level,part=part,**row))
    return rows


def sampled_errors(fields, cells):
    face = np.array([[1.,y,z] for y in (.33,.43,.57,.67) for z in (.34,.44,.56,.66)])
    epsilon = float(fields.mesh.spacing.min())*1e-4
    points = np.concatenate([face+np.array([sign*.2/cells,0,0]) for sign in (-1,1)])
    divergence = np.zeros(len(points))
    for axis in range(3):
        delta = np.eye(3)[axis]*epsilon
        divergence += (sample(fields,points+delta)[:,axis]-sample(fields,points-delta)[:,axis])/(2*epsilon)
    jumps = []
    for separation in (1e-7/cells,1e-9/cells):
        delta = np.array([separation,0,0])
        jump = sample(fields,face+delta)-sample(fields,face-delta)
        jumps.append(dict(offset=separation,normal_jump_max=float(np.abs(jump[:,0]).max())))
    return dict(divergence_max=float(np.abs(divergence).max()),difference_step=epsilon,interface_limits=jumps)


def compare_paths(fields, current, stored, cells):
    # Keep interpolated curl away from centers whose derivatives touch a
    # physical boundary, including on the coarsest four-cell blocks.
    shape = (2*cells,cells//2,cells//2)
    step = 1/(2*cells)
    origin = np.array([.5,.375,.375])
    xyz = origin[:,None,None,None]+(np.indices(tuple(n+2 for n in shape))-.5)*step
    points = np.moveaxis(xyz,0,-1).reshape(-1,3)
    b = sample(fields,points).reshape(*(n+2 for n in shape),3)
    jacobian = []
    for axis in range(3):
        lo,hi = [slice(1,-1)]*3,[slice(1,-1)]*3
        lo[axis],hi[axis] = slice(None,-2),slice(2,None)
        jacobian.append((b[tuple(hi)]-b[tuple(lo)])/(2*step))
    reconstructed = np.stack([jacobian[1][...,2]-jacobian[2][...,1],
                              jacobian[2][...,0]-jacobian[0][...,2],
                              jacobian[0][...,1]-jacobian[1][...,0]],axis=-1).reshape(-1,3)
    targets = np.moveaxis(xyz[:,1:-1,1:-1,1:-1],0,-1).reshape(-1,3)
    exact = analytic_curl(targets.T).T
    return {name:float(np.linalg.norm(values-exact,axis=1).max()) for name,values in (
        ("native_curl_then_sample",sample(current,targets)),
        ("sample_B_then_centered_curl",reconstructed),
        ("stored_analytic_curl_then_sample",sample(stored,targets)))}


def amr_report(cells, scheme, representation):
    mesh = sm.mesh_from_forest((2,1,1),np.array([False]+[True]*9),
        lower=(0,0,0),upper=(2,1,1),block_shape=(cells,)*3)
    values = np.empty((9,6,cells,cells,cells))
    for leaf in range(mesh.leaf_count):
        xyz = centers(mesh,leaf)
        values[leaf,:3] = magnetic(xyz,mesh.spacing[leaf],representation)
        values[leaf,3:] = analytic_curl(xyz)
    definitions = tuple(sm.FieldDefinition(name,"synthetic",representation) for name in ("bx","by","bz"))
    definitions += tuple(sm.FieldDefinition(name,"synthetic / coordinate-length",representation) for name in ("jx","jy","jz"))
    with sm.source_from_arrays(mesh,values,definitions,copy=False,boundary="continuous") as source:
        fields = sm.prepare(source,("bx","by","bz"),scheme=scheme,workers=1)
        stored = sm.prepare(source,("jx","jy","jz"),scheme=scheme,workers=1)
    current = sm.curl(fields)
    return dict(cells_per_block=cells,native_cells=9*cells**3,scheme=fields.scheme,
        input_interpretation=representation,boundary="continuous",units="synthetic",
        native=native_errors(fields,current),sampled=sampled_errors(fields,cells),
        path_curl_linf=compare_paths(fields,current,stored,cells))


def q_report(cells, scheme):
    mesh = sm.mesh_from_forest((1,1,1),np.array([True]),lower=(-1,-1,0),upper=(1,1,1),
                              block_shape=(cells,)*3)
    x,y,z = centers(mesh,0)
    values = np.array([[.5*x,-.5*y,np.ones_like(z)]])
    definitions = tuple(sm.FieldDefinition(name,"synthetic","cell-center") for name in ("bx","by","bz"))
    with sm.source_from_arrays(mesh,values,definitions,copy=False,boundary="continuous") as source:
        fields = sm.prepare(source,scheme=scheme,workers=1)
    exact = 2*np.cosh(.75)
    rows = []
    for method in ("variational","variational-interpolant","finite-difference"):
        for fraction in (.25,.125):
            result = sm.qsl(fields,np.array([[0.,0.,.5]]),bounds=([-.75,-.75,.125],[.75,.75,.875]),
                method=method,twist=False,step_fraction=fraction,workers=1)
            if not result.valid.all():
                raise RuntimeError("Manufactured Q mapping did not complete")
            rows.append(dict(method=method,step_fraction=fraction,q=float(result.q[0]),
                             relative_error=float(abs(result.q[0]/exact-1))))
    return dict(cells_per_block=cells,native_cells=cells**3,scheme=fields.scheme,
                input_interpretation="cell-center",analytic_q=float(exact),results=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells",type=int,nargs="+",choices=(4,8),default=(4,8))
    parser.add_argument("--scheme",choices=("exact-phase","coordinate-phase"),default="exact-phase")
    parser.add_argument("--representation",choices=("cell-average","cell-center"),default="cell-average")
    args = parser.parse_args()
    for cells in args.cells:
        print(json.dumps(dict(case="amr",**amr_report(cells,args.scheme,args.representation)),allow_nan=False))
        print(json.dumps(dict(case="q",**q_report(cells,args.scheme)),allow_nan=False))


if __name__ == "__main__":
    main()
