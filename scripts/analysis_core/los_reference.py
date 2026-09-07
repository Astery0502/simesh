"""Independent scalar LOS oracle: intersect every leaf, split reconstruction knots.

No compiled locator, marcher or sampler is used. Gaussian quadrature is exact
for the supplied trilinear reconstruction on each knot interval, up to roundoff.
"""

import itertools
import numpy as np


def scalar_at(product,leaf,point,component=0):
    slot = product.slot_of_leaf[leaf]
    if slot<0:
        raise ValueError("reference requires the intersected leaf")
    coordinate = (point-product.mesh.bounds[leaf,0])/product.mesh.spacing[leaf]-.5
    base = np.floor(coordinate).astype(int)
    weights = coordinate-base
    base += product.halo
    value = 0.
    for offset in itertools.product((0,1),repeat=3):
        index = base+offset
        if np.any(index<0) or np.any(index>=product.values.shape[1:4]):
            raise ValueError("reference sample is outside the supplied stencil")
        weight = np.prod([weights[a] if offset[a] else 1-weights[a] for a in range(3)])
        value += float(product.values[(slot,*index,component)])*weight
    return value


def ray_integral(product,origin,direction,*,near=0.,far=np.inf,component=0,
                 method="gauss2",step_fraction=.5):
    """direction is the already normalized direction actually used by the caller."""
    mesh = product.mesh
    first = np.full(mesh.leaf_count,near,dtype=float)
    last = np.full(mesh.leaf_count,far,dtype=float)
    valid = np.ones(mesh.leaf_count,dtype=bool)
    for a in range(3):
        if direction[a]==0:
            valid &= (origin[a]>=mesh.bounds[:,0,a])&(origin[a]<mesh.bounds[:,1,a])
        else:
            left = (mesh.bounds[:,0,a]-origin[a])/direction[a]
            right = (mesh.bounds[:,1,a]-origin[a])/direction[a]
            first = np.maximum(first,np.minimum(left,right))
            last = np.minimum(last,np.maximum(left,right))
    leaves = np.flatnonzero(valid&(first<last))
    leaves = leaves[np.argsort(first[leaves])]
    value,depth = 0.,0.
    for leaf in leaves:
        begin,end = first[leaf],last[leaf]
        depth += end-begin
        if method=="midpoint":
            count = max(1,int(np.ceil((end-begin)/(step_fraction*mesh.spacing[leaf].min()))))
            delta = (end-begin)/count
            for i in range(count):
                value += scalar_at(product,leaf,origin+direction*(begin+(i+.5)*delta),component)*delta
        else:
            knots = [begin,end]
            for a,n in enumerate(mesh.block_shape):
                if direction[a]!=0:
                    times = (mesh.bounds[leaf,0,a]+(np.arange(n)+.5)*mesh.spacing[leaf,a]-origin[a])/direction[a]
                    knots.extend(times[(times>begin)&(times<end)].tolist())
            knots = np.unique(knots)
            for a,b in zip(knots[:-1],knots[1:]):
                nodes = a+.5*(b-a)*(1+np.array([-1.,1.])/np.sqrt(3.))
                values = [scalar_at(product,leaf,origin+direction*t,component) for t in nodes]
                value += .5*(b-a)*(values[0]+values[1])
    return value,depth
