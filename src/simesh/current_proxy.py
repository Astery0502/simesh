"""Area-weighted, field-line-averaged current proxies on a uniform display grid.

The prescription follows Cheung & DeRosa (2012), ApJ 757, 147, Section 2.4.
It is a morphology proxy, not calibrated radiation or a thermodynamic model.
"""

from dataclasses import dataclass
import numpy as np

from .geometry import PointSet, LineSet, native_bottom_seeds
from .slices import _uniform_geometry
from .tracing import Termination, _validate_vector, _resolve_curl
from .line_profiles import sample_line_profiles
from .connectivity import Boundary
from .fields import require_fields
from .operators.line import _trapezoid


@dataclass(frozen=True)
class CurrentProxyDiagnostics:
    """Owned per-seed geometry, tracing status and proxy acceptance.

    Attributes
    ----------
    points : PointSet
        Seed positions and original IDs.
    seed_areas : ndarray
        Positive quadrature areas in squared coordinate-length units.
    mean_current_squared, length : ndarray
        Mean squared interpolated curl and accepted polyline length. Rejected
        contributions have NaN means. Curl uses field/coordinate-length units.
    closed, accepted : ndarray
        Geometric bottom-to-bottom classification and usable proxy contribution,
        respectively. Closed lines with unusable samples or zero length are not
        accepted; closure alone does not establish a valid emission contribution.
    termination, endpoint_faces : ndarray
        Original against/along tracing statuses and Boundary face IDs. Sampling
        failures do not overwrite tracing status. Unrequested outward branches
        of bottom seeds have known zmin identity.
    """

    points: PointSet
    seed_areas: np.ndarray
    mean_current_squared: np.ndarray
    length: np.ndarray
    closed: np.ndarray
    accepted: np.ndarray
    termination: np.ndarray
    endpoint_faces: np.ndarray


@dataclass(frozen=True)
class CurrentProxyBatch(CurrentProxyDiagnostics):
    """Per-seed diagnostics with detached sparse display-volume contributions.

    Attributes
    ----------
    voxel_ids, increments, visits : ndarray
        Flattened C-order display cells, summed area-times-mean-current-squared
        contributions, and contributing line counts. Each line visits a cell
        at most once. Per-seed attributes are defined by CurrentProxyDiagnostics.
    """

    voxel_ids: np.ndarray
    increments: np.ndarray
    visits: np.ndarray

    @property
    def diagnostics(self):
        """Share only seed diagnostics, without retaining sparse contributions."""
        return CurrentProxyDiagnostics(self.points,self.seed_areas,self.mean_current_squared,
            self.length,self.closed,self.accepted,self.termination,self.endpoint_faces)


@dataclass(frozen=True)
class CurrentProxyResult:
    """Owned uniform proxy volume and per-seed diagnostics.

    Attributes
    ----------
    emissivity, visits : ndarray
        Display volume and contributing-line count, shaped (nx, ny, nz).
        Zero means no accepted seed contribution, not verified empty plasma.
        Emissivity units are curl units squared times coordinate area; no
        radiometric calibration or volume normalization is applied.
    lower, upper : ndarray
        Display bounds in original coordinate units.
    batches : tuple of CurrentProxyDiagnostics
        Per-seed diagnostics, including rejected and incomplete paths. Consumed
        sparse volume increments are not retained. Inputs and output are resident.
    """

    emissivity: np.ndarray
    visits: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    batches: tuple


def _visited_cells(points, lower, upper, shape):
    """Clip polyline segments, split at voxel faces, and deduplicate cells."""
    spacing = (upper-lower)/shape
    start, delta = points[:-1], np.diff(points,axis=0)
    active = np.any(delta != 0,axis=1)
    start, delta = start[active],delta[active]
    near,far = np.zeros(len(start)),np.ones(len(start))
    active = np.ones(len(start),bool)
    for axis in range(3):
        moving = delta[:,axis] != 0
        active &= moving | ((start[:,axis]>=lower[axis])&(start[:,axis]<upper[axis]))
        t0 = np.divide(lower[axis]-start[:,axis],delta[:,axis],out=np.zeros(len(start)),where=moving)
        t1 = np.divide(upper[axis]-start[:,axis],delta[:,axis],out=np.ones(len(start)),where=moving)
        near = np.maximum(near,np.where(moving,np.minimum(t0,t1),0))
        far = np.minimum(far,np.where(moving,np.maximum(t0,t1),1))
    active &= far>near
    delta = delta[active]
    start = start[active]+near[active,None]*delta
    delta *= (far-near)[active,None]
    end = start+delta
    if not len(start):
        return np.empty(0,np.int64)
    first = np.floor((start-lower)/spacing).astype(np.int64)
    last = np.floor((end-lower)/spacing).astype(np.int64)
    # Integration step is bounded by the smallest display spacing.
    if np.max(np.abs(last-first))>1:
        raise ValueError('trace segment exceeds one display cell per axis')
    times = np.ones((len(start),5))
    times[:,0] = 0.
    for axis in range(3):
        cross = first[:,axis] != last[:,axis]
        face = lower[axis]+spacing[axis]*(first[cross,axis]+(delta[cross,axis]>0))
        times[cross,axis+1] = (face-start[cross,axis])/delta[cross,axis]
    times = np.clip(times,0,1)
    times.sort(axis=1)
    middle = .5*(times[:,:-1]+times[:,1:])
    xyz = (start[:,None,:]+middle[:,:,None]*delta[:,None,:])[np.diff(times,axis=1)>1e-13]
    index = np.floor((xyz-lower)/spacing).astype(np.int64)
    good = np.all((index>=0)&(index<shape),axis=1)
    return np.unique(np.ravel_multi_index(index[good].T,tuple(shape)))


def _face(endpoint, mesh, step):
    faces = (Boundary.ZMIN,Boundary.ZMAX,Boundary.YMIN,Boundary.YMAX,Boundary.XMIN,Boundary.XMAX)
    distances = np.array([endpoint[2]-mesh.lower[2],mesh.upper[2]-endpoint[2],
                          endpoint[1]-mesh.lower[1],mesh.upper[1]-endpoint[1],
                          endpoint[0]-mesh.lower[0],mesh.upper[0]-endpoint[0]])
    close = np.flatnonzero(distances<=step*(1+1e-10))
    return faces[close[0]] if len(close)==1 else Boundary.NONE


def _display_geometry(mesh,resolution,bounds,step):
    shape,lower,upper = _uniform_geometry(mesh,resolution,bounds)
    shape = np.asarray(shape,dtype=np.int64)
    if np.any(lower<mesh.lower) or np.any(upper>mesh.upper):
        raise ValueError('display bounds must lie inside the physical domain')
    if not np.isfinite(step) or step<=0 or step>np.min((upper-lower)/shape):
        raise ValueError('step must be positive and no larger than a display cell')
    return shape,lower,upper


def _areas(points,seed_areas):
    areas = np.array(seed_areas,dtype=float,copy=True)
    if areas.shape!=(len(points),) or not np.isfinite(areas).all() or np.any(areas<=0):
        raise ValueError('seed_areas must contain one positive finite area per seed')
    return areas


def _proxy_fields(fields,curl_field):
    _validate_vector(fields)
    require_fields(fields,halo=1,operation='current proxy tracing')
    companion = _resolve_curl(fields,curl_field,True,None)
    require_fields(companion,halo=1,operation='current proxy curl sampling')
    return companion


def _deposit_lines(fields,companion,lines,areas,shape,lower,upper,step,workers):
    count = len(lines.seeds)
    faces = np.full((count,2),Boundary.NONE,np.int64)
    for row in range(count):
        for side in range(2):
            start,stop = lines.offsets[2*row+side:2*row+side+2]
            status = lines.termination[row,side]
            if status==LineSet.NOT_REQUESTED and lines.seeds.positions[row,2]==fields.mesh.lower[2]:
                faces[row,side] = Boundary.ZMIN
            elif status==Termination.DOMAIN_EXIT and stop>start:
                faces[row,side] = _face(lines.positions[stop-1],fields.mesh,step)
    closed = np.all(faces==Boundary.ZMIN,axis=1)
    means, lengths = np.full(count,np.nan),np.zeros(count)
    accepted = np.zeros(count,bool)
    cell_parts,value_parts = [],[]
    # Open and incomplete paths never need curl samples. Reuse the standard
    # profile sampler on the compact closed subset, including its validity checks.
    rows = np.flatnonzero(closed)
    if len(rows):
        selected = lines if closed.all() else lines.select(closed)
        profiles = sample_line_profiles(companion,selected,workers=workers,boundary='native')
        for local,row in enumerate(rows):
            start,middle,stop = selected.offsets[2*local:2*local+3]
            positive = middle+1 if middle>start else middle
            indices = np.r_[np.arange(middle-1,start-1,-1),np.arange(positive,stop)]
            path = selected.positions[indices]
            if len(path)<2:
                continue
            ds = np.linalg.norm(np.diff(path,axis=0),axis=1)
            lengths[row] = ds.sum()
            if lengths[row]<=0 or not profiles.valid[indices].all() or not profiles.finite[indices].all():
                continue
            current = profiles.values[indices]
            with np.errstate(over='ignore',invalid='ignore'):
                j2 = np.einsum('ij,ij->i',current,current)
                mean = _trapezoid(j2,ds)/lengths[row]
                weighted = mean*areas[row]
            if not np.isfinite(mean) or not np.isfinite(weighted):
                continue
            means[row],accepted[row] = mean,True
            cells = _visited_cells(path,lower,upper,shape)
            cell_parts.append(cells)
            value_parts.append(np.full(len(cells),weighted))
    # Report lengths of rejected geometric paths without constructing profiles.
    for row in np.flatnonzero(~closed):
        start,middle,stop = lines.offsets[2*row:2*row+3]
        for a,b in ((start,middle),(middle,stop)):
            lengths[row] += np.linalg.norm(np.diff(lines.positions[a:b],axis=0),axis=1).sum()
    if cell_parts:
        ids,inverse,visits = np.unique(np.concatenate(cell_parts),return_inverse=True,return_counts=True)
        increments = np.bincount(inverse,weights=np.concatenate(value_parts),minlength=len(ids))
    else:
        ids,visits,increments = np.empty(0,np.int64),np.empty(0,np.int64),np.empty(0,float)
    return CurrentProxyBatch(lines.seeds,areas,means,lengths,closed,accepted,lines.termination.copy(),faces,
                             ids,increments,visits)


def current_proxy_from_lines(fields, lines, resolution, *, seed_areas, step,
                             bounds=None, curl_field=None, workers=1):
    """Deposit a current proxy from existing magnetic paths without retracing.

    Parameters
    ----------
    fields : Fields
        Prepared three-component magnetic field matching the stored paths.
    lines : LineSet
        Owned existing trajectories, including original IDs and branch statuses.
    resolution : tuple of int
        Positive display-grid shape, independent of tracing and AMR resolution.
    seed_areas : array-like
        Positive coordinate-area weights, one per seed in lines.seeds order.
    step : float
        Original tracing step cap, used for conservative endpoint classification;
        must not exceed the smallest display cell.
    bounds : pair of array-like, optional
        Deposition box inside the original domain; full paths define the means.
    curl_field : Fields, optional
        Matching raw curl(fields); otherwise computed from two valid halo layers.
    workers : int
        Number of workers used for closed-path profile sampling.

    Returns
    -------
    CurrentProxyBatch
        Sparse contributions and diagnostics for the supplied paths, with the
        same prescription and boundary approximation as iter_current_proxy.

    Notes
    -----
    The caller must establish matching snapshot, coordinates and tracing step
    when reusing loaded paths. IDs or a saved source label cannot verify this.
    This consumer samples only geometrically closed paths and never traces or
    reads a Source. Process large stored LineSets in batches to bound storage.
    """
    if not isinstance(lines,LineSet):
        raise TypeError('lines must be a LineSet')
    shape,lower,upper = _display_geometry(fields.mesh,resolution,bounds,step)
    areas = _areas(lines.seeds,seed_areas)
    companion = _proxy_fields(fields,curl_field)
    return _deposit_lines(fields,companion,lines,areas,shape,lower,upper,step,workers)


def iter_current_proxy(fields, resolution, *, points=None, seed_areas=None,
                       bounds=None, step, step_fraction=.25, max_steps=20000, max_length=np.inf,
                       curl_field=None, seed_batch=128, workers=1):
    """Yield area-weighted current-proxy contributions from explicit prepared fields.

    Parameters
    ----------
    fields : Fields
        Three magnetic components; paths require complete domain support.
        Automatic curl needs two valid halo layers; supplied curl needs one.
    resolution : tuple of int
        Positive (nx, ny, nz) display-grid shape, independent of AMR resolution.
    points : PointSet, optional
        Custom seeds, preserving IDs. Default is native_bottom_seeds(fields.mesh).
        Bottom-face seeds trace inward; other seeds trace both branches.
    seed_areas : array-like, optional
        Positive finite area weights (n,) in squared coordinate units; required
        for custom seeds. Default native seeds use their AMR cell-face areas.
    bounds : pair of array-like, optional
        Display box inside the original domain. Only deposition is clipped;
        full paths determine mean current and physical endpoint classification.
    step : float
        Coordinate-length cap, no greater than the smallest display cell.
    step_fraction : float, optional
        Local cell-size fraction (default 0.25); None selects fixed steps.
    max_steps : int
        Maximum accepted steps per traced branch.
    max_length : float
        Maximum coordinate length per traced branch.
    curl_field : Fields, optional
        Matching raw curl(fields); physical current or unrelated fields rejected.
    seed_batch : int
        Positive number of seeds held as trajectories at once. Inputs remain
        resident; dense output is not allocated by this iterator.
    workers : int
        Number of compute workers.

    Yields
    ------
    CurrentProxyBatch
        Owned sparse volume increments and all seed statuses. A completed batch
        does not imply all paths completed or that seed quadrature converged.

    Notes
    -----
    Following Cheung & DeRosa (2012), each closed line contributes its area times
    the arc-length mean of squared interpolated curl once per visited cell.
    Accepted-prefix trajectories are not localized endpoints: closure requires
    DOMAIN_EXIT and exactly one face within one step of each final point. Corner
    ambiguities and incomplete lines are excluded; check step convergence.
    Seed spacing, weights and display resolution affect this uncalibrated proxy;
    tracing both magnetic polarities does not identify or deduplicate flux tubes.
    """
    from .applications import trace
    shape,lower,upper = _display_geometry(fields.mesh,resolution,bounds,step)
    if type(seed_batch) is not int or seed_batch<1:
        raise ValueError('seed_batch must be positive')
    if points is None:
        if seed_areas is not None:
            raise ValueError('custom seed_areas require explicit points')
        points,seed_areas = native_bottom_seeds(fields.mesh)
    elif not isinstance(points,PointSet) or seed_areas is None:
        raise ValueError('custom PointSet seeds require explicit seed_areas')
    areas = _areas(points,seed_areas)
    companion = _proxy_fields(fields,curl_field)
    for first in range(0,len(points),seed_batch):
        last = min(first+seed_batch,len(points))
        selected = PointSet(points.positions[first:last],ids=points.ids[first:last])
        bottom = selected.positions[:,2]==fields.mesh.lower[2]
        pieces = []
        for on_bottom in (True,False):
            rows = np.flatnonzero(bottom==on_bottom)
            if len(rows):
                group = selected if len(rows)==len(selected) else selected.select(bottom==on_bottom)
                lines = trace(fields,group,direction='inward' if on_bottom else 'both',
                              step=step,step_fraction=step_fraction,max_steps=max_steps,max_length=max_length,
                              workers=workers,seed_batch=seed_batch)
                pieces.append((rows,lines))
        if len(pieces)==1:
            lines = pieces[0][1]
        else:
            # Restore input order once after the two tracing-direction groups.
            counts = np.zeros((len(selected),2),np.int64)
            termination = np.full((len(selected),2),LineSet.NOT_REQUESTED,np.int64)
            branches = [None]*(2*len(selected))
            for rows,lines in pieces:
                termination[rows] = lines.termination
                for local,row in enumerate(rows):
                    for side in range(2):
                        start,stop = lines.offsets[2*local+side:2*local+side+2]
                        branches[2*row+side] = lines.positions[start:stop]
                        counts[row,side] = stop-start
            positions = np.concatenate(branches)
            lines = LineSet(selected,positions,np.r_[np.int64(0),np.cumsum(counts.ravel())],termination,fields.value_identity)
        yield _deposit_lines(fields,companion,lines,areas[first:last].copy(),shape,lower,upper,step,workers)


def current_proxy(fields, resolution, *, step, **controls):
    """Collect iter_current_proxy into a resident display volume.

    Parameters
    ----------
    fields : Fields
        Prepared magnetic input, as defined by iter_current_proxy.
    resolution : tuple of int
        Display volume shape.
    step : float
        Original-coordinate integration step cap, as defined by iter_current_proxy.
    **controls : dict
        Seed, area, bounds, tracing and worker controls of iter_current_proxy.

    Returns
    -------
    CurrentProxyResult
        Owned volume, contribution counts, display bounds and all batch diagnostics.
    """
    shape,lower,upper = _display_geometry(fields.mesh,resolution,controls.get('bounds'),step)
    # Validate and start the iterator before allocating a dense display volume.
    iterator = iter_current_proxy(fields,resolution,step=step,**controls)
    first = next(iterator,None)
    volume,visits = np.zeros(shape),np.zeros(shape,np.int64)
    batches = []
    from itertools import chain
    for batch in chain(() if first is None else (first,),iterator):
        volume.ravel()[batch.voxel_ids] += batch.increments
        visits.ravel()[batch.voxel_ids] += batch.visits
        batches.append(batch.diagnostics)
    return CurrentProxyResult(volume,visits,np.array(lower,copy=True),np.array(upper,copy=True),tuple(batches))
