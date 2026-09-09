"""Standard mathematical and explicitly normalized magnetic field diagnostics."""

from dataclasses import dataclass
import math
import numpy as np

from .fields import FieldDefinition, require_fields, _field_index
from .operators.derived import derive
from .operators.derivatives import derivative, _curl_terms
from ._validation import indices, remaining


def _components(fields, components=None, *, count=None):
    require_fields(fields)
    if components is None:
        result = np.arange(len(fields.fields),dtype=np.int64)
    else:
        components = (components,) if isinstance(components,(str,int,np.integer)) else tuple(components)
        result = indices([_field_index(fields.fields,c) if isinstance(c,str) else c for c in components],
                         len(fields.fields),"components")
    if not len(result) or (count is not None and len(result) != count):
        raise ValueError(f"select {'nonempty' if count is None else count} components")
    units = {fields.fields[i].units for i in result}
    if len(units) != 1:
        raise ValueError("selected components must have common units")
    return result, next(iter(units))


def _recipe(inputs, name, function, units, memory_limit):
    groups = tuple(inputs.values()) if isinstance(inputs,dict) else (inputs,)
    halo = min(f.valid_halo for f in groups)
    scratch = 64*math.prod(n+2*halo for n in groups[0].mesh.block_shape)
    return derive(inputs,name,function,units=units,memory_limit=remaining(memory_limit,scratch))


def magnitude(fields, components=None, *, name="magnitude", memory_limit=None):
    """Euclidean norm of selected components, preserving valid support.

    Parameters
    ----------
    fields : Fields
        Completed input fields; see the operation-specific support requirement.
    components : str or int or sequence, optional
        Names or local component indices, in output order.
    name : str, optional
        Output field name.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Independent scalar norm with the common component unit and preserved valid support.
    """
    components,units = _components(fields,components)
    def evaluate(ctx):
        result = np.abs(ctx.field(components[0]))
        for component in components[1:]:
            result = np.hypot(result,ctx.field(component))
        return result
    return _recipe(fields,name,evaluate,units,memory_limit)


def dot(left, right, *, left_components=None, right_components=None,
        name="dot_product", memory_limit=None):
    """Pointwise vector dot product on matching mesh coverage.

    Parameters
    ----------
    left : Fields
        Left vector group.
    right : Fields
        Right group on the same Mesh and coverage.
    left_components : str or int or sequence, optional
        Names or local component indices, in output order.
    right_components : str or int or sequence, optional
        Names or local component indices, in output order.
    name : str, optional
        Output field name.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Scalar dot product; units multiply and common support is retained.
    """
    a,unit_a = _components(left,left_components)
    b,unit_b = _components(right,right_components)
    if len(a) != len(b):
        raise ValueError("dot products require matching component counts")
    def evaluate(ctx):
        result = ctx.field(a[0],group="left")*ctx.field(b[0],group="right")
        for i,j in zip(a[1:],b[1:]):
            result = result+ctx.field(i,group="left")*ctx.field(j,group="right")
        return result
    return _recipe({"left":left,"right":right},name,evaluate,unit_a+" * "+unit_b,memory_limit)


def gradient(fields, component=0, *, name=None, workers=1, memory_limit=None):
    """Compute the centered gradient of a scalar component.

    Parameters
    ----------
    fields : Fields
        Continuous selected components with at least one valid halo.
    component : str or int
        Name or local index of the scalar component.
    name : str, optional
        Output field name.
    workers : int
        Number of workers over disjoint ranges.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Three gradient components in input units per coordinate length, with one fewer
        valid halo.

    Notes
    -----
    To interpolate the derivative afterward, prepare the original input with two valid halo layers.
    """
    selected,units = _components(fields,(component,),count=1)
    component = int(selected[0])
    name = "grad_"+fields.fields[component].name if name is None else name
    definitions = [FieldDefinition(name+"_"+axis,units+" / coordinate-length","centered-derivative") for axis in "xyz"]
    return derivative(fields,[[(component,axis,1.)] for axis in range(3)],definitions,
                      workers=workers,memory_limit=memory_limit)


def divergence(fields, components=(0,1,2), *, name="divergence", workers=1, memory_limit=None):
    """Compute the centered divergence of three ordered vector components.

    Parameters
    ----------
    fields : Fields
        Continuous selected components with at least one valid halo.
    components : sequence of str or int
        Three ordered local vector components with common unit labels.
    name : str, optional
        Output field name.
    workers : int
        Number of workers over disjoint ranges.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Scalar divergence in input units per coordinate length, with one fewer valid
        halo.

    Notes
    -----
    To interpolate the derivative afterward, prepare the original input with two valid halo layers.
    """
    selected,units = _components(fields,components,count=3)
    return derivative(fields,[[(int(component),axis,1.) for axis,component in enumerate(selected)]],
                      [FieldDefinition(name,units+" / coordinate-length","centered-derivative")],
                      workers=workers,memory_limit=memory_limit)


@dataclass(frozen=True)
class MagneticUnits:
    """SI factors per stored field/coordinate unit, with scalar permeability.

    The default is the conventional vacuum approximation 4*pi*1e-7 H/m.
    Supply permeability_h_m explicitly when a different or more precise value
    is required. Metadata labels alone never establish these conversion factors.
    """
    field_tesla: float
    length_m: float
    permeability_h_m: float = 4*np.pi*1e-7

    def __post_init__(self):
        if any(not np.isfinite(value) or value <= 0 for value in
               (self.field_tesla,self.length_m,self.permeability_h_m)):
            raise ValueError("magnetic field, length and permeability factors must be finite and positive")


def current_density(fields, *, units, components=(0,1,2), workers=1, memory_limit=None):
    """Magnetostatic/MHD current in A/m^2, with explicit SI normalization.

    Parameters
    ----------
    fields : Fields
        Completed input fields; see the operation-specific support requirement.
    units : MagneticUnits
        Tesla per stored B, meters per coordinate unit, and uniform permeability in H/m.
    components : str or int or sequence, optional
        Names or local component indices, in output order.
    workers : int
        Number of workers over disjoint ranges.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Three SI current components in A/m², with one fewer valid halo.

    Notes
    -----
    A physically scaled current field is not the raw curl companion required for twist.
    """
    if not isinstance(units,MagneticUnits):
        raise TypeError("units must be a MagneticUnits configuration")
    selected,_ = _components(fields,components,count=3)
    scale = units.field_tesla/units.length_m/units.permeability_h_m
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("current normalization is not representable")
    terms = [[(component,axis,coefficient*scale) for component,axis,coefficient in row]
             for row in _curl_terms(selected)]
    return derivative(fields,terms,[FieldDefinition("j"+axis,"A m^-2","centered-derivative") for axis in "xyz"],
                      workers=workers,memory_limit=memory_limit)


def _magnetic_energy(fields, units, components, name, label, memory_limit):
    if not isinstance(units,MagneticUnits):
        raise TypeError("units must be a MagneticUnits configuration")
    selected,_ = _components(fields,components,count=3)
    def evaluate(ctx):
        norm = np.abs(ctx.field(selected[0])*units.field_tesla)
        for component in selected[1:]:
            norm = np.hypot(norm,ctx.field(component)*units.field_tesla)
        scaled = norm/np.sqrt(units.permeability_h_m)
        return (.5*scaled)*scaled
    return _recipe(fields,name,evaluate,label,memory_limit)


def magnetic_pressure(fields, *, units, components=(0,1,2), memory_limit=None):
    """B^2/(2*mu) in Pa for the supplied uniform scalar permeability.

    Parameters
    ----------
    fields : Fields
        Completed input fields; see the operation-specific support requirement.
    units : MagneticUnits
        Tesla per stored B, meters per coordinate unit, and uniform permeability in H/m.
    components : str or int or sequence, optional
        Names or local component indices, in output order.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Scalar magnetic pressure in Pa, preserving valid support.
    """
    return _magnetic_energy(fields,units,components,"magnetic_pressure","Pa",memory_limit)


def magnetic_energy_density(fields, *, units, components=(0,1,2), memory_limit=None):
    """B^2/(2*mu) in J/m^3 for the supplied uniform scalar permeability.

    Parameters
    ----------
    fields : Fields
        Completed input fields; see the operation-specific support requirement.
    units : MagneticUnits
        Tesla per stored B, meters per coordinate unit, and uniform permeability in H/m.
    components : str or int or sequence, optional
        Names or local component indices, in output order.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Scalar magnetic energy density in J/m³, preserving valid support.
    """
    return _magnetic_energy(fields,units,components,"magnetic_energy_density","J m^-3",memory_limit)
