"""Thermal nodes and emissivity built from completed native fields.

These operations preserve valid support and publish owned fields. Response
models evaluate arrays; line-of-sight consumers compose the resulting fields.
"""

import numpy as np

from .emission import AIA171
from ..fields import FieldDefinition, Fields, require_continuous
from .._field_data import PointwiseLayout, require_field_data
from ..fields import _component_indices


def thermal_fields(density,temperature,*,density_unit_g_cm3,model=AIA171(),density_component=0,
                   temperature_component=0,temperature_label,memory_limit=None):
    """Detach number-density and kelvin nodes using the common valid input support.

    Parameters
    ----------
    density : Fields
        Mass-density values with explicit physical conversion.
    temperature : float or Fields
        Positive kelvin scalar or kelvin field covering density on the same Mesh.
    density_unit_g_cm3 : float
        Grams per cubic centimeter per stored mass-density value.
    model : AIA171, EUV or RadioFreeFree
        Selected emitting model, composition and number-density convention. Radio
        thermal nodes are consumed by radiation_fields and radiative_los.
    density_component : str or int
        Mass-density field name or local component index.
    temperature_component : str or int
        Kelvin temperature field name or local component index; unused for a scalar temperature.
    temperature_label : str
        Explicit provenance/interpretation of the kelvin temperature input.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Independent number-density (cm^-3) and temperature (K) with common valid support.
    """
    require_field_data(density)
    if (not isinstance(temperature_label,str) or not temperature_label.strip() or
            not np.isfinite(density_unit_g_cm3) or density_unit_g_cm3<=0):
        raise ValueError("positive density units, component and temperature provenance are required")
    density_component, = _thermal_components(density, (density_component,))
    tproduct=isinstance(temperature,Fields)
    if tproduct:
        require_field_data(temperature)
        temperature_component, = _thermal_components(temperature, (temperature_component,))
        if (temperature.mesh is not density.mesh or
                temperature.fields[temperature_component].units!='K' or
                np.any(temperature.slot_of_leaf[density.leaf_ids]<0)):
            raise ValueError("temperature must be kelvin on the same mesh and prepared coverage")
    else:
        try:
            value=float(temperature)
        except (TypeError,ValueError):
            raise ValueError("temperature must be a positive kelvin scalar or prepared field") from None
        if np.ndim(temperature)!=0 or not np.isfinite(value) or value<=0:
            raise ValueError("temperature must be a positive kelvin scalar or prepared field")
        temperature=value
    layout = PointwiseLayout((density,temperature) if tproduct else (density,), allow_superset=True)
    required = layout.admit(2, memory_limit, "thermal fields", scratch_per_cell=128)
    values = layout.allocate(2)
    for index, arrays in layout.chunks():
        rho = arrays[0][..., density_component]*density_unit_g_cm3
        t = arrays[1][..., temperature_component] if tproduct else temperature
        number_density = model.composition.number_density(rho, convention=model.density_convention)
        model.response(t)
        chunk = np.empty((*rho.shape, 2))
        chunk[...,0], chunk[...,1] = number_density, t
        values[index] = chunk
    definitions=(FieldDefinition("emission_measure_density","cm^-3","prepared-node"),
                 FieldDefinition("temperature","K","prepared-node"))
    return layout.publish(values, definitions, scheme=density.scheme+'/thermal-nodes',
        source=(density.source,model.identity,temperature_label),
        stats={"model":model.identity,"temperature":temperature_label,
               "density_unit_g_cm3":float(density_unit_g_cm3),"controlled_upper_bytes":required})


def _thermal_components(fields, selected=None):
    require_field_data(fields)
    selected = _component_indices(fields, selected)
    if any(fields.fields[i].interpretation.startswith("categorical") for i in selected):
        raise ValueError("thermal operations require continuous physical components")
    return selected


def _check_thermal(fields,model,*,halo=1):
    if halo:
        require_continuous(fields,halo=halo,operation="thermal reconstruction")
    else:
        _thermal_components(fields)
    if (len(fields.fields)!=2 or tuple(f.units for f in fields.fields)!=('cm^-3','K') or
            not isinstance(fields.source,tuple) or len(fields.source)!=3 or fields.source[1]!=model.identity):
        raise ValueError("thermal fields and response model must have matching physical identity")


def emissivity_fields(thermodynamics,*,model=AIA171(),memory_limit=None):
    """Apply response to prepared nodes before interpolation, retaining that order.

    Parameters
    ----------
    thermodynamics : Fields
        Number-density and kelvin fields produced by thermal_fields with the same model.
    model : AIA171 or EUV
        Response matching the supplied thermal fields.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Node emissivity in DN s^-1 pixel^-1 cm^-1, preserving valid support.

    Notes
    -----
    Response-before-interpolation defines emissivity-first reconstruction; it is not interchangeable with thermodynamics-first.
    """
    _check_thermal(thermodynamics,model,halo=0)
    definitions = (FieldDefinition(model.emissivity_name,'DN s^-1 pixel^-1 cm^-1','prepared-node'),)
    return _map_thermal(thermodynamics, definitions,
        lambda data: model.from_number_density(data[...,0], data[...,1])[...,None],
        'response-before-interpolation', memory_limit)


def _map_thermal(thermodynamics, definitions, transform, operation, memory_limit, *,
                 stats=None, scratch_per_node=128):
    layout = PointwiseLayout((thermodynamics,))
    required = layout.admit(len(definitions), memory_limit, operation,
                            scratch_per_cell=scratch_per_node)
    values = layout.allocate(len(definitions))
    for index, (data,) in layout.chunks():
        transformed = transform(data)
        if not np.isfinite(transformed).all() or np.any(transformed < 0):
            raise ValueError("radiation coefficients must be finite and nonnegative")
        values[index] = transformed
    return layout.publish(values, definitions, scheme=thermodynamics.scheme+'/'+operation,
        source=thermodynamics.source,
        stats={**thermodynamics.preparation_stats, **(stats or {}), "controlled_upper_bytes": required})
