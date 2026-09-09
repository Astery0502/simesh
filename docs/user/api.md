# User API

Signatures and descriptions below are generated from the Python source.
Core object contracts are in the [developer API](../dev/api.md).

## Input and preparation

::: simesh.open_amrvac

::: simesh.source_from_arrays

::: simesh.source_from_dataset

::: simesh.read_fields

::: simesh.prepare

::: simesh.select_fields

::: simesh.merge_fields

## Sampling and fields

::: simesh.sample

::: simesh.sample_plane

::: simesh.iter_uniform

::: simesh.derive

::: simesh.derive_many

::: simesh.derivative

::: simesh.gradient

::: simesh.divergence

::: simesh.curl

::: simesh.magnitude

::: simesh.dot

::: simesh.MagneticUnits

::: simesh.current_density

::: simesh.magnetic_pressure

::: simesh.magnetic_energy_density

## Geometry

::: simesh.Plane

::: simesh.PointSet
    options:
      members: [from_plane, iter_plane, boundary, select, reshape, nbytes]

::: simesh.RaySet
    options:
      members: [from_plane, select, nbytes]

::: simesh.LineSet
    options:
      members: [branch, line, nbytes]
      merge_init_into_class: false

::: simesh.LineProfile
    options:
      members: [usable]
      merge_init_into_class: false

::: simesh.LineProfiles
    options:
      members: [branch, line, seed_ids, offsets, termination, usable, line_source_identity, nbytes]
      merge_init_into_class: false

## Maps and volumes

::: simesh.applications.sample

::: simesh.applications.field_map

::: simesh.applications.uniform_grid

::: simesh.applications.SampledPoints
    options:
      members: [usable, image, select]
      merge_init_into_class: false

::: simesh.applications.UniformResult
    options:
      members: [usable, axes, spacing]
      merge_init_into_class: false

## Magnetic connectivity and lines

::: simesh.qsl

::: simesh.iter_qsl

::: simesh.line_diagnostics

::: simesh.iter_line_diagnostics

::: simesh.applications.connectivity

::: simesh.applications.iter_connectivity

::: simesh.applications.surface_diagnostics

::: simesh.applications.bottom_diagnostics

::: simesh.applications.trace

::: simesh.applications.iter_lines

::: simesh.sample_line_profiles

::: simesh.iter_line_profiles

::: simesh.QSLResult
    options:
      members: [q_local]
      merge_init_into_class: false

::: simesh.Boundary
    options:
      members: true
      show_if_no_docstring: true
      merge_init_into_class: false

::: simesh.ConnectivityTermination
    options:
      members: true
      show_if_no_docstring: true
      merge_init_into_class: false

::: simesh.applications.ConnectivityMap
    options:
      members: [quantities, q_valid, twist_valid, image, threshold, select]
      merge_init_into_class: false

## MHD and thermodynamics

::: simesh.MHDUnits

::: simesh.IdealMHD

::: simesh.MHDStatus
    options:
      members: true
      show_if_no_docstring: true
      merge_init_into_class: false

::: simesh.MHDStateError
    options:
      members: [leaf_id, cell_index, status]

::: simesh.mhd_fields

::: simesh.CoronalComposition
    options:
      members: [number_density, temperature]

::: simesh.AIA171
    options:
      members: [identity, response, emissivity, from_number_density]

::: simesh.thermal_fields

::: simesh.emissivity_fields

## Integration and statistics

::: simesh.LengthUnits

::: simesh.AxisAlignedSurface

::: simesh.volume_integral

::: simesh.weighted_mean

::: simesh.extrema

::: simesh.histogram

::: simesh.surface_flux

::: simesh.reductions.Coverage
    options:
      members: [fraction, complete, outside_measure, missing_measure, invalid_measure]
      merge_init_into_class: false

::: simesh.reductions.ScalarResult
    options:
      merge_init_into_class: false

::: simesh.reductions.Extremum
    options:
      merge_init_into_class: false

::: simesh.reductions.ExtremaResult
    options:
      merge_init_into_class: false

::: simesh.reductions.HistogramResult
    options:
      merge_init_into_class: false

## Line of sight

::: simesh.orthographic_plane

::: simesh.applications.los

::: simesh.applications.thermal_los

::: simesh.applications.RayResult
    options:
      members: [valid, complete, image, select]
      merge_init_into_class: false

::: simesh.LOSStatus
    options:
      members: true
      show_if_no_docstring: true
      merge_init_into_class: false

## Outputs

::: simesh.save_result

::: simesh.load_result

::: simesh.ResultFile
    options:
      members: [source_verification]
      merge_init_into_class: false

::: simesh.ResultFileError

::: simesh.save_result_shards

::: simesh.open_result_shards

::: simesh.ResultShards
    options:
      members: [seed_ids, complete, load, __len__]
      merge_init_into_class: false

::: simesh.write_amrvac

## Retained interfaces

::: simesh.amrvac.open_dataset

::: simesh.amrvac.read_blocks

::: simesh.amrvac.read_uniform

::: simesh.amrvac.load_from_uniform

::: simesh.amrvac.write_datfile

::: simesh.amrvac.write_datfile_from_uniform

::: simesh.amrvac.load_uniform_data

::: simesh.amrvac.datfile_to_vtk

::: simesh.tools.potential_field_green

::: simesh.tools.PotentialFieldGeometry
    options:
      members: [spacing, cell_center_coordinates]
      merge_init_into_class: false

::: simesh.tools.configurations.bipolar_Avec

::: simesh.tools.configurations.bipolar_Bvec

::: simesh.tools.configurations.rbsl_Avec

::: simesh.tools.configurations.TDm_slab

::: simesh.tools.configurations.dipolez_Avec

::: simesh.tools.configurations.dipole_Bvec

::: simesh.tools.configurations.monopole_Bvec

::: simesh.tools.configurations.fan_Avec

::: simesh.tools.configurations.fan_Bvec

::: simesh.tools.configurations.fan_slab

::: simesh.tools.configurations.curl_slab
