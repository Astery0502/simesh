# Developer API

Signatures and descriptions below are generated from the Python source.
Scientific and application functions are in the [user API](../user/api.md).

## Core data

::: simesh.FieldDefinition

::: simesh.Mesh
    options:
      members: [leaf_count, children, node_leaves, leaf_nodes, nbytes, locate, descendants]
      merge_init_into_class: false

::: simesh.Selection
    options:
      merge_init_into_class: false

::: simesh.mesh_from_forest

::: simesh.select_region

::: simesh.Source
    options:
      members: [metadata, nbytes, field_ids, read_into, read_native_into, read_footprint, validate, close]
      merge_init_into_class: false

::: simesh.SnapshotMetadata
    options:
      members: [time, iteration, physics_type, field_names, parameters, to_dict, to_header]
      merge_init_into_class: false

::: simesh.Fields
    options:
      members: [values, leaf_ids, nbytes, interior, window]
      merge_init_into_class: false

::: simesh.DerivedContext
    options:
      members: [field]
      merge_init_into_class: false

## Preparation and bounded execution

::: simesh.select_source

::: simesh.cache_source

::: simesh.FillPlan
    options:
      members: [prepare]
      merge_init_into_class: false

::: simesh.plan_preparation

::: simesh.iter_prepared

::: simesh.bounded.PreparedPool
    options:
      members: [borrow, resident_leaf_ids, clear, close]

::: simesh.bounded.CurlPool
    options:
      members: [borrow, clear, close]

::: simesh.bounded.sample_plane_bounded

::: simesh.bounded.iter_uniform_bounded

::: simesh.bounded.trace_bounded

::: simesh.bounded.iter_traces_bounded

::: simesh.bounded.retrace_bounded

::: simesh.bounded.integrate_los_bounded

::: simesh.bounded.integrate_los_views_bounded

::: simesh.global_curl

::: simesh.global_curl_file

## Native consumers

Trajectory and along-line integration interfaces are in the
[user API](../user/api.md#trajectory-and-integral-results).

::: simesh.retrace

::: simesh.SliceResult
    options:
      merge_init_into_class: false

::: simesh.integrate_los

::: simesh.integrate_los_views

::: simesh.integrate_thermal_los

::: simesh.LOSResult
    options:
      members: [depth, valid, complete]
      merge_init_into_class: false

::: simesh.ThermalLOSResult
    options:
      members: [depth_cm]
      merge_init_into_class: false
