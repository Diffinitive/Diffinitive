# TODO: Double check that the interfaces for indexing and iterating are fully implemented and tested for all grids.
module Grids

using Sbplib.RegionIndices
using Sbplib.LazyTensors
using StaticArrays
using LinearAlgebra

# Grid
export Grid
export coordinate_size
export component_type
export grid_id
export boundary_id
export boundary_indices
export boundary_identifiers
export boundary_grid
export min_spacing
export coarsen
export refine
export eval_on
export componentview
export ArrayComponentView
export normal

export BoundaryIdentifier
export TensorGridBoundary
export CartesianBoundary

export TensorGrid
export ZeroDimGrid

export EquidistantGrid
export inverse_spacing
export spacing
export equidistant_grid


# MappedGrid
export MappedGrid
export jacobian
export logicalgrid
export mapped_grid
export jacobian_determinant
export metric_tensor
export metric_tensor_inverse

abstract type BoundaryIdentifier end

include("grid.jl")
include("tensor_grid.jl")
include("equidistant_grid.jl")
include("zero_dim_grid.jl")
include("mapped_grid.jl")

end # module
